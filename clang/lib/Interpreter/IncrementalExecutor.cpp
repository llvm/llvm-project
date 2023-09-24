//===--- IncrementalExecutor.cpp - Incremental Execution --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the class which performs incremental code execution.
//
//===----------------------------------------------------------------------===//

#include "IncrementalExecutor.h"
		#include "clang/AST/ASTContext.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/CodeGen/ObjectFilePCHContainerOperations.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Lex/PreprocessorOptions.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Linker/Linker.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Driver/Driver.h"
#include "clang/Interpreter/PartialTranslationUnit.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderGDB.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Host.h"

#include "llvm/Analysis/CallGraph.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/RegionPass.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/ExecutionEngine/Orc/ReOptimizeLayer.h"
#include "llvm/IR/PassManager.h"

using namespace llvm;

// Force linking some of the runtimes that helps attaching to a debugger.
LLVM_ATTRIBUTE_USED void linkComponents() {
  llvm::errs() << (void *)&llvm_orc_registerJITLoaderGDBWrapper
               << (void *)&llvm_orc_registerJITLoaderGDBAllocAction;
}

namespace clang {

static std::string buildOrcRTBasename(const llvm::Triple &TT, bool AddArch) {
  bool IsITANMSVCWindows =
      TT.isWindowsMSVCEnvironment() || TT.isWindowsItaniumEnvironment();
  IsITANMSVCWindows = false;
  const char *Prefix = IsITANMSVCWindows ? "" : "lib";
  const char *Suffix = IsITANMSVCWindows ? ".lib" : ".a";
  std::string ArchAndEnv;
  if (AddArch)
    ArchAndEnv = ("-" + llvm::Triple::getArchTypeName(TT.getArch())).str();
  return (Prefix + Twine("orc_rt") + ArchAndEnv + Suffix).str();
}

static std::string findOrcRuntimePath(const std::vector<const char *> &ClangArgv) {
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts;
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagsBuffer);

  driver::Driver Driver(/*MainBinaryName=*/ClangArgv[0],
                        llvm::sys::getProcessTriple(), Diags);
  Driver.setCheckInputsExist(false);
  llvm::ArrayRef<const char *> RF = llvm::makeArrayRef(ClangArgv);
  std::unique_ptr<driver::Compilation> Compilation(Driver.BuildCompilation(RF));

  auto RuntimePaths = *Compilation->getDefaultToolChain().getRuntimePath();
  auto &TC = Compilation->getDefaultToolChain();
  auto TT = Compilation->getDefaultToolChain().getTriple();
  for (const auto &LibPath : TC.getLibraryPaths()) {
    SmallString<128> P(LibPath);
    llvm::sys::path::append(P, buildOrcRTBasename(TT, false));
    if (Driver.getVFS().exists(P))
      return std::string(P.str());
  }

  SmallString<256> Path(RuntimePaths);
  llvm::sys::path::append(Path, buildOrcRTBasename(TT, true));
  if (Driver.getVFS().exists(Path))
    return Path.str().str();

  return "";
}

static void Optimize(TargetMachine* TM, Triple TargetTriple, llvm::Module& M, StringRef PassPipeline) {
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;

  ModuleAnalysisManager MAM;

  PassInstrumentationCallbacks PIC;
  PrintPassOptions PrintPassOpts;
  PrintPassOpts.Verbose = false;
  PrintPassOpts.SkipAnalyses = false;
  StandardInstrumentations SI(M.getContext(), false,
                              false, PrintPassOpts);
  SI.registerCallbacks(PIC, &MAM);

  PipelineTuningOptions PTO;
  // LoopUnrolling defaults on to true and DisableLoopUnrolling is initialized
  // to false above so we shouldn't necessarily need to check whether or not the
  // option has been enabled.
  PTO.LoopUnrolling = false;
  PTO.UnifiedLTO = false;
  PassBuilder PB;

  // Register all the basic analyses with the managers.
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);


  auto MPM = PB.buildPerModuleDefaultPipeline(OptimizationLevel::O2);
  //MPM.printPipeline(dbgs(), [](StringRef x){return x;});
  MPM.run(M, MAM);
}

std::unique_ptr<llvm::Module> CloneModuleToContext(llvm::Module& Src, LLVMContext& Ctx) {
    SmallVector<char, 1> ClonedModuleBuffer;

      std::set<GlobalValue *> ClonedDefsInSrc;
      ValueToValueMapTy VMap;
      auto Tmp = CloneModule(Src, VMap);

      BitcodeWriter BCWriter(ClonedModuleBuffer);

      BCWriter.writeModule(*Tmp);
      BCWriter.writeSymtab();
      BCWriter.writeStrtab();

    MemoryBufferRef ClonedModuleBufferRef(
        StringRef(ClonedModuleBuffer.data(), ClonedModuleBuffer.size()),
        "cloned module buffer");

    auto ClonedModule = cantFail(
        parseBitcodeFile(ClonedModuleBufferRef, Ctx));
    ClonedModule->setModuleIdentifier(Src.getName());
    return ClonedModule;
}

auto EPC = cantFail(llvm::orc::SelfExecutorProcessControl::Create(
    std::make_shared<llvm::orc::SymbolStringPool>()));

IncrementalExecutor::IncrementalExecutor(llvm::orc::ThreadSafeContext &TSC,
                                         llvm::Error &Err,
                                         const clang::TargetInfo &TI)
    : TSCtx(TSC) {
  using namespace llvm::orc;
  llvm::ErrorAsOutParameter EAO(&Err);

  auto JTMB = JITTargetMachineBuilder(TI.getTriple());
  JTMB.addFeatures(TI.getTargetOpts().Features);
  LLLazyJITBuilder Builder;
  Builder.setJITTargetMachineBuilder(JTMB);
  // Enable debugging of JIT'd code (only works on JITLink for ELF and MachO).
  Builder.setEnableDebuggerSupport(true);

  Builder.setObjectLinkingLayerCreator([&](llvm::orc::ExecutionSession &ES,
                                                  const llvm::Triple &TT) {
    auto L = std::make_unique<llvm::orc::ObjectLinkingLayer>(ES, ES.getExecutorProcessControl().getMemMgr());
    return L;
  });

  Builder.setPlatformSetUp(llvm::orc::ExecutorNativePlatform("/home/sunho/dev/llvm-project/build/lib/clang/18/lib/x86_64-unknown-linux-gnu/liborc_rt.a"));

  if (auto JitOrErr = Builder.create())
    Jit = std::move(*JitOrErr);
  else {
    Err = JitOrErr.takeError();
    return;
  }

  Jit->getReOptimizeLayer().setReoptimizeFunc(
      [&](ReOptimizeLayer &Parent, ReOptMaterializationUnitID MUID,
          unsigned CurVerison, ResourceTrackerSP OldRT, const std::vector<std::pair<uint32_t,uint64_t>>& Profile, ThreadSafeModule &TSM) {
        TSM.withModuleDo([&](llvm::Module &M) {
          dbgs() << "Optimizing ---------------" << "\n";
          dbgs() << "before: " << "\n";
          dbgs() << M << "\n";

          std::set<uint64_t> ToLink;

          DenseMap<uint32_t, std::vector<StringRef>> ProfileData;
    
          for (auto [CID, F] : Profile) {
            if (Parent.FuncAddrToMU.count(ExecutorAddr(F))) {
              auto [MUID, Name] = Parent.FuncAddrToMU[ExecutorAddr(F)];
              ToLink.insert(MUID);
              ProfileData[CID].push_back(Name);
            } else {
              dbgs() << F << "\n";
              assert(false);
            }
          }

          for (auto MUID : ToLink) {
            auto& State = Parent.getMaterializationUnitState(MUID);
            State.getThreadSafeModule().withModuleDo([&](llvm::Module& NM) {
              auto NNM = CloneModuleToContext(NM, M.getContext());
              for (auto& F : *NNM) {
                if (F.isDeclaration()) continue;
                F.setVisibility(GlobalValue::HiddenVisibility);
              }
              Linker::linkModules(M, std::move(NNM));
            });
          }

          for (auto& F : M) {
            if (F.isDeclaration()) continue;
            for (auto& B : F) {
              std::vector<CallInst*> Insts;
              for (auto& I : B) {
                if (auto* Call = dyn_cast<CallInst>(&I)) {
                  if (Call->isIndirectCall()) {
                    Insts.push_back(Call);
                  }
                }
              }
              for (auto* Call : Insts) {
                IRBuilder<> IRB(Call);
                auto* a = Call->getMetadata("call_id");
                if (!a) continue;
                auto* VAM = cast<ValueAsMetadata>(cast<MDNode>(a)->getOperand(0));
                int CallID = cast<ConstantInt>(VAM->getValue())->getSExtValue();
                std::vector<std::pair<BasicBlock*, Value*>> Dones;
                Instruction* IP = Call;
                std::vector<Value*> Args(Call->arg_begin(), Call->arg_end());
                for (auto Name : ProfileData[CallID]) {
                  Value *Cmp = IRB.CreateICmpEQ(Call->getCalledOperand(), M.getFunction(Name));
                  Instruction *IfPart, *ElsePart;
                  SplitBlockAndInsertIfThenElse(Cmp, IP, &IfPart, &ElsePart);
                  IRBuilder<> Builder(IfPart);
                  CallInst* Res = Builder.CreateCall(M.getFunction(Name), Args);
                  InlineFunctionInfo IFI;
                  InlineFunction(*Res, IFI);
                  Dones.push_back({IfPart->getParent(), Res});
                  IP = ElsePart;
                }
                IRBuilder<> Builder(IP);
                Builder.CreateCall(Call->getFunctionType(), Call->getCalledOperand(), Args);
                if (!Call->getFunctionType()->getReturnType()->isVoidTy()) {

                }
                Call->eraseFromParent();
              }
            }
          }
          dbgs() << "inlined: " << "\n";
          dbgs() << M << "\n";

          Optimize(nullptr, Jit->getTargetTriple(), M, "default<O2>");

          dbgs() << "after: " << "\n";
          dbgs() << M << "\n";

        });
        return Error::success();
      });
}

IncrementalExecutor::~IncrementalExecutor() {}

llvm::Error IncrementalExecutor::addModule(PartialTranslationUnit &PTU) {
  llvm::orc::ResourceTrackerSP RT =
      Jit->getMainJITDylib().createResourceTracker();
  ResourceTrackers[&PTU] = RT;

  return Jit->addLazyIRModule(RT, {std::move(PTU.TheModule), TSCtx});
}

llvm::Error IncrementalExecutor::removeModule(PartialTranslationUnit &PTU) {

  llvm::orc::ResourceTrackerSP RT = std::move(ResourceTrackers[&PTU]);
  if (!RT)
    return llvm::Error::success();

  ResourceTrackers.erase(&PTU);
  if (llvm::Error Err = RT->remove())
    return Err;
  return llvm::Error::success();
}

// Clean up the JIT instance.
llvm::Error IncrementalExecutor::cleanUp() {
  // This calls the global dtors of registered modules.
  return Jit->deinitialize(Jit->getMainJITDylib());
}

llvm::Error IncrementalExecutor::runCtors() const {
  return Jit->initialize(Jit->getMainJITDylib());
}

llvm::Expected<llvm::orc::ExecutorAddr>
IncrementalExecutor::getSymbolAddress(llvm::StringRef Name,
                                      SymbolNameKind NameKind) const {
  using namespace llvm::orc;
  auto SO = makeJITDylibSearchOrder({&Jit->getMainJITDylib(),
                                     Jit->getPlatformJITDylib().get(),
                                     Jit->getProcessSymbolsJITDylib().get()});

  ExecutionSession &ES = Jit->getExecutionSession();

  auto SymOrErr =
      ES.lookup(SO, (NameKind == LinkerName) ? ES.intern(Name)
                                             : Jit->mangleAndIntern(Name));
  if (auto Err = SymOrErr.takeError())
    return std::move(Err);
  return SymOrErr->getAddress();
}

} // end namespace clang
