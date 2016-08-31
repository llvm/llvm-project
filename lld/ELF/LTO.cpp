//===- LTO.cpp ------------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "LTO.h"
#include "Config.h"
#include "Driver.h"
#include "Error.h"
#include "InputFiles.h"
#include "Symbols.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopPassManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/ParallelCG.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/LTO/legacy/UpdateCompilerUsed.h"
#include "llvm/Linker/IRMover.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;

using namespace lld;
using namespace lld::elf;

// This is for use when debugging LTO.
static void saveBuffer(StringRef Buffer, const Twine &Path) {
  std::error_code EC;
  raw_fd_ostream OS(Path.str(), EC, sys::fs::OpenFlags::F_None);
  if (EC)
    error(EC, "cannot create " + Path);
  OS << Buffer;
}

// This is for use when debugging LTO.
static void saveBCFile(Module &M, const Twine &Path) {
  std::error_code EC;
  raw_fd_ostream OS(Path.str(), EC, sys::fs::OpenFlags::F_None);
  if (EC)
    error(EC, "cannot create " + Path);
  WriteBitcodeToFile(&M, OS, /* ShouldPreserveUseListOrder */ true);
}

static void runNewCustomLtoPasses(Module &M, TargetMachine &TM) {
  PassBuilder PB(&TM);

  AAManager AA;

  // Parse a custom AA pipeline if asked to.
  if (!PB.parseAAPipeline(AA, Config->LtoAAPipeline)) {
    error("Unable to parse AA pipeline description: " + Config->LtoAAPipeline);
    return;
  }

  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  // Register the AA manager first so that our version is the one used.
  FAM.registerPass([&] { return std::move(AA); });

  // Register all the basic analyses with the managers.
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager MPM;
  if (!Config->DisableVerify)
    MPM.addPass(VerifierPass());

  // Now, add all the passes we've been requested to.
  if (!PB.parsePassPipeline(MPM, Config->LtoNewPmPasses)) {
    error("unable to parse pass pipeline description: " +
          Config->LtoNewPmPasses);
    return;
  }

  if (!Config->DisableVerify)
    MPM.addPass(VerifierPass());
  MPM.run(M, MAM);
}

static void runOldLtoPasses(Module &M, TargetMachine &TM) {
  // Note that the gold plugin has a similar piece of code, so
  // it is probably better to move this code to a common place.
  legacy::PassManager LtoPasses;
  LtoPasses.add(createTargetTransformInfoWrapperPass(TM.getTargetIRAnalysis()));
  PassManagerBuilder PMB;
  PMB.LibraryInfo = new TargetLibraryInfoImpl(Triple(TM.getTargetTriple()));
  PMB.Inliner = createFunctionInliningPass();
  PMB.VerifyInput = PMB.VerifyOutput = !Config->DisableVerify;
  PMB.LoopVectorize = true;
  PMB.SLPVectorize = true;
  PMB.OptLevel = Config->LtoO;
  PMB.populateLTOPassManager(LtoPasses);
  LtoPasses.run(M);
}

static void runLTOPasses(Module &M, TargetMachine &TM) {
  if (!Config->LtoNewPmPasses.empty()) {
    // The user explicitly asked for a set of passes to be run.
    // This needs the new PM to work as there's no clean way to
    // pass a set of passes to run in the legacy PM.
    runNewCustomLtoPasses(M, TM);
    if (HasError)
      return;
  } else {
    // Run the 'default' set of LTO passes. This code still uses
    // the legacy PM as the new one is not the default.
    runOldLtoPasses(M, TM);
  }

  if (Config->SaveTemps)
    saveBCFile(M, Config->OutputFile + ".lto.opt.bc");
}

static bool shouldInternalize(const SmallPtrSet<GlobalValue *, 8> &Used,
                              Symbol *S, GlobalValue *GV) {
  if (S->IsUsedInRegularObj || Used.count(GV))
    return false;
  return !S->includeInDynsym();
}

BitcodeCompiler::BitcodeCompiler()
    : Combined(new Module("ld-temp.o", Driver->Context)) {}

static void undefine(Symbol *S) {
  replaceBody<Undefined>(S, S->body()->getName(), STV_DEFAULT, S->body()->Type,
                         nullptr);
}

static void handleUndefinedAsmRefs(const BasicSymbolRef &Sym, GlobalValue *GV,
                                   StringSet<> &AsmUndefinedRefs) {
  // GV associated => not an assembly symbol, bail out.
  if (GV)
    return;

  // This is an undefined reference to a symbol in asm. We put that in
  // compiler.used, so that we can preserve it from being dropped from
  // the output, without necessarily preventing its internalization.
  SmallString<64> Name;
  raw_svector_ostream OS(Name);
  Sym.printName(OS);
  AsmUndefinedRefs.insert(Name.str());
}

void BitcodeCompiler::add(BitcodeFile &F) {
  std::unique_ptr<IRObjectFile> Obj = std::move(F.Obj);
  std::vector<GlobalValue *> Keep;
  unsigned BodyIndex = 0;
  ArrayRef<Symbol *> Syms = F.getSymbols();

  Module &M = Obj->getModule();
  if (M.getDataLayoutStr().empty())
    fatal("invalid bitcode file: " + F.getName() + " has no datalayout");

  // Discard non-compatible debug infos if necessary.
  M.materializeMetadata();
  UpgradeDebugInfo(M);

  // If a symbol appears in @llvm.used, the linker is required
  // to treat the symbol as there is a reference to the symbol
  // that it cannot see. Therefore, we can't internalize.
  SmallPtrSet<GlobalValue *, 8> Used;
  collectUsedGlobalVariables(M, Used, /* CompilerUsed */ false);

  for (const BasicSymbolRef &Sym : Obj->symbols()) {
    uint32_t Flags = Sym.getFlags();
    GlobalValue *GV = Obj->getSymbolGV(Sym.getRawDataRefImpl());
    if (GV && GV->hasAppendingLinkage())
      Keep.push_back(GV);
    if (BitcodeFile::shouldSkip(Flags))
      continue;
    Symbol *S = Syms[BodyIndex++];
    if (GV) {
      if (S->HasUnnamedAddr)
        GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
      else
        GV->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
    }
    if (Flags & BasicSymbolRef::SF_Undefined) {
      handleUndefinedAsmRefs(Sym, GV, AsmUndefinedRefs);
      continue;
    }
    auto *B = dyn_cast<DefinedBitcode>(S->body());
    if (!B || B->file() != &F)
      continue;

    // We collect the set of symbols we want to internalize here
    // and change the linkage after the IRMover executed, i.e. after
    // we imported the symbols and satisfied undefined references
    // to it. We can't just change linkage here because otherwise
    // the IRMover will just rename the symbol.
    if (GV && shouldInternalize(Used, S, GV))
      InternalizedSyms.insert(GV->getName());

    // At this point we know that either the combined LTO object will provide a
    // definition of a symbol, or we will internalize it. In either case, we
    // need to undefine the symbol. In the former case, the real definition
    // needs to be able to replace the original definition without conflicting.
    // In the latter case, we need to allow the combined LTO object to provide a
    // definition with the same name, for example when doing parallel codegen.
    undefine(S);

    if (!GV)
      // Module asm symbol.
      continue;

    switch (GV->getLinkage()) {
    default:
      break;
    case GlobalValue::LinkOnceAnyLinkage:
      GV->setLinkage(GlobalValue::WeakAnyLinkage);
      break;
    case GlobalValue::LinkOnceODRLinkage:
      GV->setLinkage(GlobalValue::WeakODRLinkage);
      break;
    }

    Keep.push_back(GV);
  }

  IRMover Mover(*Combined);
  if (Error E = Mover.move(Obj->takeModule(), Keep,
                           [](GlobalValue &, IRMover::ValueAdder) {})) {
    handleAllErrors(std::move(E), [&](const ErrorInfoBase &EIB) {
      fatal("failed to link module " + F.getName() + ": " + EIB.message());
    });
  }
}

static void internalize(GlobalValue &GV) {
  assert(!GV.hasLocalLinkage() &&
         "Trying to internalize a symbol with local linkage!");
  GV.setLinkage(GlobalValue::InternalLinkage);
}

std::vector<std::unique_ptr<InputFile>> BitcodeCompiler::runSplitCodegen(
    const std::function<std::unique_ptr<TargetMachine>()> &TMFactory) {
  unsigned NumThreads = Config->LtoJobs;
  OwningData.resize(NumThreads);

  std::list<raw_svector_ostream> OSs;
  std::vector<raw_pwrite_stream *> OSPtrs;
  for (SmallString<0> &Obj : OwningData) {
    OSs.emplace_back(Obj);
    OSPtrs.push_back(&OSs.back());
  }

  splitCodeGen(std::move(Combined), OSPtrs, {}, TMFactory);

  std::vector<std::unique_ptr<InputFile>> ObjFiles;
  for (SmallString<0> &Obj : OwningData)
    ObjFiles.push_back(createObjectFile(
        MemoryBufferRef(Obj, "LLD-INTERNAL-combined-lto-object")));

  // If -save-temps is given, we need to save temporary objects to files.
  // This is for debugging.
  if (Config->SaveTemps) {
    if (NumThreads == 1) {
      saveBuffer(OwningData[0], Config->OutputFile + ".lto.o");
    } else {
      for (unsigned I = 0; I < NumThreads; ++I)
        saveBuffer(OwningData[I], Config->OutputFile + Twine(I) + ".lto.o");
    }
  }

  return ObjFiles;
}

// Merge all the bitcode files we have seen, codegen the result
// and return the resulting ObjectFile.
std::vector<std::unique_ptr<InputFile>> BitcodeCompiler::compile() {
  for (const auto &Name : InternalizedSyms) {
    GlobalValue *GV = Combined->getNamedValue(Name.first());
    assert(GV);
    internalize(*GV);
  }

  std::string TheTriple = Combined->getTargetTriple();
  std::string Msg;
  const Target *T = TargetRegistry::lookupTarget(TheTriple, Msg);
  if (!T)
    fatal("target not found: " + Msg);

  // LLD supports the new relocations.
  TargetOptions Options = InitTargetOptionsFromCodeGenFlags();
  Options.RelaxELFRelocations = true;

  auto CreateTargetMachine = [&]() {
    return std::unique_ptr<TargetMachine>(T->createTargetMachine(
        TheTriple, "", "", Options, Config->Pic ? Reloc::PIC_ : Reloc::Static));
  };

  std::unique_ptr<TargetMachine> TM = CreateTargetMachine();

  // Update llvm.compiler.used so that optimizations won't strip
  // off AsmUndefinedReferences.
  updateCompilerUsed(*Combined, *TM, AsmUndefinedRefs);

  if (Config->SaveTemps)
    saveBCFile(*Combined, Config->OutputFile + ".lto.bc");

  runLTOPasses(*Combined, *TM);
  if (HasError)
    return {};

  return runSplitCodegen(CreateTargetMachine);
}
