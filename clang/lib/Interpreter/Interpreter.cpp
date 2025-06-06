//===------ Interpreter.cpp - Incremental Compilation and Execution -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the component which performs incremental code
// compilation and execution.
//
//===----------------------------------------------------------------------===//

#include "DeviceOffload.h"
#include "IncrementalExecutor.h"
#include "IncrementalParser.h"
#include "InterpreterUtils.h"
#include "llvm/Support/VirtualFileSystem.h"
#ifdef __EMSCRIPTEN__
#include "Wasm.h"
#include <dlfcn.h>
#endif // __EMSCRIPTEN__

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/TypeVisitor.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/CodeGen/ObjectFilePCHContainerWriter.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/FrontendTool/Utils.h"
#include "clang/Interpreter/Interpreter.h"
#include "clang/Interpreter/Value.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Sema/Lookup.h"
#include "clang/Serialization/ObjectFilePCHContainerReader.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Transforms/Utils/Cloning.h" // for CloneModule

#define DEBUG_TYPE "clang-repl"

using namespace clang;
// FIXME: Figure out how to unify with namespace init_convenience from
//        tools/clang-import-test/clang-import-test.cpp
namespace {
/// Retrieves the clang CC1 specific flags out of the compilation's jobs.
/// \returns NULL on error.
static llvm::Expected<const llvm::opt::ArgStringList *>
GetCC1Arguments(DiagnosticsEngine *Diagnostics,
                driver::Compilation *Compilation) {
  // We expect to get back exactly one Command job, if we didn't something
  // failed. Extract that job from the Compilation.
  const driver::JobList &Jobs = Compilation->getJobs();
  if (!Jobs.size() || !isa<driver::Command>(*Jobs.begin()))
    return llvm::createStringError(llvm::errc::not_supported,
                                   "Driver initialization failed. "
                                   "Unable to create a driver job");

  // The one job we find should be to invoke clang again.
  const driver::Command *Cmd = cast<driver::Command>(&(*Jobs.begin()));
  if (llvm::StringRef(Cmd->getCreator().getName()) != "clang")
    return llvm::createStringError(llvm::errc::not_supported,
                                   "Driver initialization failed");

  return &Cmd->getArguments();
}

static llvm::Expected<std::unique_ptr<CompilerInstance>>
CreateCI(const llvm::opt::ArgStringList &Argv) {
  std::unique_ptr<CompilerInstance> Clang(new CompilerInstance());
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());

  // Register the support for object-file-wrapped Clang modules.
  // FIXME: Clang should register these container operations automatically.
  auto PCHOps = Clang->getPCHContainerOperations();
  PCHOps->registerWriter(std::make_unique<ObjectFilePCHContainerWriter>());
  PCHOps->registerReader(std::make_unique<ObjectFilePCHContainerReader>());

  // Buffer diagnostics from argument parsing so that we can output them using
  // a well formed diagnostic object.
  DiagnosticOptions DiagOpts;
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  DiagnosticsEngine Diags(DiagID, DiagOpts, DiagsBuffer);
  bool Success = CompilerInvocation::CreateFromArgs(
      Clang->getInvocation(), llvm::ArrayRef(Argv.begin(), Argv.size()), Diags);

  // Infer the builtin include path if unspecified.
  if (Clang->getHeaderSearchOpts().UseBuiltinIncludes &&
      Clang->getHeaderSearchOpts().ResourceDir.empty())
    Clang->getHeaderSearchOpts().ResourceDir =
        CompilerInvocation::GetResourcesPath(Argv[0], nullptr);

  // Create the actual diagnostics engine.
  Clang->createDiagnostics(*llvm::vfs::getRealFileSystem());
  if (!Clang->hasDiagnostics())
    return llvm::createStringError(llvm::errc::not_supported,
                                   "Initialization failed. "
                                   "Unable to create diagnostics engine");

  DiagsBuffer->FlushDiagnostics(Clang->getDiagnostics());
  if (!Success)
    return llvm::createStringError(llvm::errc::not_supported,
                                   "Initialization failed. "
                                   "Unable to flush diagnostics");

  // FIXME: Merge with CompilerInstance::ExecuteAction.
  llvm::MemoryBuffer *MB = llvm::MemoryBuffer::getMemBuffer("").release();
  Clang->getPreprocessorOpts().addRemappedFile("<<< inputs >>>", MB);

  Clang->setTarget(TargetInfo::CreateTargetInfo(
      Clang->getDiagnostics(), Clang->getInvocation().getTargetOpts()));
  if (!Clang->hasTarget())
    return llvm::createStringError(llvm::errc::not_supported,
                                   "Initialization failed. "
                                   "Target is missing");

  Clang->getTarget().adjust(Clang->getDiagnostics(), Clang->getLangOpts());

  // Don't clear the AST before backend codegen since we do codegen multiple
  // times, reusing the same AST.
  Clang->getCodeGenOpts().ClearASTBeforeBackend = false;

  Clang->getFrontendOpts().DisableFree = false;
  Clang->getCodeGenOpts().DisableFree = false;
  return std::move(Clang);
}

} // anonymous namespace

namespace clang {

llvm::Expected<std::unique_ptr<CompilerInstance>>
IncrementalCompilerBuilder::create(std::string TT,
                                   std::vector<const char *> &ClangArgv) {

  // If we don't know ClangArgv0 or the address of main() at this point, try
  // to guess it anyway (it's possible on some platforms).
  std::string MainExecutableName =
      llvm::sys::fs::getMainExecutable(nullptr, nullptr);

  ClangArgv.insert(ClangArgv.begin(), MainExecutableName.c_str());

  // Prepending -c to force the driver to do something if no action was
  // specified. By prepending we allow users to override the default
  // action and use other actions in incremental mode.
  // FIXME: Print proper driver diagnostics if the driver flags are wrong.
  // We do C++ by default; append right after argv[0] if no "-x" given
  ClangArgv.insert(ClangArgv.end(), "-Xclang");
  ClangArgv.insert(ClangArgv.end(), "-fincremental-extensions");
  ClangArgv.insert(ClangArgv.end(), "-c");

  // Put a dummy C++ file on to ensure there's at least one compile job for the
  // driver to construct.
  ClangArgv.push_back("<<< inputs >>>");

  // Buffer diagnostics from argument parsing so that we can output them using a
  // well formed diagnostic object.
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  std::unique_ptr<DiagnosticOptions> DiagOpts =
      CreateAndPopulateDiagOpts(ClangArgv);
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  DiagnosticsEngine Diags(DiagID, *DiagOpts, DiagsBuffer);

  driver::Driver Driver(/*MainBinaryName=*/ClangArgv[0], TT, Diags);
  Driver.setCheckInputsExist(false); // the input comes from mem buffers
  llvm::ArrayRef<const char *> RF = llvm::ArrayRef(ClangArgv);
  std::unique_ptr<driver::Compilation> Compilation(Driver.BuildCompilation(RF));

  if (Compilation->getArgs().hasArg(driver::options::OPT_v))
    Compilation->getJobs().Print(llvm::errs(), "\n", /*Quote=*/false);

  auto ErrOrCC1Args = GetCC1Arguments(&Diags, Compilation.get());
  if (auto Err = ErrOrCC1Args.takeError())
    return std::move(Err);

  return CreateCI(**ErrOrCC1Args);
}

llvm::Expected<std::unique_ptr<CompilerInstance>>
IncrementalCompilerBuilder::CreateCpp() {
  std::vector<const char *> Argv;
  Argv.reserve(5 + 1 + UserArgs.size());
  Argv.push_back("-xc++");
#ifdef __EMSCRIPTEN__
  Argv.push_back("-target");
  Argv.push_back("wasm32-unknown-emscripten");
  Argv.push_back("-fvisibility=default");
#endif
  llvm::append_range(Argv, UserArgs);

  std::string TT = TargetTriple ? *TargetTriple : llvm::sys::getProcessTriple();
  return IncrementalCompilerBuilder::create(TT, Argv);
}

llvm::Expected<std::unique_ptr<CompilerInstance>>
IncrementalCompilerBuilder::createCuda(bool device) {
  std::vector<const char *> Argv;
  Argv.reserve(5 + 4 + UserArgs.size());

  Argv.push_back("-xcuda");
  if (device)
    Argv.push_back("--cuda-device-only");
  else
    Argv.push_back("--cuda-host-only");

  std::string SDKPathArg = "--cuda-path=";
  if (!CudaSDKPath.empty()) {
    SDKPathArg += CudaSDKPath;
    Argv.push_back(SDKPathArg.c_str());
  }

  std::string ArchArg = "--offload-arch=";
  if (!OffloadArch.empty()) {
    ArchArg += OffloadArch;
    Argv.push_back(ArchArg.c_str());
  }

  llvm::append_range(Argv, UserArgs);

  std::string TT = TargetTriple ? *TargetTriple : llvm::sys::getProcessTriple();
  return IncrementalCompilerBuilder::create(TT, Argv);
}

llvm::Expected<std::unique_ptr<CompilerInstance>>
IncrementalCompilerBuilder::CreateCudaDevice() {
  return IncrementalCompilerBuilder::createCuda(true);
}

llvm::Expected<std::unique_ptr<CompilerInstance>>
IncrementalCompilerBuilder::CreateCudaHost() {
  return IncrementalCompilerBuilder::createCuda(false);
}

class InProcessPrintingASTConsumer final : public MultiplexConsumer {
  Interpreter &Interp;

public:
  InProcessPrintingASTConsumer(std::unique_ptr<ASTConsumer> C, Interpreter &I)
      : MultiplexConsumer(std::move(C)), Interp(I) {}
  bool HandleTopLevelDecl(DeclGroupRef DGR) override final {
    if (DGR.isNull())
      return true;

    for (Decl *D : DGR)
      if (auto *TLSD = llvm::dyn_cast<TopLevelStmtDecl>(D))
        if (TLSD && TLSD->isSemiMissing()) {
          auto ExprOrErr =
              Interp.ExtractValueFromExpr(cast<Expr>(TLSD->getStmt()));
          if (llvm::Error E = ExprOrErr.takeError()) {
            llvm::logAllUnhandledErrors(std::move(E), llvm::errs(),
                                        "Value printing failed: ");
            return false; // abort parsing
          }
          TLSD->setStmt(*ExprOrErr);
        }

    return MultiplexConsumer::HandleTopLevelDecl(DGR);
  }
};

/// A custom action enabling the incremental processing functionality.
///
/// The usual \p FrontendAction expects one call to ExecuteAction and once it
/// sees a call to \p EndSourceFile it deletes some of the important objects
/// such as \p Preprocessor and \p Sema assuming no further input will come.
///
/// \p IncrementalAction ensures it keep its underlying action's objects alive
/// as long as the \p IncrementalParser needs them.
///
class IncrementalAction : public WrapperFrontendAction {
private:
  bool IsTerminating = false;
  Interpreter &Interp;
  std::unique_ptr<ASTConsumer> Consumer;

public:
  IncrementalAction(CompilerInstance &CI, llvm::LLVMContext &LLVMCtx,
                    llvm::Error &Err, Interpreter &I,
                    std::unique_ptr<ASTConsumer> Consumer = nullptr)
      : WrapperFrontendAction([&]() {
          llvm::ErrorAsOutParameter EAO(&Err);
          std::unique_ptr<FrontendAction> Act;
          switch (CI.getFrontendOpts().ProgramAction) {
          default:
            Err = llvm::createStringError(
                std::errc::state_not_recoverable,
                "Driver initialization failed. "
                "Incremental mode for action %d is not supported",
                CI.getFrontendOpts().ProgramAction);
            return Act;
          case frontend::ASTDump:
          case frontend::ASTPrint:
          case frontend::ParseSyntaxOnly:
            Act = CreateFrontendAction(CI);
            break;
          case frontend::PluginAction:
          case frontend::EmitAssembly:
          case frontend::EmitBC:
          case frontend::EmitObj:
          case frontend::PrintPreprocessedInput:
          case frontend::EmitLLVMOnly:
            Act.reset(new EmitLLVMOnlyAction(&LLVMCtx));
            break;
          }
          return Act;
        }()),
        Interp(I), Consumer(std::move(Consumer)) {}
  FrontendAction *getWrapped() const { return WrappedAction.get(); }
  TranslationUnitKind getTranslationUnitKind() override {
    return TU_Incremental;
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    std::unique_ptr<ASTConsumer> C =
        WrapperFrontendAction::CreateASTConsumer(CI, InFile);

    if (Consumer) {
      std::vector<std::unique_ptr<ASTConsumer>> Cs;
      Cs.push_back(std::move(Consumer));
      Cs.push_back(std::move(C));
      return std::make_unique<MultiplexConsumer>(std::move(Cs));
    }

    return std::make_unique<InProcessPrintingASTConsumer>(std::move(C), Interp);
  }

  void ExecuteAction() override {
    WrapperFrontendAction::ExecuteAction();
    getCompilerInstance().getSema().CurContext = nullptr;
  }

  // Do not terminate after processing the input. This allows us to keep various
  // clang objects alive and to incrementally grow the current TU.
  void EndSourceFile() override {
    // The WrappedAction can be nullptr if we issued an error in the ctor.
    if (IsTerminating && getWrapped())
      WrapperFrontendAction::EndSourceFile();
  }

  void FinalizeAction() {
    assert(!IsTerminating && "Already finalized!");
    IsTerminating = true;
    EndSourceFile();
  }
};

Interpreter::Interpreter(std::unique_ptr<CompilerInstance> Instance,
                         llvm::Error &ErrOut,
                         std::unique_ptr<llvm::orc::LLJITBuilder> JITBuilder,
                         std::unique_ptr<clang::ASTConsumer> Consumer)
    : JITBuilder(std::move(JITBuilder)) {
  CI = std::move(Instance);
  llvm::ErrorAsOutParameter EAO(&ErrOut);
  auto LLVMCtx = std::make_unique<llvm::LLVMContext>();
  TSCtx = std::make_unique<llvm::orc::ThreadSafeContext>(std::move(LLVMCtx));

  Act = std::make_unique<IncrementalAction>(*CI, *TSCtx->getContext(), ErrOut,
                                            *this, std::move(Consumer));
  if (ErrOut)
    return;
  CI->ExecuteAction(*Act);

  IncrParser = std::make_unique<IncrementalParser>(*CI, ErrOut);

  if (ErrOut)
    return;

  if (getCodeGen()) {
    CachedInCodeGenModule = GenModule();
    // The initial PTU is filled by `-include` or by CUDA includes
    // automatically.
    if (!CI->getPreprocessorOpts().Includes.empty()) {
      // We can't really directly pass the CachedInCodeGenModule to the Jit
      // because it will steal it, causing dangling references as explained in
      // Interpreter::Execute
      auto M = llvm::CloneModule(*CachedInCodeGenModule);
      ASTContext &C = CI->getASTContext();
      RegisterPTU(C.getTranslationUnitDecl(), std::move(M));
    }
    if (llvm::Error Err = CreateExecutor()) {
      ErrOut = joinErrors(std::move(ErrOut), std::move(Err));
      return;
    }
  }

  // Not all frontends support code-generation, e.g. ast-dump actions don't
  if (getCodeGen()) {
    // Process the PTUs that came from initialization. For example -include will
    // give us a header that's processed at initialization of the preprocessor.
    for (PartialTranslationUnit &PTU : PTUs)
      if (llvm::Error Err = Execute(PTU)) {
        ErrOut = joinErrors(std::move(ErrOut), std::move(Err));
        return;
      }
  }
}

Interpreter::~Interpreter() {
  IncrParser.reset();
  Act->FinalizeAction();
  if (DeviceParser)
    DeviceParser.reset();
  if (DeviceAct)
    DeviceAct->FinalizeAction();
  if (IncrExecutor) {
    if (llvm::Error Err = IncrExecutor->cleanUp())
      llvm::report_fatal_error(
          llvm::Twine("Failed to clean up IncrementalExecutor: ") +
          toString(std::move(Err)));
  }
}

// These better to put in a runtime header but we can't. This is because we
// can't find the precise resource directory in unittests so we have to hard
// code them.
const char *const Runtimes = R"(
    #define __CLANG_REPL__ 1
#ifdef __cplusplus
    #define EXTERN_C extern "C"
    void *__clang_Interpreter_SetValueWithAlloc(void*, void*, void*);
    struct __clang_Interpreter_NewTag{} __ci_newtag;
    void* operator new(__SIZE_TYPE__, void* __p, __clang_Interpreter_NewTag) noexcept;
    template <class T, class = T (*)() /*disable for arrays*/>
    void __clang_Interpreter_SetValueCopyArr(T* Src, void* Placement, unsigned long Size) {
      for (auto Idx = 0; Idx < Size; ++Idx)
        new ((void*)(((T*)Placement) + Idx), __ci_newtag) T(Src[Idx]);
    }
    template <class T, unsigned long N>
    void __clang_Interpreter_SetValueCopyArr(const T (*Src)[N], void* Placement, unsigned long Size) {
      __clang_Interpreter_SetValueCopyArr(Src[0], Placement, Size);
    }
#else
    #define EXTERN_C extern
#endif // __cplusplus

  EXTERN_C void __clang_Interpreter_SetValueNoAlloc(void *This, void *OutVal, void *OpaqueType, ...);
)";

llvm::Expected<std::unique_ptr<Interpreter>>
Interpreter::create(std::unique_ptr<CompilerInstance> CI) {
  llvm::Error Err = llvm::Error::success();
  auto Interp =
      std::unique_ptr<Interpreter>(new Interpreter(std::move(CI), Err));
  if (Err)
    return std::move(Err);

  // Add runtime code and set a marker to hide it from user code. Undo will not
  // go through that.
  auto PTU = Interp->Parse(Runtimes);
  if (!PTU)
    return PTU.takeError();
  Interp->markUserCodeStart();

  Interp->ValuePrintingInfo.resize(4);
  return std::move(Interp);
}

llvm::Expected<std::unique_ptr<Interpreter>>
Interpreter::createWithCUDA(std::unique_ptr<CompilerInstance> CI,
                            std::unique_ptr<CompilerInstance> DCI) {
  // avoid writing fat binary to disk using an in-memory virtual file system
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> IMVFS =
      std::make_unique<llvm::vfs::InMemoryFileSystem>();
  llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayVFS =
      std::make_unique<llvm::vfs::OverlayFileSystem>(
          llvm::vfs::getRealFileSystem());
  OverlayVFS->pushOverlay(IMVFS);
  CI->createFileManager(OverlayVFS);

  llvm::Expected<std::unique_ptr<Interpreter>> InterpOrErr =
      Interpreter::create(std::move(CI));
  if (!InterpOrErr)
    return InterpOrErr;

  std::unique_ptr<Interpreter> Interp = std::move(*InterpOrErr);

  llvm::Error Err = llvm::Error::success();
  llvm::LLVMContext &LLVMCtx = *Interp->TSCtx->getContext();

  auto DeviceAct =
      std::make_unique<IncrementalAction>(*DCI, LLVMCtx, Err, *Interp);

  if (Err)
    return std::move(Err);

  Interp->DeviceAct = std::move(DeviceAct);

  DCI->ExecuteAction(*Interp->DeviceAct);

  Interp->DeviceCI = std::move(DCI);

  auto DeviceParser = std::make_unique<IncrementalCUDADeviceParser>(
      *Interp->DeviceCI, *Interp->getCompilerInstance(), IMVFS, Err,
      Interp->PTUs);

  if (Err)
    return std::move(Err);

  Interp->DeviceParser = std::move(DeviceParser);
  return std::move(Interp);
}

const CompilerInstance *Interpreter::getCompilerInstance() const {
  return CI.get();
}

CompilerInstance *Interpreter::getCompilerInstance() { return CI.get(); }

llvm::Expected<llvm::orc::LLJIT &> Interpreter::getExecutionEngine() {
  if (!IncrExecutor) {
    if (auto Err = CreateExecutor())
      return std::move(Err);
  }

  return IncrExecutor->GetExecutionEngine();
}

ASTContext &Interpreter::getASTContext() {
  return getCompilerInstance()->getASTContext();
}

const ASTContext &Interpreter::getASTContext() const {
  return getCompilerInstance()->getASTContext();
}

void Interpreter::markUserCodeStart() {
  assert(!InitPTUSize && "We only do this once");
  InitPTUSize = PTUs.size();
}

size_t Interpreter::getEffectivePTUSize() const {
  assert(PTUs.size() >= InitPTUSize && "empty PTU list?");
  return PTUs.size() - InitPTUSize;
}

PartialTranslationUnit &
Interpreter::RegisterPTU(TranslationUnitDecl *TU,
                         std::unique_ptr<llvm::Module> M /*={}*/,
                         IncrementalAction *Action) {
  PTUs.emplace_back(PartialTranslationUnit());
  PartialTranslationUnit &LastPTU = PTUs.back();
  LastPTU.TUPart = TU;

  if (!M)
    M = GenModule(Action);

  assert((!getCodeGen(Action) || M) &&
         "Must have a llvm::Module at this point");

  LastPTU.TheModule = std::move(M);
  LLVM_DEBUG(llvm::dbgs() << "compile-ptu " << PTUs.size() - 1
                          << ": [TU=" << LastPTU.TUPart);
  if (LastPTU.TheModule)
    LLVM_DEBUG(llvm::dbgs() << ", M=" << LastPTU.TheModule.get() << " ("
                            << LastPTU.TheModule->getName() << ")");
  LLVM_DEBUG(llvm::dbgs() << "]\n");
  return LastPTU;
}

llvm::Expected<PartialTranslationUnit &>
Interpreter::Parse(llvm::StringRef Code) {
  // If we have a device parser, parse it first. The generated code will be
  // included in the host compilation
  if (DeviceParser) {
    llvm::Expected<TranslationUnitDecl *> DeviceTU = DeviceParser->Parse(Code);
    if (auto E = DeviceTU.takeError())
      return std::move(E);

    RegisterPTU(*DeviceTU, nullptr, DeviceAct.get());

    llvm::Expected<llvm::StringRef> PTX = DeviceParser->GeneratePTX();
    if (!PTX)
      return PTX.takeError();

    llvm::Error Err = DeviceParser->GenerateFatbinary();
    if (Err)
      return std::move(Err);
  }

  // Tell the interpreter sliently ignore unused expressions since value
  // printing could cause it.
  getCompilerInstance()->getDiagnostics().setSeverity(
      clang::diag::warn_unused_expr, diag::Severity::Ignored, SourceLocation());

  llvm::Expected<TranslationUnitDecl *> TuOrErr = IncrParser->Parse(Code);
  if (!TuOrErr)
    return TuOrErr.takeError();

  return RegisterPTU(*TuOrErr);
}

static llvm::Expected<llvm::orc::JITTargetMachineBuilder>
createJITTargetMachineBuilder(const std::string &TT) {
  if (TT == llvm::sys::getProcessTriple())
    // This fails immediately if the target backend is not registered
    return llvm::orc::JITTargetMachineBuilder::detectHost();

  // If the target backend is not registered, LLJITBuilder::create() will fail
  return llvm::orc::JITTargetMachineBuilder(llvm::Triple(TT));
}

llvm::Error Interpreter::CreateExecutor() {
  if (IncrExecutor)
    return llvm::make_error<llvm::StringError>("Operation failed. "
                                               "Execution engine exists",
                                               std::error_code());
  if (!getCodeGen())
    return llvm::make_error<llvm::StringError>("Operation failed. "
                                               "No code generator available",
                                               std::error_code());
  if (!JITBuilder) {
    const std::string &TT = getCompilerInstance()->getTargetOpts().Triple;
    auto JTMB = createJITTargetMachineBuilder(TT);
    if (!JTMB)
      return JTMB.takeError();
    auto JB = IncrementalExecutor::createDefaultJITBuilder(std::move(*JTMB));
    if (!JB)
      return JB.takeError();
    JITBuilder = std::move(*JB);
  }

  llvm::Error Err = llvm::Error::success();
#ifdef __EMSCRIPTEN__
  auto Executor = std::make_unique<WasmIncrementalExecutor>(*TSCtx);
#else
  auto Executor =
      std::make_unique<IncrementalExecutor>(*TSCtx, *JITBuilder, Err);
#endif
  if (!Err)
    IncrExecutor = std::move(Executor);

  return Err;
}

void Interpreter::ResetExecutor() { IncrExecutor.reset(); }

llvm::Error Interpreter::Execute(PartialTranslationUnit &T) {
  assert(T.TheModule);
  LLVM_DEBUG(
      llvm::dbgs() << "execute-ptu "
                   << (llvm::is_contained(PTUs, T)
                           ? std::distance(PTUs.begin(), llvm::find(PTUs, T))
                           : -1)
                   << ": [TU=" << T.TUPart << ", M=" << T.TheModule.get()
                   << " (" << T.TheModule->getName() << ")]\n");
  if (!IncrExecutor) {
    auto Err = CreateExecutor();
    if (Err)
      return Err;
  }
  // FIXME: Add a callback to retain the llvm::Module once the JIT is done.
  if (auto Err = IncrExecutor->addModule(T))
    return Err;

  if (auto Err = IncrExecutor->runCtors())
    return Err;

  return llvm::Error::success();
}

llvm::Error Interpreter::ParseAndExecute(llvm::StringRef Code, Value *V) {

  auto PTU = Parse(Code);
  if (!PTU)
    return PTU.takeError();
  if (PTU->TheModule)
    if (llvm::Error Err = Execute(*PTU))
      return Err;

  if (LastValue.isValid()) {
    if (!V) {
      LastValue.dump();
      LastValue.clear();
    } else
      *V = std::move(LastValue);
  }
  return llvm::Error::success();
}

llvm::Expected<llvm::orc::ExecutorAddr>
Interpreter::getSymbolAddress(GlobalDecl GD) const {
  if (!IncrExecutor)
    return llvm::make_error<llvm::StringError>("Operation failed. "
                                               "No execution engine",
                                               std::error_code());
  llvm::StringRef MangledName = getCodeGen()->GetMangledName(GD);
  return getSymbolAddress(MangledName);
}

llvm::Expected<llvm::orc::ExecutorAddr>
Interpreter::getSymbolAddress(llvm::StringRef IRName) const {
  if (!IncrExecutor)
    return llvm::make_error<llvm::StringError>("Operation failed. "
                                               "No execution engine",
                                               std::error_code());

  return IncrExecutor->getSymbolAddress(IRName, IncrementalExecutor::IRName);
}

llvm::Expected<llvm::orc::ExecutorAddr>
Interpreter::getSymbolAddressFromLinkerName(llvm::StringRef Name) const {
  if (!IncrExecutor)
    return llvm::make_error<llvm::StringError>("Operation failed. "
                                               "No execution engine",
                                               std::error_code());

  return IncrExecutor->getSymbolAddress(Name, IncrementalExecutor::LinkerName);
}

llvm::Error Interpreter::Undo(unsigned N) {

  if (N > getEffectivePTUSize())
    return llvm::make_error<llvm::StringError>("Operation failed. "
                                               "Too many undos",
                                               std::error_code());
  for (unsigned I = 0; I < N; I++) {
    if (IncrExecutor) {
      if (llvm::Error Err = IncrExecutor->removeModule(PTUs.back()))
        return Err;
    }

    IncrParser->CleanUpPTU(PTUs.back().TUPart);
    PTUs.pop_back();
  }
  return llvm::Error::success();
}

llvm::Error Interpreter::LoadDynamicLibrary(const char *name) {
#ifdef __EMSCRIPTEN__
  void *handle = dlopen(name, RTLD_NOW | RTLD_GLOBAL);
  if (!handle) {
    llvm::errs() << dlerror() << '\n';
    return llvm::make_error<llvm::StringError>("Failed to load dynamic library",
                                               llvm::inconvertibleErrorCode());
  }
#else
  auto EE = getExecutionEngine();
  if (!EE)
    return EE.takeError();

  auto &DL = EE->getDataLayout();

  if (auto DLSG = llvm::orc::DynamicLibrarySearchGenerator::Load(
          name, DL.getGlobalPrefix()))
    EE->getMainJITDylib().addGenerator(std::move(*DLSG));
  else
    return DLSG.takeError();
#endif

  return llvm::Error::success();
}

std::unique_ptr<llvm::Module>
Interpreter::GenModule(IncrementalAction *Action) {
  static unsigned ID = 0;
  if (CodeGenerator *CG = getCodeGen(Action)) {
    // Clang's CodeGen is designed to work with a single llvm::Module. In many
    // cases for convenience various CodeGen parts have a reference to the
    // llvm::Module (TheModule or Module) which does not change when a new
    // module is pushed. However, the execution engine wants to take ownership
    // of the module which does not map well to CodeGen's design. To work this
    // around we created an empty module to make CodeGen happy. We should make
    // sure it always stays empty.
    assert(((!CachedInCodeGenModule ||
             !getCompilerInstance()->getPreprocessorOpts().Includes.empty()) ||
            (CachedInCodeGenModule->empty() &&
             CachedInCodeGenModule->global_empty() &&
             CachedInCodeGenModule->alias_empty() &&
             CachedInCodeGenModule->ifunc_empty())) &&
           "CodeGen wrote to a readonly module");
    std::unique_ptr<llvm::Module> M(CG->ReleaseModule());
    CG->StartModule("incr_module_" + std::to_string(ID++), M->getContext());
    return M;
  }
  return nullptr;
}

CodeGenerator *Interpreter::getCodeGen(IncrementalAction *Action) const {
  if (!Action)
    Action = Act.get();
  FrontendAction *WrappedAct = Action->getWrapped();
  if (!WrappedAct->hasIRSupport())
    return nullptr;
  return static_cast<CodeGenAction *>(WrappedAct)->getCodeGenerator();
}
} // namespace clang
