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
#include "IncrementalAction.h"
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
#include "llvm/ExecutionEngine/Orc/EPCDynamicLibrarySearchGenerator.h"
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

  // Register the support for object-file-wrapped Clang modules.
  // FIXME: Clang should register these container operations automatically.
  auto PCHOps = Clang->getPCHContainerOperations();
  PCHOps->registerWriter(std::make_unique<ObjectFilePCHContainerWriter>());
  PCHOps->registerReader(std::make_unique<ObjectFilePCHContainerReader>());

  // Buffer diagnostics from argument parsing so that we can output them using
  // a well formed diagnostic object.
  DiagnosticOptions DiagOpts;
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  DiagnosticsEngine Diags(DiagnosticIDs::create(), DiagOpts, DiagsBuffer);
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

  Clang->getTarget().adjust(Clang->getDiagnostics(), Clang->getLangOpts(),
                            Clang->getAuxTarget());

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
  std::unique_ptr<DiagnosticOptions> DiagOpts =
      CreateAndPopulateDiagOpts(ClangArgv);
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  DiagnosticsEngine Diags(DiagnosticIDs::create(), *DiagOpts, DiagsBuffer);

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

Interpreter::Interpreter(std::unique_ptr<CompilerInstance> Instance,
                         llvm::Error &ErrOut,
                         std::unique_ptr<llvm::orc::LLJITBuilder> JITBuilder,
                         std::unique_ptr<clang::ASTConsumer> Consumer,
                         JITConfig Config)
    : JITBuilder(std::move(JITBuilder)) {
  CI = std::move(Instance);
  llvm::ErrorAsOutParameter EAO(&ErrOut);
  auto LLVMCtx = std::make_unique<llvm::LLVMContext>();
  TSCtx = std::make_unique<llvm::orc::ThreadSafeContext>(std::move(LLVMCtx));

  Act = TSCtx->withContextDo([&](llvm::LLVMContext *Ctx) {
    return std::make_unique<IncrementalAction>(*CI, *Ctx, ErrOut, *this,
                                               std::move(Consumer));
  });

  if (ErrOut)
    return;
  CI->ExecuteAction(*Act);

  IncrParser =
      std::make_unique<IncrementalParser>(*CI, Act.get(), ErrOut, PTUs);

  if (ErrOut)
    return;

  if (Act->getCodeGen()) {
    Act->CacheCodeGenModule();
    // The initial PTU is filled by `-include` or by CUDA includes
    // automatically.
    if (!CI->getPreprocessorOpts().Includes.empty()) {
      // We can't really directly pass the CachedInCodeGenModule to the Jit
      // because it will steal it, causing dangling references as explained in
      // Interpreter::Execute
      auto M = llvm::CloneModule(*Act->getCachedCodeGenModule());
      ASTContext &C = CI->getASTContext();
      IncrParser->RegisterPTU(C.getTranslationUnitDecl(), std::move(M));
    }
    if (llvm::Error Err = CreateExecutor(Config)) {
      ErrOut = joinErrors(std::move(ErrOut), std::move(Err));
      return;
    }
  }

  // Not all frontends support code-generation, e.g. ast-dump actions don't
  if (Act->getCodeGen()) {
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
    struct __clang_Interpreter_NewTag{} __ci_newtag;
    void* operator new(__SIZE_TYPE__, void* __p, __clang_Interpreter_NewTag) noexcept;
    template <class T, class = T (*)() /*disable for arrays*/>
    void __clang_Interpreter_SetValueCopyArr(const T* Src, void* Placement, unsigned long Size) {
      for (auto Idx = 0; Idx < Size; ++Idx)
        new ((void*)(((T*)Placement) + Idx), __ci_newtag) T(Src[Idx]);
    }
    template <class T, unsigned long N>
    void __clang_Interpreter_SetValueCopyArr(const T (*Src)[N], void* Placement, unsigned long Size) {
      __clang_Interpreter_SetValueCopyArr(Src[0], Placement, Size);
    }
#else
    #define EXTERN_C extern
    EXTERN_C void *memcpy(void *restrict dst, const void *restrict src, __SIZE_TYPE__ n);
    EXTERN_C inline void __clang_Interpreter_SetValueCopyArr(const void* Src, void* Placement, unsigned long Size) {
      memcpy(Placement, Src, Size);
    }
#endif // __cplusplus
  EXTERN_C void *__clang_Interpreter_SetValueWithAlloc(void*, void*, void*);
  EXTERN_C void __clang_Interpreter_SetValueNoAlloc(void *This, void *OutVal, void *OpaqueType, ...);
)";

llvm::Expected<std::pair<std::unique_ptr<llvm::orc::LLJITBuilder>, uint32_t>>
Interpreter::outOfProcessJITBuilder(JITConfig Config) {
  std::unique_ptr<llvm::orc::ExecutorProcessControl> EPC;
  uint32_t childPid = -1;
  if (!Config.OOPExecutor.empty()) {
    // Launch an out-of-process executor locally in a child process.
    auto ResultOrErr = IncrementalExecutor::launchExecutor(
        Config.OOPExecutor, Config.UseSharedMemory, Config.SlabAllocateSize);
    if (!ResultOrErr)
      return ResultOrErr.takeError();
    childPid = ResultOrErr->second;
    auto EPCOrErr = std::move(ResultOrErr->first);
    EPC = std::move(EPCOrErr);
  } else if (Config.OOPExecutorConnect != "") {
#if LLVM_ON_UNIX && LLVM_ENABLE_THREADS
    auto EPCOrErr = IncrementalExecutor::connectTCPSocket(
        Config.OOPExecutorConnect, Config.UseSharedMemory,
        Config.SlabAllocateSize);
    if (!EPCOrErr)
      return EPCOrErr.takeError();
    EPC = std::move(*EPCOrErr);
#else
    return llvm::make_error<llvm::StringError>(
        "Out-of-process JIT over TCP is not supported on this platform",
        std::error_code());
#endif
  }

  std::unique_ptr<llvm::orc::LLJITBuilder> JB;
  if (EPC) {
    auto JBOrErr = clang::Interpreter::createLLJITBuilder(
        std::move(EPC), Config.OrcRuntimePath);
    if (!JBOrErr)
      return JBOrErr.takeError();
    JB = std::move(*JBOrErr);
  }

  return std::make_pair(std::move(JB), childPid);
}

llvm::Expected<std::string>
Interpreter::getOrcRuntimePath(const driver::ToolChain &TC) {
  std::optional<std::string> CompilerRTPath = TC.getCompilerRTPath();
  std::optional<std::string> ResourceDir = TC.getRuntimePath();

  if (!CompilerRTPath) {
    return llvm::make_error<llvm::StringError>("CompilerRT path not found",
                                               std::error_code());
  }

  const std::array<const char *, 3> OrcRTLibNames = {
      "liborc_rt.a", "liborc_rt_osx.a", "liborc_rt-x86_64.a"};

  for (const char *LibName : OrcRTLibNames) {
    llvm::SmallString<256> CandidatePath((*CompilerRTPath).c_str());
    llvm::sys::path::append(CandidatePath, LibName);

    if (llvm::sys::fs::exists(CandidatePath)) {
      return CandidatePath.str().str();
    }
  }

  return llvm::make_error<llvm::StringError>(
      llvm::Twine("OrcRuntime library not found in: ") + (*CompilerRTPath),
      std::error_code());
}

llvm::Expected<std::unique_ptr<Interpreter>>
Interpreter::create(std::unique_ptr<CompilerInstance> CI, JITConfig Config) {
  llvm::Error Err = llvm::Error::success();

  std::unique_ptr<llvm::orc::LLJITBuilder> JB;

  if (Config.IsOutOfProcess) {
    const TargetInfo &TI = CI->getTarget();
    const llvm::Triple &Triple = TI.getTriple();

    DiagnosticsEngine &Diags = CI->getDiagnostics();
    std::string BinaryName = llvm::sys::fs::getMainExecutable(nullptr, nullptr);
    driver::Driver Driver(BinaryName, Triple.str(), Diags);
    // Need fake args to get the driver to create a compilation.
    std::vector<const char *> Args = {"clang", "--version"};
    std::unique_ptr<clang::driver::Compilation> C(
        Driver.BuildCompilation(Args));
    if (!C) {
      return llvm::make_error<llvm::StringError>(
          "Failed to create driver compilation for out-of-process JIT",
          std::error_code());
    }
    if (Config.OrcRuntimePath == "") {
      const clang::driver::ToolChain &TC = C->getDefaultToolChain();

      auto OrcRuntimePathOrErr = getOrcRuntimePath(TC);
      if (!OrcRuntimePathOrErr) {
        return OrcRuntimePathOrErr.takeError();
      }

      Config.OrcRuntimePath = *OrcRuntimePathOrErr;
    }
  }

  auto Interp = std::unique_ptr<Interpreter>(new Interpreter(
      std::move(CI), Err, std::move(JB), /*Consumer=*/nullptr, Config));
  if (auto E = std::move(Err))
    return std::move(E);

  // Add runtime code and set a marker to hide it from user code. Undo will not
  // go through that.
  if (auto E = Interp->ParseAndExecute(Runtimes))
    return std::move(E);

  Interp->markUserCodeStart();

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

  auto DeviceAct = Interp->TSCtx->withContextDo([&](llvm::LLVMContext *Ctx) {
    return std::make_unique<IncrementalAction>(*DCI, *Ctx, Err, *Interp);
  });

  if (Err)
    return std::move(Err);

  Interp->DeviceAct = std::move(DeviceAct);

  DCI->ExecuteAction(*Interp->DeviceAct);

  Interp->DeviceCI = std::move(DCI);

  auto DeviceParser = std::make_unique<IncrementalCUDADeviceParser>(
      *Interp->DeviceCI, *Interp->getCompilerInstance(),
      Interp->DeviceAct.get(), IMVFS, Err, Interp->PTUs);

  if (Err)
    return std::move(Err);

  Interp->DeviceParser = std::move(DeviceParser);
  return std::move(Interp);
}

CompilerInstance *Interpreter::getCompilerInstance() { return CI.get(); }
const CompilerInstance *Interpreter::getCompilerInstance() const {
  return const_cast<Interpreter *>(this)->getCompilerInstance();
}

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

uint32_t Interpreter::getOutOfProcessExecutorPID() const {
  if (IncrExecutor)
    return IncrExecutor->getOutOfProcessChildPid();
  return -1;
}

llvm::Expected<PartialTranslationUnit &>
Interpreter::Parse(llvm::StringRef Code) {
  // If we have a device parser, parse it first. The generated code will be
  // included in the host compilation
  if (DeviceParser) {
    llvm::Expected<TranslationUnitDecl *> DeviceTU = DeviceParser->Parse(Code);
    if (auto E = DeviceTU.takeError())
      return std::move(E);

    DeviceParser->RegisterPTU(*DeviceTU);

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

  PTUs.emplace_back(PartialTranslationUnit());
  PartialTranslationUnit &LastPTU = PTUs.back();
  LastPTU.TUPart = *TuOrErr;

  if (std::unique_ptr<llvm::Module> M = Act->GenModule())
    LastPTU.TheModule = std::move(M);

  return LastPTU;
}

static llvm::Expected<llvm::orc::JITTargetMachineBuilder>
createJITTargetMachineBuilder(const std::string &TT) {
  if (TT == llvm::sys::getProcessTriple())
    // This fails immediately if the target backend is not registered
    return llvm::orc::JITTargetMachineBuilder::detectHost();

  // If the target backend is not registered, LLJITBuilder::create() will fail
  return llvm::orc::JITTargetMachineBuilder(llvm::Triple(TT));
}

llvm::Expected<std::unique_ptr<llvm::orc::LLJITBuilder>>
Interpreter::createLLJITBuilder(
    std::unique_ptr<llvm::orc::ExecutorProcessControl> EPC,
    llvm::StringRef OrcRuntimePath) {
  const std::string &TT = EPC->getTargetTriple().getTriple();
  auto JTMB = createJITTargetMachineBuilder(TT);
  if (!JTMB)
    return JTMB.takeError();
  auto JB = IncrementalExecutor::createDefaultJITBuilder(std::move(*JTMB));
  if (!JB)
    return JB.takeError();

  (*JB)->setExecutorProcessControl(std::move(EPC));
  (*JB)->setPlatformSetUp(
      llvm::orc::ExecutorNativePlatform(OrcRuntimePath.str()));

  return std::move(*JB);
}

llvm::Error Interpreter::CreateExecutor(JITConfig Config) {
  if (IncrExecutor)
    return llvm::make_error<llvm::StringError>("Operation failed. "
                                               "Execution engine exists",
                                               std::error_code());
  if (!Act->getCodeGen())
    return llvm::make_error<llvm::StringError>("Operation failed. "
                                               "No code generator available",
                                               std::error_code());

  const std::string &TT = getCompilerInstance()->getTargetOpts().Triple;
  llvm::Triple TargetTriple(TT);
  bool IsWindowsTarget = TargetTriple.isOSWindows();

  if (!IsWindowsTarget && Config.IsOutOfProcess) {
    if (!JITBuilder) {
      auto ResOrErr = outOfProcessJITBuilder(Config);
      if (!ResOrErr)
        return ResOrErr.takeError();
      JITBuilder = std::move(ResOrErr->first);
      Config.ExecutorPID = ResOrErr->second;
    }
    if (!JITBuilder)
      return llvm::make_error<llvm::StringError>(
          "Operation failed. No LLJITBuilder for out-of-process JIT",
          std::error_code());
  }

  if (!JITBuilder) {
    auto JTMB = createJITTargetMachineBuilder(TT);
    if (!JTMB)
      return JTMB.takeError();
    if (Config.CM)
      JTMB->setCodeModel(Config.CM);
    auto JB = IncrementalExecutor::createDefaultJITBuilder(std::move(*JTMB));
    if (!JB)
      return JB.takeError();
    JITBuilder = std::move(*JB);
  }

  llvm::Error Err = llvm::Error::success();

  // Fix: Declare Executor as the appropriate unique_ptr type
  std::unique_ptr<IncrementalExecutor> Executor;

#ifdef __EMSCRIPTEN__
  Executor = std::make_unique<WasmIncrementalExecutor>(*TSCtx);
#else
  Executor =
      std::make_unique<IncrementalExecutor>(*TSCtx, *JITBuilder, Config, Err);
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
  llvm::StringRef MangledName = Act->getCodeGen()->GetMangledName(GD);
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

  if (getEffectivePTUSize() == 0) {
    return llvm::make_error<llvm::StringError>("Operation failed. "
                                               "No input left to undo",
                                               std::error_code());
  } else if (N > getEffectivePTUSize()) {
    return llvm::make_error<llvm::StringError>(
        llvm::formatv(
            "Operation failed. Wanted to undo {0} inputs, only have {1}.", N,
            getEffectivePTUSize()),
        std::error_code());
  }

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

  if (llvm::Expected<
          std::unique_ptr<llvm::orc::EPCDynamicLibrarySearchGenerator>>
          DLSG = llvm::orc::EPCDynamicLibrarySearchGenerator::Load(
              EE->getExecutionSession(), name))
    // FIXME: Eventually we should put each library in its own JITDylib and
    //        turn off process symbols by default.
    EE->getProcessSymbolsJITDylib()->addGenerator(std::move(*DLSG));
  else
    return DLSG.takeError();
#endif

  return llvm::Error::success();
}
} // end namespace clang
