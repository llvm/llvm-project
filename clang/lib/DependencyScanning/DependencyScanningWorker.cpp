//===- DependencyScanningWorker.cpp - Thread-Safe Scanning Worker ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/DependencyScanning/DependencyScanningWorker.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/DependencyScanning/DependencyActionController.h"
#include "clang/DependencyScanning/DependencyConsumer.h"
#include "clang/DependencyScanning/DependencyScannerImpl.h"
#include "clang/DependencyScanning/DependencyScanningUtils.h"
#include "clang/DependencyScanning/InProcessModuleCache.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Serialization/ObjectFilePCHContainerReader.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <optional>

using namespace clang;
using namespace dependencies;

namespace clang {
namespace dependencies {
class CompilerInstanceWithContext {
  // Context
  DependencyScanningWorker &Worker;
  llvm::StringRef CWD;
  std::vector<std::string> CommandLine;

  // Context - compiler invocation
  std::unique_ptr<CompilerInvocation> OriginalInvocation;

  // Context - output options
  std::unique_ptr<DependencyOutputOptions> OutputOpts;

  // Context - stable directory handling
  llvm::SmallVector<StringRef> StableDirs;
  PrebuiltModulesAttrsMap PrebuiltModuleASTMap;

  // Context - used by AsyncScan's prescan pass
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> ScanFS;

  // Compiler Instance
  std::unique_ptr<CompilerInstance> CIPtr;

  // Source location offset.
  int32_t SrcLocOffset = 0;

  CompilerInstanceWithContext(DependencyScanningWorker &Worker, StringRef CWD,
                              ArrayRef<std::string> CMD)
      : Worker(Worker), CWD(CWD), CommandLine(CMD.begin(), CMD.end()) {}

  bool initialize(
      DependencyActionController &Controller,
      std::unique_ptr<DiagnosticsEngineWithDiagOpts> DiagEngineWithDiagOpts,
      IntrusiveRefCntPtr<llvm::vfs::FileSystem> OverlayFS) {
    {
      auto LogLine = Worker.Service.getLogger().log();
      LogLine.logArray("init_compiler_instance_with_context:", " ",
                       CommandLine);
    }
    assert(DiagEngineWithDiagOpts && "Valid diagnostics engine required!");
    ScanFS = Worker.makeEffectiveVFS(CWD, std::move(OverlayFS));

    OriginalInvocation = createCompilerInvocation(
        CommandLine, *DiagEngineWithDiagOpts->DiagEngine);
    if (!OriginalInvocation) {
      DiagEngineWithDiagOpts->DiagEngine->Report(
          diag::err_fe_expected_compiler_job)
          << llvm::join(CommandLine, " ");
      return false;
    }

    return initializeScanInstance(
        Controller, DiagEngineWithDiagOpts->DiagEngine->getClient());
  }

  bool initializeScanInstance(DependencyActionController &Controller,
                              DiagnosticConsumer *DiagConsumer) {
    assert(OriginalInvocation && ScanFS &&
           "OriginalInvocation and ScanFS must be set before this call");

    if (any(Worker.Service.getOpts().OptimizeArgs &
            ScanningOptimizations::Macros))
      canonicalizeDefines(OriginalInvocation->getPreprocessorOpts());

    // Create the CompilerInstance.
    std::shared_ptr<ModuleCache> ModCache = makeInProcessModuleCache(
        Worker.Service.getModuleCacheEntries(), Worker.Service.getLogger());
    CIPtr = std::make_unique<CompilerInstance>(
        createScanCompilerInvocation(*OriginalInvocation, Worker.Service,
                                     Controller),
        Worker.PCHContainerOps, std::move(ModCache));
    auto &CI = *CIPtr;

    initializeScanCompilerInstance(CI, ScanFS, DiagConsumer, Worker.Service,
                                   Worker.DepFS);

    StableDirs = getInitialStableDirs(CI);
    auto MaybePrebuiltModulesASTMap =
        computePrebuiltModulesASTMap(CI, StableDirs);
    if (!MaybePrebuiltModulesASTMap)
      return false;

    PrebuiltModuleASTMap = std::move(*MaybePrebuiltModulesASTMap);
    OutputOpts = createDependencyOutputOptions(*OriginalInvocation);

    // We do not create the target in initializeScanCompilerInstance because
    // setting it here is unique for by-name lookups. We create the target only
    // once here, and the information is reused for all computeDependencies
    // calls. We do not need to call createTarget explicitly if we go through
    // CompilerInstance::ExecuteAction to perform scanning.
    CI.createTarget();

    return true;
  }

  bool prescanModulesAsync(AsyncModuleCompiles &Compiles,
                           DependencyActionController &Controller) {
    auto ModCache = makeInProcessModuleCache(
        Worker.Service.getModuleCacheEntries(), Worker.Service.getLogger());
    CompilerInstance PrescanCI(
        std::make_shared<CompilerInvocation>(CIPtr->getInvocation()),
        Worker.PCHContainerOps, std::move(ModCache));

    DiagnosticConsumer DiagConsumer;
    initializeScanCompilerInstance(PrescanCI, ScanFS, &DiagConsumer,
                                   Worker.Service, Worker.DepFS);

    // FIXME: reuse the StableDirs/PrebuiltModuleASTMap computed in
    // initialize().
    SmallVector<StringRef> PrescanStableDirs = getInitialStableDirs(PrescanCI);
    if (!computePrebuiltModulesASTMap(PrescanCI, PrescanStableDirs))
      return false;

    if (PrescanCI.getFrontendOpts().ProgramAction == frontend::GeneratePCH)
      PrescanCI.getLangOpts().CompilingPCH = true;

    runTUModulePrescan(PrescanCI, Worker.Service, Controller, Compiles);
    return true;
  }

public:
  static std::optional<CompilerInstanceWithContext>
  initializeFromCC1Commandline(
      DependencyScanningWorker &Worker, StringRef CWD,
      ArrayRef<std::string> CC1CommandLine,
      std::unique_ptr<DiagnosticsEngineWithDiagOpts> DiagEngineWithDiagOpts,
      IntrusiveRefCntPtr<llvm::vfs::FileSystem> OverlayFS,
      DependencyActionController &Controller) {
    CompilerInstanceWithContext CIWC(Worker, CWD, CC1CommandLine);
    if (!CIWC.initialize(Controller, std::move(DiagEngineWithDiagOpts),
                         std::move(OverlayFS)))
      return std::nullopt;
    return std::move(CIWC);
  }

  bool computeDependencies(StringRef ModuleName, DependencyConsumer &Consumer,
                           DependencyActionController &Controller) {
    Worker.Service.getLogger().log() << "start scan_by_name: " << ModuleName;
    llvm::scope_exit ExitLogging([&]() {
      Worker.Service.getLogger().log() << "finish scan_by_name: " << ModuleName;
    });
    if (SrcLocOffset >= DependencyScanningWorker::MaxNumOfByNameQueries)
      llvm::report_fatal_error("exceeded maximum by-name scans for worker");

    assert(CIPtr && "CIPtr must be initialized before calling this method");
    auto &CI = *CIPtr;

    // We need to reset the diagnostics, so that the diagnostics issued
    // during a previous computeDependencies call do not affect the current
    // call. If we do not reset, we may inherit fatal errors from a previous
    // call.
    CI.getDiagnostics().Reset();

    // We create this cleanup object because computeDependencies may exit
    // early with errors.
    llvm::scope_exit CleanUp([&]() {
      CI.clearDependencyCollectors();
      // The preprocessor may not be created at the entry of this method,
      // but it must have been created when this method returns, whether
      // there are errors during scanning or not.
      CI.getPreprocessor().removePPCallbacks();
    });

    auto MDC = initializeScanInstanceDependencyCollector(
        CI, std::make_unique<DependencyOutputOptions>(*OutputOpts),
        Worker.Service,
        /* The MDC's constructor makes a copy of the OriginalInvocation, so
        we can pass it in without worrying that it might be changed across
        invocations of computeDependencies. */
        *OriginalInvocation, Controller, PrebuiltModuleASTMap, StableDirs);

    CompilerInvocation ModuleInvocation(*OriginalInvocation);
    if (!Controller.initialize(CI, ModuleInvocation))
      return false;

    if (!SrcLocOffset) {
      // When SrcLocOffset is zero, we are at the beginning of the fake source
      // file. In this case, we call BeginSourceFile to initialize.
      std::unique_ptr<FrontendAction> Action =
          std::make_unique<PreprocessOnlyAction>();
      auto *InputFile = CI.getFrontendOpts().Inputs.begin();
      bool ActionBeginSucceeded = Action->BeginSourceFile(CI, *InputFile);
      assert(ActionBeginSucceeded && "Action BeginSourceFile must succeed");
      (void)ActionBeginSucceeded;
    }

    Preprocessor &PP = CI.getPreprocessor();
    SourceManager &SM = PP.getSourceManager();
    FileID MainFileID = SM.getMainFileID();
    SourceLocation FileStart = SM.getLocForStartOfFile(MainFileID);
    SourceLocation IDLocation = FileStart.getLocWithOffset(SrcLocOffset);
    PPCallbacks *CB = nullptr;
    if (!SrcLocOffset) {
      // We need to call EnterSourceFile when SrcLocOffset is zero to initialize
      // the preprocessor.
      bool PPFailed = PP.EnterSourceFile(MainFileID, nullptr, SourceLocation());
      assert(!PPFailed && "Preprocess must be able to enter the main file.");
      (void)PPFailed;
      CB = MDC->getPPCallbacks();
    } else {
      // When SrcLocOffset is non-zero, the preprocessor has already been
      // initialized through a previous call of computeDependencies. We want to
      // preserve the PP's state, hence we do not call EnterSourceFile again.
      MDC->attachToPreprocessor(PP);
      CB = MDC->getPPCallbacks();

      FileID PrevFID;
      SrcMgr::CharacteristicKind FileType =
          SM.getFileCharacteristic(IDLocation);
      CB->LexedFileChanged(MainFileID,
                           PPChainedCallbacks::LexedFileChangeReason::EnterFile,
                           FileType, PrevFID, IDLocation);
    }

    // FIXME: Scan modules asynchronously here as well.

    SrcLocOffset++;
    SmallVector<IdentifierLoc, 2> Path;
    IdentifierInfo *ModuleID = PP.getIdentifierInfo(ModuleName);
    Path.emplace_back(IDLocation, ModuleID);
    auto ModResult = CI.loadModule(IDLocation, Path, Module::Hidden, false);

    assert(CB && "Must have PPCallbacks after module loading");
    CB->moduleImport(SourceLocation(), Path, ModResult);

    if (!ModResult)
      return false;

    if (CI.getDiagnostics().hasErrorOccurred())
      return false;

    MDC->run(Consumer);
    return applyAndReport(*MDC, ModuleInvocation, Consumer, Controller,
                          CommandLine[0]);
  }

  std::shared_ptr<ModuleDepCollector>
  scanTranslationUnit(DependencyConsumer &Consumer,
                      DependencyActionController &Controller) {
    assert(CIPtr && "CIPtr must be initialized before calling this method");
    auto &CI = *CIPtr;

    std::optional<AsyncModuleCompiles> AsyncCompiles;
    if (Worker.Service.getOpts().AsyncScanModules) {
      AsyncCompiles.emplace();
      if (!prescanModulesAsync(*AsyncCompiles, Controller))
        return nullptr;
    }

    auto MDC = initializeScanInstanceDependencyCollector(
        CI, std::make_unique<DependencyOutputOptions>(*OutputOpts),
        Worker.Service, *OriginalInvocation, Controller, PrebuiltModuleASTMap,
        StableDirs);

    if (CI.getDiagnostics().hasErrorOccurred())
      return nullptr;

    if (!Controller.initialize(CI, *OriginalInvocation))
      return nullptr;

    ReadPCHAndPreprocessAction Action;
    if (!CI.ExecuteAction(Action))
      return nullptr;

    MDC->run(Consumer);
    if (!applyAndReport(*MDC, *OriginalInvocation, Consumer, Controller,
                        CommandLine[0]))
      return nullptr;
    return MDC;
  }

  bool applyAndReport(ModuleDepCollector &MDC,
                      CompilerInvocation &ModuleInvocation,
                      DependencyConsumer &Consumer,
                      DependencyActionController &Controller,
                      StringRef Executable) {
    MDC.applyDiscoveredDependencies(ModuleInvocation);

    bool Success = ModuleInvocation.withCowRef<bool>(
        [&](CowCompilerInvocation &CowModuleInvocation) {
          return Controller.finalize(*CIPtr, CowModuleInvocation);
        });
    if (!Success)
      return false;

    Consumer.handleBuildCommand(
        {Executable.str(), ModuleInvocation.getCC1CommandLine()});
    return true;
  }
};
} // namespace dependencies
} // namespace clang

DependencyScanningWorker::DependencyScanningWorker(
    DependencyScanningService &Service)
    : Service(Service) {
  PCHContainerOps = std::make_shared<PCHContainerOperations>();
  // We need to read object files from PCH built outside the scanner.
  PCHContainerOps->registerReader(
      std::make_unique<ObjectFilePCHContainerReader>());
  // The scanner itself writes only raw ast files.
  PCHContainerOps->registerWriter(std::make_unique<RawPCHContainerWriter>());

  auto BaseFS = Service.getOpts().MakeVFS();

  if (Service.getOpts().TraceVFS) {
    TracingFS = llvm::makeIntrusiveRefCnt<llvm::vfs::TracingFileSystem>(
        std::move(BaseFS));
    BaseFS = TracingFS;
  }

  DepFS = llvm::makeIntrusiveRefCnt<DependencyScanningWorkerFilesystem>(
      Service, std::move(BaseFS));
}

DependencyScanningWorker::~DependencyScanningWorker() = default;

IntrusiveRefCntPtr<llvm::vfs::FileSystem>
DependencyScanningWorker::makeEffectiveVFS(
    StringRef WorkingDirectory,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> OverlayFS) const {
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS = DepFS;
  if (OverlayFS) {
    auto NewFS =
        llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(std::move(FS));
    NewFS->pushOverlay(std::move(OverlayFS));
    FS = std::move(NewFS);
  }
  FS->setCurrentWorkingDirectory(WorkingDirectory);
  return FS;
}

bool DependencyScanningWorker::computeDependenciesByNameWithDrain(
    StringRef CWD, ArrayRef<std::string> CC1CommandLine,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> OverlayFS,
    DiagnosticConsumer &DiagConsumer, DependencyActionController &Controller,
    const llvm::DenseSet<ModuleID> &AlreadySeen,
    llvm::function_ref<std::optional<std::string>()> getNextInput,
    llvm::function_ref<void(StringRef, std::optional<TranslationUnitDeps>)>
        deliverResult) {
  auto FS = makeEffectiveVFS(CWD, OverlayFS);
  auto DiagEngine = std::make_unique<DiagnosticsEngineWithDiagOpts>(
      CC1CommandLine, FS, DiagConsumer);

  std::optional<CompilerInstanceWithContext> CIWC =
      CompilerInstanceWithContext::initializeFromCC1Commandline(
          *this, CWD, CC1CommandLine, std::move(DiagEngine),
          std::move(OverlayFS), Controller);
  if (!CIWC)
    return false;

  while (std::optional<std::string> NextInput = getNextInput()) {
    FullDependencyConsumer Consumer(AlreadySeen);
    if (CIWC->computeDependencies(*NextInput, Consumer, Controller))
      deliverResult(*NextInput, Consumer.takeTranslationUnitDeps());
    else
      deliverResult(*NextInput, std::nullopt);
  }
  return true;
}

bool DependencyScanningWorker::computeDependencies(
    StringRef WorkingDirectory, ArrayRef<ArrayRef<std::string>> CommandLines,
    DependencyConsumer &DepConsumer, DependencyActionController &Controller,
    DiagnosticConsumer &DiagConsumer,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> OverlayFS) {
  auto FS = makeEffectiveVFS(WorkingDirectory, std::move(OverlayFS));

  bool Scanned = false;
  std::shared_ptr<ModuleDepCollector> MDC;

  std::optional<CompilerInstanceWithContext> CIWC;
  const bool Success = llvm::all_of(CommandLines, [&](const auto &Cmd) {
    if (StringRef(Cmd[1]) != "-cc1") {
      // Non-clang command. Just pass through to the dependency consumer.
      DepConsumer.handleBuildCommand(
          {Cmd.front(), {Cmd.begin() + 1, Cmd.end()}});
      return true;
    }

    Service.getLogger().log().logArray("starting scanning command:", " ", Cmd);
    llvm::scope_exit ExitLogging([&] {
      Service.getLogger().log().logArray("finished scanning command:", " ",
                                         Cmd);
    });

    auto DiagEngineWithDiagOpts =
        std::make_unique<DiagnosticsEngineWithDiagOpts>(Cmd, FS, DiagConsumer);
    if (!Scanned) {
      Scanned = true;
      auto Result = CompilerInstanceWithContext::initializeFromCC1Commandline(
          *this, WorkingDirectory, Cmd, std::move(DiagEngineWithDiagOpts),
          OverlayFS, Controller);
      if (!Result)
        return false;
      CIWC.emplace(std::move(*Result));
      MDC = CIWC->scanTranslationUnit(DepConsumer, Controller);
      return MDC != nullptr;
    }

    auto Invocation =
        createCompilerInvocation(Cmd, *DiagEngineWithDiagOpts->DiagEngine);

    if (!Invocation)
      return false;

    assert(CIWC && "Must have an initialized CIWC");
    return CIWC->applyAndReport(*MDC, *Invocation, DepConsumer, Controller,
                                Cmd.front());
  });

  return Success && Scanned;
}
