//===- CompilerInstanceWithContext.cpp - CI for dependency scanning -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/DependencyScanning/CompilerInstanceWithContext.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/DependencyScanning/DependencyActionController.h"
#include "clang/DependencyScanning/DependencyConsumer.h"
#include "clang/DependencyScanning/DependencyScannerImpl.h"
#include "clang/Frontend/FrontendActions.h"
#include "llvm/ADT/ScopeExit.h"

using namespace clang;
using namespace dependencies;

std::optional<CompilerInstanceWithContext>
CompilerInstanceWithContext::initializeFromCC1Commandline(
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

bool CompilerInstanceWithContext::initialize(
    DependencyActionController &Controller,
    std::unique_ptr<DiagnosticsEngineWithDiagOpts> DiagEngineWithDiagOpts,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> OverlayFS) {
  assert(DiagEngineWithDiagOpts && "Valid diagnostics engine required!");
  assert(OverlayFS && "OverlayFS required!");
  auto FS = Worker.makeEffectiveVFS(CWD, std::move(OverlayFS));

  OriginalInvocation = createCompilerInvocation(
      CommandLine, *DiagEngineWithDiagOpts->DiagEngine);
  if (!OriginalInvocation) {
    DiagEngineWithDiagOpts->DiagEngine->Report(
        diag::err_fe_expected_compiler_job)
        << llvm::join(CommandLine, " ");
    return false;
  }

  if (any(Worker.Service.getOpts().OptimizeArgs &
          ScanningOptimizations::Macros))
    canonicalizeDefines(OriginalInvocation->getPreprocessorOpts());

  // Create the CompilerInstance.
  std::shared_ptr<ModuleCache> ModCache =
      makeInProcessModuleCache(Worker.Service.getModuleCacheEntries());
  CIPtr = std::make_unique<CompilerInstance>(
      createScanCompilerInvocation(*OriginalInvocation, Worker.Service,
                                   Controller),
      Worker.PCHContainerOps, std::move(ModCache));
  auto &CI = *CIPtr;

  initializeScanCompilerInstance(
      CI, std::move(FS), DiagEngineWithDiagOpts->DiagEngine->getClient(),
      Worker.Service, Worker.DepFS);

  StableDirs = getInitialStableDirs(CI);
  auto MaybePrebuiltModulesASTMap =
      computePrebuiltModulesASTMap(CI, StableDirs);
  if (!MaybePrebuiltModulesASTMap)
    return false;

  PrebuiltModuleASTMap = std::move(*MaybePrebuiltModulesASTMap);
  OutputOpts = createDependencyOutputOptions(*OriginalInvocation);

  // We do not create the target in initializeScanCompilerInstance because
  // setting it here is unique for by-name lookups. We create the target only
  // once here, and the information is reused for all computeDependencies calls.
  // We do not need to call createTarget explicitly if we go through
  // CompilerInstance::ExecuteAction to perform scanning.
  CI.createTarget();

  return true;
}

bool CompilerInstanceWithContext::computeDependencies(
    StringRef ModuleName, DependencyConsumer &Consumer,
    DependencyActionController &Controller) {
  if (SrcLocOffset >= MaxNumOfQueries)
    llvm::report_fatal_error("exceeded maximum by-name scans for worker");

  assert(CIPtr && "CIPtr must be initialized before calling this method");
  auto &CI = *CIPtr;

  // We need to reset the diagnostics, so that the diagnostics issued
  // during a previous computeDependencies call do not affect the current call.
  // If we do not reset, we may inherit fatal errors from a previous call.
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
    SrcMgr::CharacteristicKind FileType = SM.getFileCharacteristic(IDLocation);
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
  MDC->applyDiscoveredDependencies(ModuleInvocation);

  bool Success = ModuleInvocation.withCowRef<bool>(
      [&](CowCompilerInvocation &CowModuleInvocation) {
        return Controller.finalize(CI, CowModuleInvocation);
      });
  if (!Success)
    return false;

  Consumer.handleBuildCommand(
      {CommandLine[0], ModuleInvocation.getCC1CommandLine()});

  return true;
}
