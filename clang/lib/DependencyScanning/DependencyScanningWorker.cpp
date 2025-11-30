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
#include "clang/DependencyScanning/DependencyScannerImpl.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/Serialization/ObjectFilePCHContainerReader.h"

using namespace clang;
using namespace dependencies;

DependencyScanningWorker::DependencyScanningWorker(
    DependencyScanningService &Service,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS)
    : Service(Service) {
  PCHContainerOps = std::make_shared<PCHContainerOperations>();
  // We need to read object files from PCH built outside the scanner.
  PCHContainerOps->registerReader(
      std::make_unique<ObjectFilePCHContainerReader>());
  // The scanner itself writes only raw ast files.
  PCHContainerOps->registerWriter(std::make_unique<RawPCHContainerWriter>());

  if (Service.shouldTraceVFS())
    FS = llvm::makeIntrusiveRefCnt<llvm::vfs::TracingFileSystem>(std::move(FS));

  switch (Service.getMode()) {
  case ScanningMode::DependencyDirectivesScan:
    DepFS = llvm::makeIntrusiveRefCnt<DependencyScanningWorkerFilesystem>(
        Service.getSharedCache(), FS);
    BaseFS = DepFS;
    break;
  case ScanningMode::CanonicalPreprocessing:
    DepFS = nullptr;
    BaseFS = FS;
    break;
  }
}

DependencyScanningWorker::~DependencyScanningWorker() = default;
DependencyActionController::~DependencyActionController() = default;

static bool createAndRunToolInvocation(
    const std::vector<std::string> &CommandLine,
    DependencyScanningAction &Action,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
    std::shared_ptr<clang::PCHContainerOperations> &PCHContainerOps,
    DiagnosticsEngine &Diags, DependencyConsumer &Consumer) {
  auto Invocation = createCompilerInvocation(CommandLine, Diags);
  if (!Invocation)
    return false;

  if (!Action.runInvocation(std::move(Invocation), std::move(FS),
                            PCHContainerOps, Diags.getClient()))
    return false;

  std::vector<std::string> Args = Action.takeLastCC1Arguments();
  Consumer.handleBuildCommand({CommandLine[0], std::move(Args)});
  return true;
}

bool DependencyScanningWorker::scanDependencies(
    StringRef WorkingDirectory, ArrayRef<std::vector<std::string>> CommandLines,
    DependencyConsumer &Consumer, DependencyActionController &Controller,
    DiagnosticsEngine &Diags,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS) {
  DependencyScanningAction Action(Service, WorkingDirectory, Consumer,
                                  Controller, DepFS);

  const bool Success = llvm::all_of(CommandLines, [&](const auto &Cmd) {
    if (StringRef(Cmd[1]) != "-cc1") {
      // Non-clang command. Just pass through to the dependency consumer.
      Consumer.handleBuildCommand({Cmd.front(), {Cmd.begin() + 1, Cmd.end()}});
      return true;
    }
    // Create an invocation that uses the underlying file
    // system to ensure that any file system requests that
    // are made by the driver do not go through the
    // dependency scanning filesystem.
    return createAndRunToolInvocation(Cmd, Action, FS, PCHContainerOps, Diags,
                                      Consumer);
  });

  // Ensure finish() is called even if we never reached ExecuteAction().
  if (!Action.hasDiagConsumerFinished())
    Diags.getClient()->finish();

  return Success && Action.hasScanned();
}

bool DependencyScanningWorker::computeDependencies(
    StringRef WorkingDirectory, const std::vector<std::string> &CommandLine,
    DependencyConsumer &DepConsumer, DependencyActionController &Controller,
    DiagnosticConsumer &DiagConsumer,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> ScanFS) {
  auto FinalFS = ScanFS == nullptr ? BaseFS : ScanFS;
  DignosticsEngineWithDiagOpts DiagEngineWithDiagOpts(CommandLine, FinalFS,
                                                      DiagConsumer);
  return computeDependencies(
      WorkingDirectory, ArrayRef<std::vector<std::string>>{CommandLine},
      DepConsumer, Controller, *DiagEngineWithDiagOpts.DiagEngine, ScanFS);
}

bool DependencyScanningWorker::computeDependencies(
    StringRef WorkingDirectory, ArrayRef<std::vector<std::string>> CommandLines,
    DependencyConsumer &DepConsumer, DependencyActionController &Controller,
    DiagnosticsEngine &Diags,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> ScanFS) {
  auto FinalFS = ScanFS == nullptr ? BaseFS : ScanFS;
  return scanDependencies(WorkingDirectory, CommandLines, DepConsumer,
                          Controller, Diags, FinalFS);
}

bool DependencyScanningWorker::initializeCompilerInstanceWithContext(
    StringRef CWD, ArrayRef<std::string> CommandLine, DiagnosticConsumer &DC) {
  auto DiagEngineWithCmdAndOpts =
      std::make_unique<DignosticsEngineWithDiagOpts>(CommandLine, BaseFS, DC);
  return initializeCompilerInstanceWithContext(
      CWD, CommandLine, std::move(DiagEngineWithCmdAndOpts), BaseFS);
}

bool DependencyScanningWorker::initializeCompilerInstanceWithContext(
    StringRef CWD, ArrayRef<std::string> CommandLine,
    std::unique_ptr<DignosticsEngineWithDiagOpts> DiagEngineWithDiagOpts,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS) {
  CIWithContext =
      std::make_unique<CompilerInstanceWithContext>(*this, CWD, CommandLine);
  return CIWithContext->initialize(std::move(DiagEngineWithDiagOpts), FS);
}

bool DependencyScanningWorker::computeDependenciesByNameWithContext(
    StringRef ModuleName, DependencyConsumer &Consumer,
    DependencyActionController &Controller) {
  assert(CIWithContext && "CompilerInstance with context required!");
  return CIWithContext->computeDependencies(ModuleName, Consumer, Controller);
}

bool DependencyScanningWorker::finalizeCompilerInstanceWithContext() {
  return CIWithContext->finalize();
}
