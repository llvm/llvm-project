//===- DependencyScanningWorker.cpp - clang-scan-deps worker --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/DependencyScanningWorker.h"
#include "DependencyScannerImpl.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"

using namespace clang;
using namespace tooling;
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

llvm::Error DependencyScanningWorker::computeDependencies(
    StringRef WorkingDirectory, const std::vector<std::string> &CommandLine,
    DependencyConsumer &Consumer, DependencyActionController &Controller,
    std::optional<llvm::MemoryBufferRef> TUBuffer) {
  // Capture the emitted diagnostics and report them to the client
  // in the case of a failure.
  TextDiagnosticsPrinterWithOutput DiagPrinterWithOS(CommandLine);

  if (computeDependencies(WorkingDirectory, CommandLine, Consumer, Controller,
                          DiagPrinterWithOS.DiagPrinter, TUBuffer))
    return llvm::Error::success();
  return llvm::make_error<llvm::StringError>(
      DiagPrinterWithOS.DiagnosticsOS.str(), llvm::inconvertibleErrorCode());
}

static bool forEachDriverJob(
    ArrayRef<std::string> ArgStrs, DiagnosticsEngine &Diags,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
    llvm::function_ref<bool(const driver::Command &Cmd)> Callback) {
  // Compilation holds a non-owning a reference to the Driver, hence we need to
  // keep the Driver alive when we use Compilation. Arguments to commands may be
  // owned by Alloc when expanded from response files.
  llvm::BumpPtrAllocator Alloc;
  auto [Driver, Compilation] = buildCompilation(ArgStrs, Diags, FS, Alloc);
  if (!Compilation)
    return false;
  for (const driver::Command &Job : Compilation->getJobs()) {
    if (!Callback(Job))
      return false;
  }
  return true;
}

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
    StringRef WorkingDirectory, const std::vector<std::string> &CommandLine,
    DependencyConsumer &Consumer, DependencyActionController &Controller,
    DiagnosticConsumer &DC,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS) {
  DignosticsEngineWithDiagOpts DiagEngineWithCmdAndOpts(CommandLine, FS, DC);
  DependencyScanningAction Action(Service, WorkingDirectory, Consumer,
                                  Controller, DepFS);

  bool Success = false;
  if (CommandLine[1] == "-cc1") {
    Success = createAndRunToolInvocation(
        CommandLine, Action, FS, PCHContainerOps,
        *DiagEngineWithCmdAndOpts.DiagEngine, Consumer);
  } else {
    Success = forEachDriverJob(
        CommandLine, *DiagEngineWithCmdAndOpts.DiagEngine, FS,
        [&](const driver::Command &Cmd) {
          if (StringRef(Cmd.getCreator().getName()) != "clang") {
            // Non-clang command. Just pass through to the dependency
            // consumer.
            Consumer.handleBuildCommand(
                {Cmd.getExecutable(),
                 {Cmd.getArguments().begin(), Cmd.getArguments().end()}});
            return true;
          }

          // Insert -cc1 comand line options into Argv
          std::vector<std::string> Argv;
          Argv.push_back(Cmd.getExecutable());
          llvm::append_range(Argv, Cmd.getArguments());

          // Create an invocation that uses the underlying file
          // system to ensure that any file system requests that
          // are made by the driver do not go through the
          // dependency scanning filesystem.
          return createAndRunToolInvocation(
              std::move(Argv), Action, FS, PCHContainerOps,
              *DiagEngineWithCmdAndOpts.DiagEngine, Consumer);
        });
  }

  if (Success && !Action.hasScanned())
    DiagEngineWithCmdAndOpts.DiagEngine->Report(
        diag::err_fe_expected_compiler_job)
        << llvm::join(CommandLine, " ");

  // Ensure finish() is called even if we never reached ExecuteAction().
  if (!Action.hasDiagConsumerFinished())
    DC.finish();

  return Success && Action.hasScanned();
}

bool DependencyScanningWorker::computeDependencies(
    StringRef WorkingDirectory, const std::vector<std::string> &CommandLine,
    DependencyConsumer &Consumer, DependencyActionController &Controller,
    DiagnosticConsumer &DC, std::optional<llvm::MemoryBufferRef> TUBuffer) {
  if (TUBuffer) {
    auto [FinalFS, FinalCommandLine] = initVFSForTUBuferScanning(
        BaseFS, CommandLine, WorkingDirectory, *TUBuffer);
    return scanDependencies(WorkingDirectory, FinalCommandLine, Consumer,
                            Controller, DC, FinalFS);
  } else {
    BaseFS->setCurrentWorkingDirectory(WorkingDirectory);
    return scanDependencies(WorkingDirectory, CommandLine, Consumer, Controller,
                            DC, BaseFS);
  }
}

llvm::Error
DependencyScanningWorker::initializeCompilerInstanceWithContextOrError(
    StringRef CWD, const std::vector<std::string> &CommandLine) {
  bool Success = initializeCompilerInstanceWithContext(CWD, CommandLine);
  return CIWithContext->handleReturnStatus(Success);
}

llvm::Error
DependencyScanningWorker::computeDependenciesByNameWithContextOrError(
    StringRef ModuleName, DependencyConsumer &Consumer,
    DependencyActionController &Controller) {
  bool Success =
      computeDependenciesByNameWithContext(ModuleName, Consumer, Controller);
  return CIWithContext->handleReturnStatus(Success);
}

llvm::Error
DependencyScanningWorker::finalizeCompilerInstanceWithContextOrError() {
  bool Success = finalizeCompilerInstance();
  return CIWithContext->handleReturnStatus(Success);
}

bool DependencyScanningWorker::initializeCompilerInstanceWithContext(
    StringRef CWD, const std::vector<std::string> &CommandLine,
    DiagnosticConsumer *DC) {
  CIWithContext =
      std::make_unique<CompilerInstanceWithContext>(*this, CWD, CommandLine);
  return CIWithContext->initialize(DC);
}

bool DependencyScanningWorker::computeDependenciesByNameWithContext(
    StringRef ModuleName, DependencyConsumer &Consumer,
    DependencyActionController &Controller) {
  assert(CIWithContext && "CompilerInstance with context required!");
  return CIWithContext->computeDependencies(ModuleName, Consumer, Controller);
}

bool DependencyScanningWorker::finalizeCompilerInstance() {
  return CIWithContext->finalize();
}
