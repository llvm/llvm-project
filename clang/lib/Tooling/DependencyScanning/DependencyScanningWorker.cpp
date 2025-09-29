//===- DependencyScanningWorker.cpp - clang-scan-deps worker --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/DependencyScanningWorker.h"
#include "DependencyScannerImpl.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Basic/DiagnosticSerialization.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Serialization/ObjectFilePCHContainerReader.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningService.h"
#include "clang/Tooling/DependencyScanning/InProcessModuleCache.h"
#include "clang/Tooling/DependencyScanning/ModuleDepCollector.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TargetParser/Host.h"
#include <optional>

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

llvm::Error DependencyScanningWorker::computeDependencies(
    StringRef WorkingDirectory, const std::vector<std::string> &CommandLine,
    DependencyConsumer &Consumer, DependencyActionController &Controller,
    StringRef ModuleName) {
  // Capture the emitted diagnostics and report them to the client
  // in the case of a failure.
  TextDiagnosticsPrinterWithOutput DiagPrinterWithOS(CommandLine);

  if (computeDependencies(WorkingDirectory, CommandLine, Consumer, Controller,
                          DiagPrinterWithOS.DiagPrinter, ModuleName))
    return llvm::Error::success();
  return llvm::make_error<llvm::StringError>(
      DiagPrinterWithOS.DiagnosticsOS.str(), llvm::inconvertibleErrorCode());
}

static bool forEachDriverJob(
    ArrayRef<std::string> ArgStrs, DiagnosticsEngine &Diags,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
    llvm::function_ref<bool(const driver::Command &Cmd)> Callback) {
  // Compilation owns a reference to the Driver, hence we need to
  // keep the Driver alive when we use Compilation.
  auto [Driver, Compilation] = buildCompilation(ArgStrs, Diags, FS);
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
    DiagnosticConsumer &DC, llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
    std::optional<StringRef> ModuleName) {
  std::vector<const char *> CCommandLine(CommandLine.size(), nullptr);
  llvm::transform(CommandLine, CCommandLine.begin(),
                  [](const std::string &Str) { return Str.c_str(); });
  auto DiagOpts = CreateAndPopulateDiagOpts(CCommandLine);
  sanitizeDiagOpts(*DiagOpts);
  auto Diags = CompilerInstance::createDiagnostics(*FS, *DiagOpts, &DC,
                                                   /*ShouldOwnClient=*/false);

  DependencyScanningAction Action(Service, WorkingDirectory, Consumer,
                                  Controller, DepFS, ModuleName);

  bool Success = false;
  if (CommandLine[1] == "-cc1") {
    Success = createAndRunToolInvocation(CommandLine, Action, FS,
                                         PCHContainerOps, *Diags, Consumer);
  } else {
    Success = forEachDriverJob(
        CommandLine, *Diags, FS, [&](const driver::Command &Cmd) {
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
          return createAndRunToolInvocation(std::move(Argv), Action, FS,
                                            PCHContainerOps, *Diags, Consumer);
        });
  }

  if (Success && !Action.hasScanned())
    Diags->Report(diag::err_fe_expected_compiler_job)
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
  std::optional<std::vector<std::string>> ModifiedCommandLine;
  auto FinalFS = initVFSForTUBufferScanning(
      BaseFS, ModifiedCommandLine, CommandLine, WorkingDirectory, TUBuffer);
  const std::vector<std::string> &FinalCommandLine =
      ModifiedCommandLine ? *ModifiedCommandLine : CommandLine;

  return scanDependencies(WorkingDirectory, FinalCommandLine, Consumer,
                          Controller, DC, FinalFS, /*ModuleName=*/std::nullopt);
}

bool DependencyScanningWorker::computeDependencies(
    StringRef WorkingDirectory, const std::vector<std::string> &CommandLine,
    DependencyConsumer &Consumer, DependencyActionController &Controller,
    DiagnosticConsumer &DC, StringRef ModuleName) {
  auto ModifiedCommandLine = CommandLine;
  auto OverlayFS = initVFSForByNameScanning(BaseFS, ModifiedCommandLine,
                                            WorkingDirectory, ModuleName);

  return scanDependencies(WorkingDirectory, ModifiedCommandLine, Consumer,
                          Controller, DC, OverlayFS, ModuleName);
}

DependencyActionController::~DependencyActionController() {}
