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
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace clang;
using namespace dependencies;

DependencyScanningWorker::DependencyScanningWorker(
    DependencyScanningService &Service,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS)
    : Service(Service) {
  PCHContainerOps = std::make_shared<PCHContainerOperations>();
  // We need to read object files from PCH built outside the scanner.
  PCHContainerOps->registerReader(
      std::make_unique<ObjectFilePCHContainerReader>());
  // The scanner itself writes only raw ast files.
  PCHContainerOps->registerWriter(std::make_unique<RawPCHContainerWriter>());

  if (Service.shouldTraceVFS())
    BaseFS = llvm::makeIntrusiveRefCnt<llvm::vfs::TracingFileSystem>(
        std::move(BaseFS));

  DepFS = llvm::makeIntrusiveRefCnt<DependencyScanningWorkerFilesystem>(
      Service.getSharedCache(), std::move(BaseFS));
}

DependencyScanningWorker::~DependencyScanningWorker() = default;
DependencyActionController::~DependencyActionController() = default;

static bool createAndRunToolInvocation(
    ArrayRef<std::string> CommandLine, DependencyScanningAction &Action,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
    std::shared_ptr<clang::PCHContainerOperations> &PCHContainerOps,
    DiagnosticsEngine &Diags) {
  auto Invocation = createCompilerInvocation(CommandLine, Diags);
  if (!Invocation)
    return false;

  return Action.runInvocation(CommandLine[0], std::move(Invocation),
                              std::move(FS), PCHContainerOps,
                              Diags.getClient());
}

bool DependencyScanningWorker::computeDependencies(
    StringRef WorkingDirectory, ArrayRef<std::string> CommandLine,
    DependencyConsumer &DepConsumer, DependencyActionController &Controller,
    DiagnosticConsumer &DiagConsumer,
    llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFS) {
  return computeDependencies(WorkingDirectory,
                             ArrayRef<ArrayRef<std::string>>(CommandLine),
                             DepConsumer, Controller, DiagConsumer, OverlayFS);
}

bool DependencyScanningWorker::computeDependencies(
    StringRef WorkingDirectory, ArrayRef<ArrayRef<std::string>> CommandLines,
    DependencyConsumer &DepConsumer, DependencyActionController &Controller,
    DiagnosticConsumer &DiagConsumer,
    llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFS) {
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS = nullptr;
  if (OverlayFS) {
#ifndef NDEBUG
    bool SawDepFS = false;
    OverlayFS->visit(
        [&](llvm::vfs::FileSystem &VFS) { SawDepFS |= &VFS == DepFS.get(); });
    assert(SawDepFS && "OverlayFS not based on DepFS");
#endif
    FS = std::move(OverlayFS);
  } else {
    FS = DepFS;
    FS->setCurrentWorkingDirectory(WorkingDirectory);
  }

  DependencyScanningAction Action(Service, WorkingDirectory, DepConsumer,
                                  Controller, DepFS);

  const bool Success = llvm::all_of(CommandLines, [&](const auto &Cmd) {
    if (StringRef(Cmd[1]) != "-cc1") {
      // Non-clang command. Just pass through to the dependency consumer.
      DepConsumer.handleBuildCommand(
          {Cmd.front(), {Cmd.begin() + 1, Cmd.end()}});
      return true;
    }

    auto DiagEngineWithDiagOpts =
        DiagnosticsEngineWithDiagOpts(Cmd, FS, DiagConsumer);
    auto &Diags = *DiagEngineWithDiagOpts.DiagEngine;

    // Create an invocation that uses the underlying file system to ensure that
    // any file system requests that are made by the driver do not go through
    // the dependency scanning filesystem.
    return createAndRunToolInvocation(Cmd, Action, FS, PCHContainerOps, Diags);
  });

  // Ensure finish() is called even if we never reached ExecuteAction().
  if (!Action.hasDiagConsumerFinished())
    DiagConsumer.finish();

  return Success && Action.hasScanned();
}

bool DependencyScanningWorker::initializeCompilerInstanceWithContext(
    StringRef CWD, ArrayRef<std::string> CommandLine, DiagnosticConsumer &DC) {
  auto [OverlayFS, ModifiedCommandLine] =
      initVFSForByNameScanning(DepFS, CommandLine, CWD, "ScanningByName");
  auto DiagEngineWithCmdAndOpts =
      std::make_unique<DiagnosticsEngineWithDiagOpts>(ModifiedCommandLine,
                                                      OverlayFS, DC);
  return initializeCompilerInstanceWithContext(
      CWD, ModifiedCommandLine, std::move(DiagEngineWithCmdAndOpts), OverlayFS);
}

bool DependencyScanningWorker::initializeCompilerInstanceWithContext(
    StringRef CWD, ArrayRef<std::string> CommandLine,
    std::unique_ptr<DiagnosticsEngineWithDiagOpts> DiagEngineWithDiagOpts,
    IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFS) {
  CIWithContext =
      std::make_unique<CompilerInstanceWithContext>(*this, CWD, CommandLine);
  return CIWithContext->initialize(std::move(DiagEngineWithDiagOpts),
                                   OverlayFS);
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

std::pair<IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem>,
          std::vector<std::string>>
dependencies::initVFSForTUBufferScanning(
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS,
    ArrayRef<std::string> CommandLine, StringRef WorkingDirectory,
    llvm::MemoryBufferRef TUBuffer) {
  // Reset what might have been modified in the previous worker invocation.
  BaseFS->setCurrentWorkingDirectory(WorkingDirectory);

  auto OverlayFS =
      llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(BaseFS);
  auto InMemoryFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  InMemoryFS->setCurrentWorkingDirectory(WorkingDirectory);
  auto InputPath = TUBuffer.getBufferIdentifier();
  InMemoryFS->addFile(
      InputPath, 0, llvm::MemoryBuffer::getMemBufferCopy(TUBuffer.getBuffer()));
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> InMemoryOverlay = InMemoryFS;

  OverlayFS->pushOverlay(InMemoryOverlay);
  std::vector<std::string> ModifiedCommandLine(CommandLine);
  ModifiedCommandLine.emplace_back(InputPath);

  return std::make_pair(OverlayFS, ModifiedCommandLine);
}

std::pair<IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem>,
          std::vector<std::string>>
dependencies::initVFSForByNameScanning(
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS,
    ArrayRef<std::string> CommandLine, StringRef WorkingDirectory,
    StringRef ModuleName) {
  // Reset what might have been modified in the previous worker invocation.
  BaseFS->setCurrentWorkingDirectory(WorkingDirectory);

  // If we're scanning based on a module name alone, we don't expect the client
  // to provide us with an input file. However, the driver really wants to have
  // one. Let's just make it up to make the driver happy.
  auto OverlayFS =
      llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(BaseFS);
  auto InMemoryFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  InMemoryFS->setCurrentWorkingDirectory(WorkingDirectory);
  SmallString<128> FakeInputPath;
  // TODO: We should retry the creation if the path already exists.
  llvm::sys::fs::createUniquePath(ModuleName + "-%%%%%%%%.input", FakeInputPath,
                                  /*MakeAbsolute=*/false);
  InMemoryFS->addFile(FakeInputPath, 0, llvm::MemoryBuffer::getMemBuffer(""));
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> InMemoryOverlay = InMemoryFS;
  OverlayFS->pushOverlay(InMemoryOverlay);

  std::vector<std::string> ModifiedCommandLine(CommandLine);
  ModifiedCommandLine.emplace_back(FakeInputPath);

  return std::make_pair(OverlayFS, ModifiedCommandLine);
}
