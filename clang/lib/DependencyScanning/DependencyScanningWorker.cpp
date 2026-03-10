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
#include "llvm/CAS/CASProvidingFileSystem.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace clang;
using namespace dependencies;
using llvm::Error;

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

  if (Service.getOpts().TraceVFS)
    BaseFS = llvm::makeIntrusiveRefCnt<llvm::vfs::TracingFileSystem>(
        std::move(BaseFS));

  DepFS = llvm::makeIntrusiveRefCnt<DependencyScanningWorkerFilesystem>(
      Service, std::move(BaseFS));
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
                                  Controller, DepFS,
                                  /*EmitDependencyFile=*/false,
                                  /*DiagGenerationAsCompilation=*/false);

  const bool Success = llvm::all_of(CommandLines, [&](const auto &Cmd) {
    if (StringRef(Cmd[1]) != "-cc1") {
      // Non-clang command. Just pass through to the dependency consumer.
      DepConsumer.handleBuildCommand(
          {Cmd.front(), {Cmd.begin() + 1, Cmd.end()}, std::nullopt});
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

  return Success && Action.hasScanned();
}

void DependencyScanningWorker::computeDependenciesFromCompilerInvocation(
    std::shared_ptr<CompilerInvocation> Invocation, StringRef WorkingDirectory,
    DependencyConsumer &DepsConsumer, DependencyActionController &Controller,
    DiagnosticConsumer &DiagsConsumer, raw_ostream *VerboseOS,
    bool DiagGenerationAsCompilation) {
  DepFS->setCurrentWorkingDirectory(WorkingDirectory);

  // Adjust the invocation.
  auto &Frontend = Invocation->getFrontendOpts();
  Frontend.OutputFile = "/dev/null";
  Frontend.DisableFree = false;

  // // Reset dependency options.
  // Dependencies = DependencyOutputOptions();
  // Dependencies.IncludeSystemHeaders = true;
  // Dependencies.OutputFile = "/dev/null";

  // Make the output file path absolute relative to WorkingDirectory.
  std::string &DepFile = Invocation->getDependencyOutputOpts().OutputFile;
  if (!DepFile.empty() && !llvm::sys::path::is_absolute(DepFile)) {
    // FIXME: On Windows, WorkingDirectory is insufficient for making an
    // absolute path if OutputFile has a root name.
    llvm::SmallString<128> Path = StringRef(DepFile);
    llvm::sys::path::make_absolute(WorkingDirectory, Path);
    DepFile = Path.str().str();
  }

  // FIXME: EmitDependencyFile should only be set when it's for a real
  // compilation.
  DependencyScanningAction Action(
      Service, WorkingDirectory, DepsConsumer, Controller, DepFS,
      /*EmitDependencyFile=*/!DepFile.empty(), DiagGenerationAsCompilation,
      /*ModuleName=*/std::nullopt, VerboseOS);

  // Ignore result; we're just collecting dependencies.
  //
  // FIXME: will clients other than -cc1scand care?
  (void)Action.runInvocation("<clang>", std::move(Invocation), DepFS,
                             PCHContainerOps, &DiagsConsumer);
}
