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
#include "clang/DependencyScanning/CompilerInstanceWithContext.h"
#include "clang/DependencyScanning/DependencyConsumer.h"
#include "clang/DependencyScanning/DependencyScannerImpl.h"
#include "clang/DependencyScanning/DependencyScanningUtils.h"
#include "clang/Serialization/ObjectFilePCHContainerReader.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace clang;
using namespace dependencies;

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
