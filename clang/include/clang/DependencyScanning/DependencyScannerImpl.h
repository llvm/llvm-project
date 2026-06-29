//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYSCANNERIMPL_H
#define LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYSCANNERIMPL_H

#include "clang/DependencyScanning/DependencyScanningFilesystem.h"
#include "clang/DependencyScanning/ModuleDepCollector.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <mutex>
#include <thread>

namespace clang {
class DiagnosticConsumer;

namespace dependencies {
class DependencyScanningService;
class DependencyScanningWorker;

class DependencyActionController;
class DependencyScanningWorkerFilesystem;

// Helper functions and data types.
std::unique_ptr<DiagnosticOptions>
createDiagOptions(ArrayRef<std::string> CommandLine);

struct DiagnosticsEngineWithDiagOpts {
  // We need to bound the lifetime of the DiagOpts used to create the
  // DiganosticsEngine with the DiagnosticsEngine itself.
  std::unique_ptr<DiagnosticOptions> DiagOpts;
  IntrusiveRefCntPtr<DiagnosticsEngine> DiagEngine;

  DiagnosticsEngineWithDiagOpts(ArrayRef<std::string> CommandLine,
                                IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
                                DiagnosticConsumer &DC);
};

struct TextDiagnosticsPrinterWithOutput {
  // We need to bound the lifetime of the data that supports the DiagPrinter
  // with it together so they have the same lifetime.
  std::string DiagnosticOutput;
  llvm::raw_string_ostream DiagnosticsOS;
  std::unique_ptr<DiagnosticOptions> DiagOpts;
  TextDiagnosticPrinter DiagPrinter;

  TextDiagnosticsPrinterWithOutput(ArrayRef<std::string> CommandLine)
      : DiagnosticsOS(DiagnosticOutput),
        DiagOpts(createDiagOptions(CommandLine)),
        DiagPrinter(DiagnosticsOS, *DiagOpts) {}
};

/// Manages (and terminates) the asynchronous compilation of modules.
class AsyncModuleCompiles {
  std::mutex Mutex;
  bool Stop = false;
  // FIXME: Have the service own a thread pool and use that instead.
  std::vector<std::thread> Compiles;

public:
  /// Registers the module compilation, unless this instance is about to be
  /// destroyed.
  void add(llvm::unique_function<void()> Compile) {
    std::lock_guard<std::mutex> Lock(Mutex);
    if (!Stop)
      Compiles.emplace_back(std::move(Compile));
  }

  ~AsyncModuleCompiles() {
    {
      // Prevent registration of further module compiles.
      std::lock_guard<std::mutex> Lock(Mutex);
      Stop = true;
    }

    // Wait for outstanding module compiles to finish.
    for (std::thread &Compile : Compiles)
      Compile.join();
  }
};

void runTUModulePrescan(CompilerInstance &PrescanCI,
                        DependencyScanningService &Service,
                        DependencyActionController &Controller,
                        AsyncModuleCompiles &Compiles);

std::unique_ptr<CompilerInvocation>
createCompilerInvocation(ArrayRef<std::string> CommandLine,
                         DiagnosticsEngine &Diags);

/// Canonicalizes command-line macro defines (e.g. removing "-DX -UX").
void canonicalizeDefines(PreprocessorOptions &PPOpts);

/// Creates a CompilerInvocation suitable for the dependency scanner.
std::shared_ptr<CompilerInvocation>
createScanCompilerInvocation(const CompilerInvocation &Invocation,
                             const DependencyScanningService &Service,
                             DependencyActionController &Controller);

/// Creates dependency output options to be reported to the dependency consumer,
/// deducing missing information if necessary.
std::unique_ptr<DependencyOutputOptions>
createDependencyOutputOptions(const CompilerInvocation &Invocation);

void initializeScanCompilerInstance(
    CompilerInstance &ScanInstance,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
    DiagnosticConsumer *DiagConsumer, DependencyScanningService &Service,
    IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> DepFS);

SmallVector<StringRef>
getInitialStableDirs(const CompilerInstance &ScanInstance);

std::optional<PrebuiltModulesAttrsMap>
computePrebuiltModulesASTMap(CompilerInstance &ScanInstance,
                             SmallVector<StringRef> &StableDirs);

/// Create the dependency collector that will collect the produced
/// dependencies. May return the created ModuleDepCollector depending
/// on the scanning format.
std::shared_ptr<ModuleDepCollector> initializeScanInstanceDependencyCollector(
    CompilerInstance &ScanInstance,
    std::unique_ptr<DependencyOutputOptions> DepOutputOpts,
    DependencyScanningService &Service, CompilerInvocation &Inv,
    DependencyActionController &Controller,
    PrebuiltModulesAttrsMap PrebuiltModulesASTMap,
    SmallVector<StringRef> &StableDirs);
} // namespace dependencies
} // namespace clang

#endif // LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYSCANNERIMPL_H
