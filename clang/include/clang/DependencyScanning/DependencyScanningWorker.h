//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYSCANNINGWORKER_H
#define LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYSCANNINGWORKER_H

#include "clang/Basic/AtomicLineLogger.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LLVM.h"
#include "clang/DependencyScanning/DependencyScannerImpl.h"
#include "clang/DependencyScanning/DependencyScanningService.h"
#include "clang/DependencyScanning/ModuleDepCollector.h"
#include "clang/Frontend/PCHContainerOperations.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <optional>
#include <string>

namespace clang {

class DependencyOutputOptions;

namespace dependencies {

class DependencyConsumer;
class DependencyScanningWorkerFilesystem;

/// An individual dependency scanning worker that is able to run on its own
/// thread.
///
/// The worker computes the dependencies for the input files by preprocessing
/// sources either using a fast mode where the source files are minimized, or
/// using the regular processing run.
class DependencyScanningWorker {
public:
  /// Construct a dependency scanning worker.
  ///
  /// @param Service The parent service. Must outlive the worker.
  DependencyScanningWorker(DependencyScanningService &Service);

  ~DependencyScanningWorker();

  /// Run the dependency scanning tool for all given frontend command-lines,
  /// and report the discovered dependencies to the provided consumer. The
  /// \c OverlayFS will be used to call \c makeEffectiveVFS().
  ///
  /// \returns false if any errors occurred (with diagnostics reported to
  /// \c DiagConsumer), true otherwise.
  bool computeDependencies(
      StringRef WorkingDirectory, ArrayRef<ArrayRef<std::string>> CommandLines,
      DependencyConsumer &DepConsumer, DependencyActionController &Controller,
      DiagnosticConsumer &DiagConsumer,
      IntrusiveRefCntPtr<llvm::vfs::FileSystem> OverlayFS = nullptr);

  /// By-name scanning over a single cc1 command line. Builds a scanning session
  /// local to this call, then pulls module names from \p getNextName until it
  /// returns std::nullopt. Results flow to \p DepConsumer (per-name status via
  /// finishQuery); diagnostics flow to \p DiagConsumer.
  /// \returns false if session setup failed, true otherwise.
  bool computeDependenciesByName(
      StringRef CWD, ArrayRef<std::string> CC1CommandLine,
      IntrusiveRefCntPtr<llvm::vfs::FileSystem> OverlayFS,
      DiagnosticConsumer &DiagConsumer, DependencyActionController &Controller,
      llvm::function_ref<std::optional<std::string>()> getNextName,
      DependencyConsumer &DepConsumer);

  /// Creates the effective VFS that will be used for the scan.
  ///
  /// If provided, OverlayFS will be overlaid on top of the Worker's dependency
  /// scanning file-system and can be used to provide any input specified on the
  /// command-line as in-memory file. If no overlay file-system is provided, the
  /// Worker's dependency scanning file-system is used directly.
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> makeEffectiveVFS(
      StringRef WorkingDirectory,
      IntrusiveRefCntPtr<llvm::vfs::FileSystem> OverlayFS = nullptr) const;

  /// Returns the worker tracing VFS, if it was requested via the service.
  llvm::vfs::TracingFileSystem *getTracingVFS() const {
    return TracingFS.get();
  }

  // MaxNumOfByNameQueries is the upper limit of the number of names the by-name
  // scanning API (computeDependenciesByName) can drain per call. At the time of
  // this commit, the estimated number of total unique importable names is
  // around 3000 from Apple's SDKs. We usually import them in parallel, so it is
  // unlikely that all names are scanned by the same worker. Therefore the 64k
  // (20x our estimate) size is sufficient to hold the unique source locations
  // to report diagnostics per worker.
  static const int32_t MaxNumOfByNameQueries = 1 << 16;

private:
  /// The parent dependency scanning service.
  DependencyScanningService &Service;
  std::shared_ptr<PCHContainerOperations> PCHContainerOps;
  /// This is the caching (and optionally dependency-directives-providing) VFS
  /// overlaid on top of the base VFS.
  IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> DepFS;
  /// The tracing VFS overlaid on top of the base VFS.
  IntrusiveRefCntPtr<llvm::vfs::TracingFileSystem> TracingFS;

  friend class CompilerInstanceWithContext;
};
} // end namespace dependencies
} // end namespace clang

#endif // LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYSCANNINGWORKER_H
