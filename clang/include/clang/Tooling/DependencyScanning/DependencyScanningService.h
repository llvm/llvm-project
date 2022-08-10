//===- DependencyScanningService.h - clang-scan-deps service ===-*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNINGSERVICE_H
#define LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNINGSERVICE_H

#include "clang/CAS/CASOptions.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningCASFilesystem.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningFilesystem.h"

namespace clang {
namespace tooling {
namespace dependencies {

/// The mode in which the dependency scanner will operate to find the
/// dependencies.
enum class ScanningMode {
  /// This mode is used to compute the dependencies by running the preprocessor
  /// over the source files.
  CanonicalPreprocessing,

  /// This mode is used to compute the dependencies by running the preprocessor
  /// with special kind of lexing after scanning header and source files to get
  /// the minimum necessary preprocessor directives for evaluating includes.
  DependencyDirectivesScan,
};

/// The format that is output by the dependency scanner.
enum class ScanningOutputFormat {
  /// This is the Makefile compatible dep format. This will include all of the
  /// deps necessary for an implicit modules build, but won't include any
  /// intermodule dependency information.
  Make,

  /// This outputs the full module dependency graph suitable for use for
  /// explicitly building modules.
  Full,

  /// This emits the CAS ID of the scanned files.
  Tree,

  /// This emits the full dependency graph but with CAS tree embedded as file
  /// dependency.
  FullTree,

  /// This emits the CAS ID of the include tree.
  IncludeTree,
};

/// The dependency scanning service contains the shared state that is used by
/// the invidual dependency scanning workers.
class DependencyScanningService {
public:
  DependencyScanningService(
      ScanningMode Mode, ScanningOutputFormat Format, CASOptions CASOpts,
      IntrusiveRefCntPtr<llvm::cas::CachingOnDiskFileSystem> SharedFS,
      bool ReuseFileManager = true, bool OptimizeArgs = false);

  ~DependencyScanningService();

  ScanningMode getMode() const { return Mode; }

  ScanningOutputFormat getFormat() const { return Format; }

  const CASOptions &getCASOpts() const { return CASOpts; }

  bool canReuseFileManager() const { return ReuseFileManager; }

  bool canOptimizeArgs() const { return OptimizeArgs; }

  DependencyScanningFilesystemSharedCache &getSharedCache() {
    assert(!SharedFS && "Expected not to have a CASFS");
    assert(SharedCache && "Expected a shared cache");
    return *SharedCache;
  }

  llvm::cas::CachingOnDiskFileSystem &getSharedFS() { return *SharedFS; }

  bool useCASScanning() const { return (bool)SharedFS; }

private:
  const ScanningMode Mode;
  const ScanningOutputFormat Format;
  CASOptions CASOpts;
  const bool ReuseFileManager;
  /// Whether to optimize the modules' command-line arguments.
  const bool OptimizeArgs;
  /// Shared CachingOnDiskFileSystem. Set to nullptr to not use CAS dependency
  /// scanning.
  IntrusiveRefCntPtr<llvm::cas::CachingOnDiskFileSystem> SharedFS;
  /// The global file system cache.
  Optional<DependencyScanningFilesystemSharedCache> SharedCache;
};

} // end namespace dependencies
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNINGSERVICE_H
