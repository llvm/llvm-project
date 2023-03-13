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
#include "llvm/CAS/ActionCache.h"

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

  /// This emits the full dependency graph but with include tree.
  FullIncludeTree,
};

/// The dependency scanning service contains shared configuration and state that
/// is used by the individual dependency scanning workers.
class DependencyScanningService {
public:
  DependencyScanningService(
      ScanningMode Mode, ScanningOutputFormat Format, CASOptions CASOpts,
      std::shared_ptr<llvm::cas::ObjectStore> CAS,
      std::shared_ptr<llvm::cas::ActionCache> Cache,
      IntrusiveRefCntPtr<llvm::cas::CachingOnDiskFileSystem> SharedFS,
      bool OptimizeArgs = false, bool EagerLoadModules = false);

  ScanningMode getMode() const { return Mode; }

  ScanningOutputFormat getFormat() const { return Format; }

  bool canOptimizeArgs() const { return OptimizeArgs; }

  bool shouldEagerLoadModules() const { return EagerLoadModules; }

  DependencyScanningFilesystemSharedCache &getSharedCache() {
    assert(!SharedFS && "Expected not to have a CASFS");
    assert(SharedCache && "Expected a shared cache");
    return *SharedCache;
  }

  const CASOptions &getCASOpts() const { return CASOpts; }

  std::shared_ptr<llvm::cas::ObjectStore> getCAS() const { return CAS; }
  std::shared_ptr<llvm::cas::ActionCache> getCache() const { return Cache; }

  llvm::cas::CachingOnDiskFileSystem &getSharedFS() { return *SharedFS; }

  bool useCASFS() const { return (bool)SharedFS; }

private:
  const ScanningMode Mode;
  const ScanningOutputFormat Format;
  CASOptions CASOpts;
  std::shared_ptr<llvm::cas::ObjectStore> CAS;
  std::shared_ptr<llvm::cas::ActionCache> Cache;
  /// Whether to optimize the modules' command-line arguments.
  const bool OptimizeArgs;

  /// Shared CachingOnDiskFileSystem. Set to nullptr to not use CAS dependency
  /// scanning.
  IntrusiveRefCntPtr<llvm::cas::CachingOnDiskFileSystem> SharedFS;

  /// Whether to set up command-lines to load PCM files eagerly.
  const bool EagerLoadModules;

  /// The global file system cache.
  Optional<DependencyScanningFilesystemSharedCache> SharedCache;
};

} // end namespace dependencies
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNINGSERVICE_H
