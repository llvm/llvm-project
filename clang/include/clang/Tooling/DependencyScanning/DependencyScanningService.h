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
#include "llvm/ADT/BitmaskEnum.h"
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

  /// This outputs the full clang module dependency graph suitable for use for
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

  /// This outputs the dependency graph for standard c++ modules in P1689R5
  /// format.
  P1689,
};

#define DSS_LAST_BITMASK_ENUM(Id)                                              \
  LLVM_MARK_AS_BITMASK_ENUM(Id), All = llvm::NextPowerOf2(Id) - 1

enum class ScanningOptimizations {
  None = 0,

  /// Remove unused header search paths including header maps.
  HeaderSearch = 1,

  /// Remove warnings from system modules.
  SystemWarnings = (1 << 1),

  /// Remove unused -ivfsoverlay arguments.
  VFS = (1 << 2),

  /// Canonicalize -D and -U options.
  Macros = (1 << 3),

  /// Ignore the compiler's working directory if it is safe.
  IgnoreCWD = (1 << 4),

  DSS_LAST_BITMASK_ENUM(IgnoreCWD),
  Default = All,
  FullIncludeTreeIrrelevant = HeaderSearch | VFS,
};

#undef DSS_LAST_BITMASK_ENUM

/// The dependency scanning service contains shared configuration and state that
/// is used by the individual dependency scanning workers.
class DependencyScanningService {
public:
  DependencyScanningService(
      ScanningMode Mode, ScanningOutputFormat Format, CASOptions CASOpts,
      std::shared_ptr<llvm::cas::ObjectStore> CAS,
      std::shared_ptr<llvm::cas::ActionCache> Cache,
      IntrusiveRefCntPtr<llvm::cas::CachingOnDiskFileSystem> SharedFS,
      ScanningOptimizations OptimizeArgs = ScanningOptimizations::Default,
      bool EagerLoadModules = false, bool TraceVFS = false);

  ScanningMode getMode() const { return Mode; }

  ScanningOutputFormat getFormat() const { return Format; }

  ScanningOptimizations getOptimizeArgs() const { return OptimizeArgs; }

  bool shouldEagerLoadModules() const { return EagerLoadModules; }

  bool shouldTraceVFS() const { return TraceVFS; }

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
  ScanningOptimizations OptimizeArgs;
  /// Whether to set up command-lines to load PCM files eagerly.
  const bool EagerLoadModules;
  /// Whether to trace VFS accesses.
  const bool TraceVFS;
  /// Shared CachingOnDiskFileSystem. Set to nullptr to not use CAS dependency
  /// scanning.
  IntrusiveRefCntPtr<llvm::cas::CachingOnDiskFileSystem> SharedFS;
  /// The global file system cache.
  std::optional<DependencyScanningFilesystemSharedCache> SharedCache;
};

} // end namespace dependencies
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNINGSERVICE_H
