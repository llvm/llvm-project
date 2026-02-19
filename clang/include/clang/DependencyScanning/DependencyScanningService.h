//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYSCANNINGSERVICE_H
#define LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYSCANNINGSERVICE_H

#include "clang/DependencyScanning/DependencyScanningFilesystem.h"
#include "clang/DependencyScanning/InProcessModuleCache.h"
#include "llvm/ADT/BitmaskEnum.h"

namespace clang {
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

  // The build system needs to be aware that the current working
  // directory is ignored. Without a good way of notifying the build
  // system, it is less risky to default to off.
  Default = All & (~IgnoreCWD)
};

#undef DSS_LAST_BITMASK_ENUM

/// The configuration knobs for the dependency scanning service.
struct DependencyScanningServiceOptions {
  DependencyScanningServiceOptions();

  /// The function invoked to create each worker's VFS. This function and the
  /// VFS itself must be thread-safe whenever using multiple workers
  /// concurrently or whenever \c AsyncScanModules is true.
  std::function<IntrusiveRefCntPtr<llvm::vfs::FileSystem>()>
      MakeVFS; // = [] { return llvm::vfs::createPhysicalFileSystem(); }
  /// Whether to use optimized dependency directive scan or full preprocessing.
  ScanningMode Mode = ScanningMode::DependencyDirectivesScan;
  /// What output format are we expected to produce.
  ScanningOutputFormat Format = ScanningOutputFormat::Full;
  /// How to optimize resulting explicit module command lines.
  ScanningOptimizations OptimizeArgs = ScanningOptimizations::Default;
  /// Whether the resulting command lines should load explicit PCMs eagerly.
  bool EagerLoadModules = false;
  /// Whether to trace VFS accesses during the scan.
  bool TraceVFS = false;
  /// Whether to scan modules asynchronously.
  bool AsyncScanModules = false;
  /// The build session timestamp for validate-once-per-build-session logic.
  std::time_t BuildSessionTimestamp; // = std::chrono::system_clock::now();
};

/// The dependency scanning service contains shared configuration and state that
/// is used by the individual dependency scanning workers.
class DependencyScanningService {
public:
  explicit DependencyScanningService(DependencyScanningServiceOptions Opts)
      : Opts(std::move(Opts)) {}

  const DependencyScanningServiceOptions &getOpts() const { return Opts; }

  DependencyScanningFilesystemSharedCache &getSharedCache() {
    return SharedCache;
  }

  ModuleCacheEntries &getModuleCacheEntries() { return ModCacheEntries; }

private:
  /// The options customizing dependency scanning behavior.
  DependencyScanningServiceOptions Opts;
  /// The global file system cache.
  DependencyScanningFilesystemSharedCache SharedCache;
  /// The global module cache entries.
  ModuleCacheEntries ModCacheEntries;
};

} // end namespace dependencies
} // end namespace clang

#endif // LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYSCANNINGSERVICE_H
