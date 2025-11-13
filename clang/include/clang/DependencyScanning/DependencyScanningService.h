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
#include "llvm/Support/Chrono.h"

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

struct DependencyScanningServiceOptions {
  DependencyScanningServiceOptions(ScanningMode Mode,
                                   ScanningOutputFormat Format)
      : Mode(Mode), Format(Format) {}

  ScanningMode Mode;
  ScanningOutputFormat Format;
  /// How to optimize the modules' command-line arguments.
  ScanningOptimizations OptimizeArgs = ScanningOptimizations::Default;
  /// Whether to set up command-lines to load PCM files eagerly.
  bool EagerLoadModules = false;
  /// Whether to trace VFS accesses.
  bool TraceVFS = false;
  std::time_t BuildSessionTimestamp =
      llvm::sys::toTimeT(std::chrono::system_clock::now());
};

/// The dependency scanning service contains shared configuration and state that
/// is used by the individual dependency scanning workers.
class DependencyScanningService {
public:
  DependencyScanningService(const DependencyScanningServiceOptions &Options);

  ScanningMode getMode() const { return Options.Mode; }

  ScanningOutputFormat getFormat() const { return Options.Format; }

  ScanningOptimizations getOptimizeArgs() const { return Options.OptimizeArgs; }

  bool shouldEagerLoadModules() const { return Options.EagerLoadModules; }

  bool shouldTraceVFS() const { return Options.TraceVFS; }

  DependencyScanningFilesystemSharedCache &getSharedCache() {
    return SharedCache;
  }

  ModuleCacheEntries &getModuleCacheEntries() { return ModCacheEntries; }

  std::time_t getBuildSessionTimestamp() const {
    return Options.BuildSessionTimestamp;
  }

private:
  const DependencyScanningServiceOptions Options;

  /// The global file system cache.
  DependencyScanningFilesystemSharedCache SharedCache;
  /// The global module cache entries.
  ModuleCacheEntries ModCacheEntries;
};

} // end namespace dependencies
} // end namespace clang

#endif // LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYSCANNINGSERVICE_H
