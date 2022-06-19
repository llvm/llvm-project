//===- DependencyScanningService.h - clang-scan-deps service ===-*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNINGSERVICE_H
#define LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNINGSERVICE_H

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
};

/// The dependency scanning service contains the shared state that is used by
/// the invidual dependency scanning workers.
class DependencyScanningService {
public:
  DependencyScanningService(ScanningMode Mode, ScanningOutputFormat Format,
                            bool ReuseFileManager = true,
                            bool OptimizeArgs = false);

  ScanningMode getMode() const { return Mode; }

  ScanningOutputFormat getFormat() const { return Format; }

  bool canReuseFileManager() const { return ReuseFileManager; }

  bool canOptimizeArgs() const { return OptimizeArgs; }

  DependencyScanningFilesystemSharedCache &getSharedCache() {
    return SharedCache;
  }

private:
  const ScanningMode Mode;
  const ScanningOutputFormat Format;
  const bool ReuseFileManager;
  /// Whether to optimize the modules' command-line arguments.
  const bool OptimizeArgs;
  /// The global file system cache.
  DependencyScanningFilesystemSharedCache SharedCache;
};

} // end namespace dependencies
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNINGSERVICE_H
