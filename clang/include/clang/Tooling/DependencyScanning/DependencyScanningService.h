//===- DependencyScanningService.h - clang-scan-deps service ===-*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_DEPENDENCY_SCANNING_SERVICE_H
#define LLVM_CLANG_TOOLING_DEPENDENCY_SCANNING_SERVICE_H

#include "clang/Tooling/DependencyScanning/DependencyScanningFilesystem.h"

namespace clang {
namespace tooling {
namespace dependencies {

/// The mode in which the dependency scanner will operate to find the
/// dependencies.
enum class ScanningMode {
  /// This mode is used to compute the dependencies by running the preprocessor
  /// over
  /// the unmodified source files.
  CanonicalPreprocessing,

  /// This mode is used to compute the dependencies by running the preprocessor
  /// over
  /// the source files that have been minimized to contents that might affect
  /// the dependencies.
  MinimizedSourcePreprocessing
};

/// The dependency scanning service contains the shared state that is used by
/// the invidual dependency scanning workers.
class DependencyScanningService {
public:
  DependencyScanningService(ScanningMode Mode);

  ScanningMode getMode() const { return Mode; }

  DependencyScanningFilesystemSharedCache &getSharedCache() {
    return SharedCache;
  }

private:
  const ScanningMode Mode;
  /// The global file system cache.
  DependencyScanningFilesystemSharedCache SharedCache;
};

} // end namespace dependencies
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_DEPENDENCY_SCANNING_SERVICE_H
