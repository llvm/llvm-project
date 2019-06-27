//===- DependencyScanningWorker.h - clang-scan-deps worker ===---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_DEPENDENCY_SCANNING_WORKER_H
#define LLVM_CLANG_TOOLING_DEPENDENCY_SCANNING_WORKER_H

#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/LLVM.h"
#include "clang/Frontend/PCHContainerOperations.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include <string>

namespace clang {
namespace tooling {
namespace dependencies {

/// An individual dependency scanning worker that is able to run on its own
/// thread.
///
/// The worker computes the dependencies for the input files by preprocessing
/// sources either using a fast mode where the source files are minimized, or
/// using the regular processing run.
class DependencyScanningWorker {
public:
  DependencyScanningWorker();

  /// Print out the dependency information into a string using the dependency
  /// file format that is specified in the options (-MD is the default) and
  /// return it.
  ///
  /// \returns A \c StringError with the diagnostic output if clang errors
  /// occurred, dependency file contents otherwise.
  llvm::Expected<std::string> getDependencyFile(const std::string &Input,
                                                StringRef WorkingDirectory,
                                                const CompilationDatabase &CDB);

private:
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts;
  std::shared_ptr<PCHContainerOperations> PCHContainerOps;

  /// The file system that is used by each worker when scanning for
  /// dependencies. This filesystem persists accross multiple compiler
  /// invocations.
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> WorkerFS;
};

} // end namespace dependencies
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_DEPENDENCY_SCANNING_WORKER_H
