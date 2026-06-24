//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DEPENDENCYSCANNING_COMPILERINSTANCEWITHCONTEXT_H
#define LLVM_CLANG_DEPENDENCYSCANNING_COMPILERINSTANCEWITHCONTEXT_H

#include "clang/DependencyScanning/DependencyScanningWorker.h"

namespace clang {
namespace dependencies {
class CompilerInstanceWithContext {
  // Context
  DependencyScanningWorker &Worker;
  llvm::StringRef CWD;
  std::vector<std::string> CommandLine;

  // Context - compiler invocation
  std::unique_ptr<CompilerInvocation> OriginalInvocation;

  // Context - output options
  std::unique_ptr<DependencyOutputOptions> OutputOpts;

  // Context - stable directory handling
  llvm::SmallVector<StringRef> StableDirs;
  PrebuiltModulesAttrsMap PrebuiltModuleASTMap;

  // Compiler Instance
  std::unique_ptr<CompilerInstance> CIPtr;

  // Source location offset.
  int32_t SrcLocOffset = 0;

  CompilerInstanceWithContext(DependencyScanningWorker &Worker, StringRef CWD,
                              ArrayRef<std::string> CMD)
      : Worker(Worker), CWD(CWD), CommandLine(CMD.begin(), CMD.end()) {}

  bool initialize(
      DependencyActionController &Controller,
      std::unique_ptr<DiagnosticsEngineWithDiagOpts> DiagEngineWithDiagOpts,
      IntrusiveRefCntPtr<llvm::vfs::FileSystem> OverlayFS);

public:
  /// @brief Initialize the tool's compiler instance from the cc1 commandline.
  /// @param Worker The dependency scanning worker to initialize the compiler
  ///        instance.
  /// @param CWD The current working directory.
  /// @param CC1CommandLine A cc1 command.
  /// @param DiagEngineWithDiagOpts The diagnostic engine used during scan.
  /// @param OverlayFS An overlay FS containing the input file, which may be
  ///        from an in-memory buffer.
  /// @param Controller A dependency action controller to gather some results.
  static std::optional<CompilerInstanceWithContext>
  initializeFromCC1Commandline(
      DependencyScanningWorker &Worker, StringRef CWD,
      ArrayRef<std::string> CC1CommandLine,
      std::unique_ptr<DiagnosticsEngineWithDiagOpts> DiagEngineWithDiagOpts,
      IntrusiveRefCntPtr<llvm::vfs::FileSystem> OverlayFS,
      DependencyActionController &Controller);

  bool computeDependencies(StringRef ModuleName, DependencyConsumer &Consumer,
                           DependencyActionController &Controller);

  // MaxNumOfQueries is the upper limit of the number of names the by-name
  // scanning API (computeDependencies) can support after a
  // CompilerInstanceWithContext is initialized. At the time of this commit, the
  // estimated number of total unique importable names is around 3000 from
  // Apple's SDKs. We usually import them in parallel, so it is unlikely that
  // all names are all scanned by the same dependency scanning worker. Therefore
  // the 64k (20x bigger than our estimate) size is sufficient to hold the
  // unique source locations to report diagnostics per worker.
  static const int32_t MaxNumOfQueries = 1 << 16;
};
} // namespace dependencies
} // namespace clang

#endif