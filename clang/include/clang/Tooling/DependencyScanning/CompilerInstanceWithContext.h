//===- CompilerInstanceWithContext.h - clang scanning compiler instance ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_COMPILERINSTANCEWITHCONTEXT_H
#define LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_COMPILERINSTANCEWITHCONTEXT_H

#include "clang/Basic/FileManager.h"
#include "clang/Basic/LLVM.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Serialization/ModuleCache.h"
#include "clang/Tooling/DependencyScanning/ModuleDepCollector.h"
#include <string>
#include <vector>

namespace clang {
namespace tooling {
namespace dependencies {

// Forward declarations.
class DependencyScanningWorker;
class DependencyConsumer;
class DependencyActionController;

class CompilerInstanceWithContext {
  // Context
  DependencyScanningWorker &Worker;
  llvm::StringRef CWD;
  std::vector<std::string> CommandLine;
  static const uint64_t MAX_NUM_NAMES = (1 << 12);
  static const std::string FakeFileBuffer;

  // Context - file systems
  llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFS;
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFS;
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> InMemoryOverlay;

  // Context - Diagnostics engine, file manager and source mamanger.
  std::string DiagnosticOutput;
  llvm::raw_string_ostream DiagnosticsOS;
  std::unique_ptr<TextDiagnosticPrinter> DiagPrinter;
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags;
  std::unique_ptr<FileManager> FileMgr;
  std::unique_ptr<SourceManager> SrcMgr;

  // Context - compiler invocation
  std::unique_ptr<clang::driver::Driver> Driver;
  std::unique_ptr<clang::driver::Compilation> Compilation;
  std::unique_ptr<CompilerInvocation> Invocation;
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFSFromCompilerInvocation;

  // Context - output options
  std::unique_ptr<DependencyOutputOptions> OutputOpts;

  // Context - stable directory handling
  llvm::SmallVector<StringRef> StableDirs;
  PrebuiltModulesAttrsMap PrebuiltModuleVFSMap;

  // Compiler Instance
  IntrusiveRefCntPtr<ModuleCache> ModCache;
  std::unique_ptr<CompilerInstance> CIPtr;

  //   // Source location offset.
  int32_t SrcLocOffset = 0;

public:
  CompilerInstanceWithContext(DependencyScanningWorker &Worker, StringRef CWD,
                              const std::vector<std::string> &CMD)
      : Worker(Worker), CWD(CWD), CommandLine(CMD),
        DiagnosticsOS(DiagnosticOutput) {};

  llvm::Error initialize();
  llvm::Error computeDependencies(StringRef ModuleName,
                                  DependencyConsumer &Consumer,
                                  DependencyActionController &Controller);
  llvm::Error finalize();
};
} // namespace dependencies
} // namespace tooling
} // namespace clang

#endif
