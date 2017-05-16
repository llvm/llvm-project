//===--- GlobalCompilationDatabase.h ----------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_GLOBALCOMPILATIONDATABASE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_GLOBALCOMPILATIONDATABASE_H

#include "Path.h"
#include "llvm/ADT/StringMap.h"
#include <memory>
#include <mutex>
#include <vector>

namespace clang {

namespace tooling {
class CompilationDatabase;
struct CompileCommand;
} // namespace tooling

namespace clangd {

/// Provides compilation arguments used for building ClangdUnit.
class GlobalCompilationDatabase {
public:
  virtual ~GlobalCompilationDatabase() = default;

  virtual std::vector<tooling::CompileCommand>
  getCompileCommands(PathRef File) = 0;

  /// FIXME(ibiryukov): add facilities to track changes to compilation flags of
  /// existing targets.
};

/// Gets compile args from tooling::CompilationDatabases built for parent
/// directories.
class DirectoryBasedGlobalCompilationDatabase
    : public GlobalCompilationDatabase {
public:
  std::vector<tooling::CompileCommand>
  getCompileCommands(PathRef File) override;

private:
  tooling::CompilationDatabase *getCompilationDatabase(PathRef File);

  std::mutex Mutex;
  /// Caches compilation databases loaded from directories(keys are
  /// directories).
  llvm::StringMap<std::unique_ptr<clang::tooling::CompilationDatabase>>
      CompilationDatabases;
};
} // namespace clangd
} // namespace clang

#endif
