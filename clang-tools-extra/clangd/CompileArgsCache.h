//===--- CompileArgsCache.h -------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_COMPILEARGSCACHE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_COMPILEARGSCACHE_H

#include "GlobalCompilationDatabase.h"
#include "Path.h"
#include "clang/Tooling/CompilationDatabase.h"

namespace clang {
namespace clangd {

/// A helper class used by ClangdServer to get compile commands from CDB.
/// Also caches CompileCommands produced by compilation database on per-file
/// basis. This avoids queries to CDB that can be much more expensive than a
/// table lookup.
class CompileArgsCache {
public:
  CompileArgsCache(GlobalCompilationDatabase &CDB, Path ResourceDir);

  /// Gets compile command for \p File from cache or CDB if it's not in the
  /// cache.
  tooling::CompileCommand getCompileCommand(PathRef File);

  /// Removes a cache entry for \p File, if it's present in the cache.
  void invalidate(PathRef File);

private:
  GlobalCompilationDatabase &CDB;
  const Path ResourceDir;
  llvm::StringMap<tooling::CompileCommand> Cached;
};

} // namespace clangd
} // namespace clang
#endif
