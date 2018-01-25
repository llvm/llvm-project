//===--- CompileArgsCache.cpp --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include "CompileArgsCache.h"

namespace clang {
namespace clangd {
namespace {
tooling::CompileCommand getCompileCommand(GlobalCompilationDatabase &CDB,
                                          PathRef File, PathRef ResourceDir) {
  llvm::Optional<tooling::CompileCommand> C = CDB.getCompileCommand(File);
  if (!C) // FIXME: Suppress diagnostics? Let the user know?
    C = CDB.getFallbackCommand(File);

  // Inject the resource dir.
  // FIXME: Don't overwrite it if it's already there.
  C->CommandLine.push_back("-resource-dir=" + ResourceDir.str());
  return std::move(*C);
}
} // namespace

CompileArgsCache::CompileArgsCache(GlobalCompilationDatabase &CDB,
                                   Path ResourceDir)
    : CDB(CDB), ResourceDir(std::move(ResourceDir)) {}

tooling::CompileCommand CompileArgsCache::getCompileCommand(PathRef File) {
  auto It = Cached.find(File);
  if (It == Cached.end()) {
    It = Cached.insert({File, clangd::getCompileCommand(CDB, File, ResourceDir)})
             .first;
  }
  return It->second;
}

void CompileArgsCache::invalidate(PathRef File) { Cached.erase(File); }

} // namespace clangd
} // namespace clang
