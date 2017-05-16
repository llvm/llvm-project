//===--- ClangdUnitStore.cpp - A ClangdUnits container -----------*-C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ClangdUnitStore.h"
#include "llvm/Support/Path.h"

using namespace clang::clangd;
using namespace clang;

void ClangdUnitStore::removeUnitIfPresent(PathRef File) {
  std::lock_guard<std::mutex> Lock(Mutex);

  auto It = OpenedFiles.find(File);
  if (It == OpenedFiles.end())
    return;
  OpenedFiles.erase(It);
}

std::vector<tooling::CompileCommand> ClangdUnitStore::getCompileCommands(GlobalCompilationDatabase &CDB, PathRef File) {
  std::vector<tooling::CompileCommand> Commands = CDB.getCompileCommands(File);
  if (Commands.empty()) {
    // Add a fake command line if we know nothing.
    Commands.push_back(tooling::CompileCommand(
        llvm::sys::path::parent_path(File), llvm::sys::path::filename(File),
        {"clang", "-fsyntax-only", File.str()}, ""));
  }
  return Commands;
}
