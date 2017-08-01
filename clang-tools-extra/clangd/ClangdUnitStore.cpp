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

std::shared_ptr<CppFile> CppFileCollection::removeIfPresent(PathRef File) {
  std::lock_guard<std::mutex> Lock(Mutex);

  auto It = OpenedFiles.find(File);
  if (It == OpenedFiles.end())
    return nullptr;

  std::shared_ptr<CppFile> Result = It->second;
  OpenedFiles.erase(It);
  return Result;
}

tooling::CompileCommand
CppFileCollection::getCompileCommand(GlobalCompilationDatabase &CDB,
                                     PathRef File, PathRef ResourceDir) {
  std::vector<tooling::CompileCommand> Commands = CDB.getCompileCommands(File);
  if (Commands.empty())
    // Add a fake command line if we know nothing.
    Commands.push_back(getDefaultCompileCommand(File));

  // Inject the resource dir.
  // FIXME: Don't overwrite it if it's already there.
  Commands.front().CommandLine.push_back("-resource-dir=" +
                                         std::string(ResourceDir));
  return std::move(Commands.front());
}
