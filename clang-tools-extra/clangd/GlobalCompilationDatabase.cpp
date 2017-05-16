//===--- GlobalCompilationDatabase.cpp --------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include "GlobalCompilationDatabase.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace clang::clangd;
using namespace clang;

std::vector<tooling::CompileCommand>
DirectoryBasedGlobalCompilationDatabase::getCompileCommands(PathRef File) {
  std::vector<tooling::CompileCommand> Commands;

  auto CDB = getCompilationDatabase(File);
  if (!CDB)
    return {};
  return CDB->getCompileCommands(File);
}

tooling::CompilationDatabase *
DirectoryBasedGlobalCompilationDatabase::getCompilationDatabase(PathRef File) {
  std::lock_guard<std::mutex> Lock(Mutex);

  namespace path = llvm::sys::path;

  assert((path::is_absolute(File, path::Style::posix) ||
          path::is_absolute(File, path::Style::windows)) &&
         "path must be absolute");

  for (auto Path = path::parent_path(File); !Path.empty();
       Path = path::parent_path(Path)) {

    auto CachedIt = CompilationDatabases.find(Path);
    if (CachedIt != CompilationDatabases.end())
      return CachedIt->second.get();
    std::string Error;
    auto CDB = tooling::CompilationDatabase::loadFromDirectory(Path, Error);
    if (!CDB) {
      if (!Error.empty()) {
        // FIXME(ibiryukov): logging
        // Output.log("Error when trying to load compilation database from " +
        //            Twine(Path) + ": " + Twine(Error) + "\n");
      }
      continue;
    }

    // FIXME(ibiryukov): Invalidate cached compilation databases on changes
    auto result = CDB.get();
    CompilationDatabases.insert(std::make_pair(Path, std::move(CDB)));
    return result;
  }

  // FIXME(ibiryukov): logging
  // Output.log("Failed to find compilation database for " + Twine(File) +
  // "\n");
  return nullptr;
}
