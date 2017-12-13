//===--- GlobalCompilationDatabase.cpp --------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include "GlobalCompilationDatabase.h"
#include "Logger.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace clang {
namespace clangd {

tooling::CompileCommand
GlobalCompilationDatabase::getFallbackCommand(PathRef File) const {
  return tooling::CompileCommand(llvm::sys::path::parent_path(File),
                                 llvm::sys::path::filename(File),
                                 {"clang", File.str()},
                                 /*Output=*/"");
}

DirectoryBasedGlobalCompilationDatabase::
    DirectoryBasedGlobalCompilationDatabase(
        llvm::Optional<Path> CompileCommandsDir)
    : CompileCommandsDir(std::move(CompileCommandsDir)) {}

llvm::Optional<tooling::CompileCommand>
DirectoryBasedGlobalCompilationDatabase::getCompileCommand(PathRef File) const {
  if (auto CDB = getCompilationDatabase(File)) {
    auto Candidates = CDB->getCompileCommands(File);
    if (!Candidates.empty()) {
      addExtraFlags(File, Candidates.front());
      return std::move(Candidates.front());
    }
  }
  return llvm::None;
}

tooling::CompileCommand
DirectoryBasedGlobalCompilationDatabase::getFallbackCommand(
    PathRef File) const {
  auto C = GlobalCompilationDatabase::getFallbackCommand(File);
  addExtraFlags(File, C);
  return C;
}

void DirectoryBasedGlobalCompilationDatabase::setExtraFlagsForFile(
    PathRef File, std::vector<std::string> ExtraFlags) {
  std::lock_guard<std::mutex> Lock(Mutex);
  ExtraFlagsForFile[File] = std::move(ExtraFlags);
}

void DirectoryBasedGlobalCompilationDatabase::addExtraFlags(
    PathRef File, tooling::CompileCommand &C) const {
  std::lock_guard<std::mutex> Lock(Mutex);

  auto It = ExtraFlagsForFile.find(File);
  if (It == ExtraFlagsForFile.end())
    return;

  auto &Args = C.CommandLine;
  assert(Args.size() >= 2 && "Expected at least [compiler, source file]");
  // The last argument of CommandLine is the name of the input file.
  // Add ExtraFlags before it.
  Args.insert(Args.end() - 1, It->second.begin(), It->second.end());
}

tooling::CompilationDatabase *
DirectoryBasedGlobalCompilationDatabase::tryLoadDatabaseFromPath(
    PathRef File) const {

  namespace path = llvm::sys::path;
  auto CachedIt = CompilationDatabases.find(File);

  assert((path::is_absolute(File, path::Style::posix) ||
          path::is_absolute(File, path::Style::windows)) &&
         "path must be absolute");

  if (CachedIt != CompilationDatabases.end())
    return CachedIt->second.get();
  std::string Error = "";
  auto CDB = tooling::CompilationDatabase::loadFromDirectory(File, Error);
  if (CDB) {
    auto Result = CDB.get();
    CompilationDatabases.insert(std::make_pair(File, std::move(CDB)));
    return Result;
  }

  return nullptr;
}

tooling::CompilationDatabase *
DirectoryBasedGlobalCompilationDatabase::getCompilationDatabase(
    PathRef File) const {
  std::lock_guard<std::mutex> Lock(Mutex);

  namespace path = llvm::sys::path;
  if (CompileCommandsDir.hasValue()) {
    tooling::CompilationDatabase *ReturnValue =
        tryLoadDatabaseFromPath(CompileCommandsDir.getValue());
    if (ReturnValue == nullptr) {
      // FIXME(ibiryukov): pass a proper Context here.
      log(Context::empty(), "Failed to find compilation database for " +
                                Twine(File) + "in overriden directory " +
                                CompileCommandsDir.getValue());
    }
    return ReturnValue;
  }

  for (auto Path = path::parent_path(File); !Path.empty();
       Path = path::parent_path(Path)) {
    auto CDB = tryLoadDatabaseFromPath(Path);
    if (!CDB)
      continue;
    // FIXME(ibiryukov): Invalidate cached compilation databases on changes
    return CDB;
  }

  // FIXME(ibiryukov): pass a proper Context here.
  log(Context::empty(),
      "Failed to find compilation database for " + Twine(File));
  return nullptr;
}

} // namespace clangd
} // namespace clang
