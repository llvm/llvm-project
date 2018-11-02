//===--- GlobalCompilationDatabase.cpp ---------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "GlobalCompilationDatabase.h"
#include "Logger.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace llvm;
namespace clang {
namespace clangd {

tooling::CompileCommand
GlobalCompilationDatabase::getFallbackCommand(PathRef File) const {
  std::vector<std::string> Argv = {"clang"};
  // Clang treats .h files as C by default, resulting in unhelpful diagnostics.
  // Parsing as Objective C++ is friendly to more cases.
  if (sys::path::extension(File) == ".h")
    Argv.push_back("-xobjective-c++-header");
  Argv.push_back(File);
  return tooling::CompileCommand(sys::path::parent_path(File),
                                 sys::path::filename(File), std::move(Argv),
                                 /*Output=*/"");
}

DirectoryBasedGlobalCompilationDatabase::
    DirectoryBasedGlobalCompilationDatabase(Optional<Path> CompileCommandsDir)
    : CompileCommandsDir(std::move(CompileCommandsDir)) {}

DirectoryBasedGlobalCompilationDatabase::
    ~DirectoryBasedGlobalCompilationDatabase() = default;

Optional<tooling::CompileCommand>
DirectoryBasedGlobalCompilationDatabase::getCompileCommand(PathRef File) const {
  if (auto CDB = getCDBForFile(File)) {
    auto Candidates = CDB->getCompileCommands(File);
    if (!Candidates.empty())
      return std::move(Candidates.front());
  } else {
    log("Failed to find compilation database for {0}", File);
  }
  return None;
}

tooling::CompilationDatabase *
DirectoryBasedGlobalCompilationDatabase::getCDBInDirLocked(PathRef Dir) const {
  // FIXME(ibiryukov): Invalidate cached compilation databases on changes
  auto CachedIt = CompilationDatabases.find(Dir);
  if (CachedIt != CompilationDatabases.end())
    return CachedIt->second.get();
  std::string Error = "";
  auto CDB = tooling::CompilationDatabase::loadFromDirectory(Dir, Error);
  auto Result = CDB.get();
  CompilationDatabases.insert(std::make_pair(Dir, std::move(CDB)));
  return Result;
}

tooling::CompilationDatabase *
DirectoryBasedGlobalCompilationDatabase::getCDBForFile(PathRef File) const {
  namespace path = sys::path;
  assert((path::is_absolute(File, path::Style::posix) ||
          path::is_absolute(File, path::Style::windows)) &&
         "path must be absolute");

  std::lock_guard<std::mutex> Lock(Mutex);
  if (CompileCommandsDir)
    return getCDBInDirLocked(*CompileCommandsDir);
  for (auto Path = path::parent_path(File); !Path.empty();
       Path = path::parent_path(Path))
    if (auto CDB = getCDBInDirLocked(Path))
      return CDB;
  return nullptr;
}

Optional<tooling::CompileCommand>
OverlayCDB::getCompileCommand(PathRef File) const {
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    auto It = Commands.find(File);
    if (It != Commands.end())
      return It->second;
  }
  return Base ? Base->getCompileCommand(File) : None;
}

tooling::CompileCommand OverlayCDB::getFallbackCommand(PathRef File) const {
  auto Cmd = Base ? Base->getFallbackCommand(File)
                  : GlobalCompilationDatabase::getFallbackCommand(File);
  std::lock_guard<std::mutex> Lock(Mutex);
  Cmd.CommandLine.insert(Cmd.CommandLine.end(), FallbackFlags.begin(),
                         FallbackFlags.end());
  return Cmd;
}

void OverlayCDB::setCompileCommand(
    PathRef File, llvm::Optional<tooling::CompileCommand> Cmd) {
  std::unique_lock<std::mutex> Lock(Mutex);
  if (Cmd)
    Commands[File] = std::move(*Cmd);
  else
    Commands.erase(File);
}

} // namespace clangd
} // namespace clang
