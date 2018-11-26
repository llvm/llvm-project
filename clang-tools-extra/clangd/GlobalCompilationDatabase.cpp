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
DirectoryBasedGlobalCompilationDatabase::getCompileCommand(
    PathRef File, ProjectInfo *Project) const {
  if (auto CDB = getCDBForFile(File, Project)) {
    auto Candidates = CDB->getCompileCommands(File);
    if (!Candidates.empty()) {
      return std::move(Candidates.front());
    }
  } else {
    log("Failed to find compilation database for {0}", File);
  }
  return None;
}

std::pair<tooling::CompilationDatabase *, /*Cached*/ bool>
DirectoryBasedGlobalCompilationDatabase::getCDBInDirLocked(PathRef Dir) const {
  // FIXME(ibiryukov): Invalidate cached compilation databases on changes
  auto CachedIt = CompilationDatabases.find(Dir);
  if (CachedIt != CompilationDatabases.end())
    return {CachedIt->second.get(), true};
  std::string Error = "";
  auto CDB = tooling::CompilationDatabase::loadFromDirectory(Dir, Error);
  auto Result = CDB.get();
  CompilationDatabases.insert(std::make_pair(Dir, std::move(CDB)));
  return {Result, false};
}

tooling::CompilationDatabase *
DirectoryBasedGlobalCompilationDatabase::getCDBForFile(
    PathRef File, ProjectInfo *Project) const {
  namespace path = sys::path;
  assert((path::is_absolute(File, path::Style::posix) ||
          path::is_absolute(File, path::Style::windows)) &&
         "path must be absolute");

  tooling::CompilationDatabase *CDB = nullptr;
  bool Cached = false;
  std::lock_guard<std::mutex> Lock(Mutex);
  if (CompileCommandsDir) {
    std::tie(CDB, Cached) = getCDBInDirLocked(*CompileCommandsDir);
    if (Project && CDB)
      Project->SourceRoot = *CompileCommandsDir;
  } else {
    for (auto Path = path::parent_path(File); !CDB && !Path.empty();
         Path = path::parent_path(Path)) {
      std::tie(CDB, Cached) = getCDBInDirLocked(Path);
      if (Project && CDB)
        Project->SourceRoot = Path;
    }
  }
  // FIXME: getAllFiles() may return relative paths, we need absolute paths.
  // Hopefully the fix is to change JSONCompilationDatabase and the interface.
  if (CDB && !Cached)
    OnCommandChanged.broadcast(CDB->getAllFiles());
  return CDB;
}

OverlayCDB::OverlayCDB(const GlobalCompilationDatabase *Base,
                       std::vector<std::string> FallbackFlags)
    : Base(Base), FallbackFlags(std::move(FallbackFlags)) {
  if (Base)
    BaseChanged = Base->watch([this](const std::vector<std::string> Changes) {
      OnCommandChanged.broadcast(Changes);
    });
}

Optional<tooling::CompileCommand>
OverlayCDB::getCompileCommand(PathRef File, ProjectInfo *Project) const {
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    auto It = Commands.find(File);
    if (It != Commands.end()) {
      if (Project)
        Project->SourceRoot = "";
      return It->second;
    }
  }
  return Base ? Base->getCompileCommand(File, Project) : None;
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
  {
    std::unique_lock<std::mutex> Lock(Mutex);
    if (Cmd)
      Commands[File] = std::move(*Cmd);
    else
      Commands.erase(File);
  }
  OnCommandChanged.broadcast({File});
}

} // namespace clangd
} // namespace clang
