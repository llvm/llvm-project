//===--- GlobalCompilationDatabase.cpp ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GlobalCompilationDatabase.h"
#include "Logger.h"
#include "Path.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <string>
#include <tuple>
#include <vector>

namespace clang {
namespace clangd {
namespace {

void adjustArguments(tooling::CompileCommand &Cmd,
                     llvm::StringRef ResourceDir) {
  tooling::ArgumentsAdjuster ArgsAdjuster = tooling::combineAdjusters(
      // clangd should not write files to disk, including dependency files
      // requested on the command line.
      tooling::getClangStripDependencyFileAdjuster(),
      // Strip plugin related command line arguments. Clangd does
      // not support plugins currently. Therefore it breaks if
      // compiler tries to load plugins.
      tooling::combineAdjusters(tooling::getStripPluginsAdjuster(),
                                tooling::getClangSyntaxOnlyAdjuster()));

  Cmd.CommandLine = ArgsAdjuster(Cmd.CommandLine, Cmd.Filename);
  // Inject the resource dir.
  // FIXME: Don't overwrite it if it's already there.
  if (!ResourceDir.empty())
    Cmd.CommandLine.push_back(("-resource-dir=" + ResourceDir).str());
}

std::string getStandardResourceDir() {
  static int Dummy; // Just an address in this process.
  return CompilerInvocation::GetResourcesPath("clangd", (void *)&Dummy);
}

// Runs the given action on all parent directories of filename, starting from
// deepest directory and going up to root. Stops whenever action succeeds.
void actOnAllParentDirectories(PathRef FileName,
                               llvm::function_ref<bool(PathRef)> Action) {
  for (auto Path = llvm::sys::path::parent_path(FileName);
       !Path.empty() && !Action(Path);
       Path = llvm::sys::path::parent_path(Path))
    ;
}

} // namespace

static std::string getFallbackClangPath() {
  static int Dummy;
  std::string ClangdExecutable =
      llvm::sys::fs::getMainExecutable("clangd", (void *)&Dummy);
  SmallString<128> ClangPath;
  ClangPath = llvm::sys::path::parent_path(ClangdExecutable);
  llvm::sys::path::append(ClangPath, "clang");
  return ClangPath.str();
}

tooling::CompileCommand
GlobalCompilationDatabase::getFallbackCommand(PathRef File) const {
  std::vector<std::string> Argv = {getFallbackClangPath()};
  // Clang treats .h files as C by default and files without extension as linker
  // input, resulting in unhelpful diagnostics.
  // Parsing as Objective C++ is friendly to more cases.
  auto FileExtension = llvm::sys::path::extension(File);
  if (FileExtension.empty() || FileExtension == ".h")
    Argv.push_back("-xobjective-c++-header");
  Argv.push_back(File);
  tooling::CompileCommand Cmd(llvm::sys::path::parent_path(File),
                              llvm::sys::path::filename(File), std::move(Argv),
                              /*Output=*/"");
  Cmd.Heuristic = "clangd fallback";
  return Cmd;
}

DirectoryBasedGlobalCompilationDatabase::
    DirectoryBasedGlobalCompilationDatabase(
        llvm::Optional<Path> CompileCommandsDir)
    : CompileCommandsDir(std::move(CompileCommandsDir)) {}

DirectoryBasedGlobalCompilationDatabase::
    ~DirectoryBasedGlobalCompilationDatabase() = default;

llvm::Optional<tooling::CompileCommand>
DirectoryBasedGlobalCompilationDatabase::getCompileCommand(PathRef File) const {
  CDBLookupRequest Req;
  Req.FileName = File;
  Req.ShouldBroadcast = true;

  auto Res = lookupCDB(Req);
  if (!Res) {
    log("Failed to find compilation database for {0}", File);
    return llvm::None;
  }

  auto Candidates = Res->CDB->getCompileCommands(File);
  if (!Candidates.empty())
    return std::move(Candidates.front());

  return None;
}

std::pair<tooling::CompilationDatabase *, /*SentBroadcast*/ bool>
DirectoryBasedGlobalCompilationDatabase::getCDBInDirLocked(PathRef Dir) const {
  // FIXME(ibiryukov): Invalidate cached compilation databases on changes
  auto CachedIt = CompilationDatabases.find(Dir);
  if (CachedIt != CompilationDatabases.end())
    return {CachedIt->second.CDB.get(), CachedIt->second.SentBroadcast};
  std::string Error = "";

  CachedCDB Entry;
  Entry.CDB = tooling::CompilationDatabase::loadFromDirectory(Dir, Error);
  auto Result = Entry.CDB.get();
  CompilationDatabases[Dir] = std::move(Entry);

  return {Result, false};
}

llvm::Optional<DirectoryBasedGlobalCompilationDatabase::CDBLookupResult>
DirectoryBasedGlobalCompilationDatabase::lookupCDB(
    CDBLookupRequest Request) const {
  assert(llvm::sys::path::is_absolute(Request.FileName) &&
         "path must be absolute");

  CDBLookupResult Result;
  bool SentBroadcast = false;

  {
    std::lock_guard<std::mutex> Lock(Mutex);
    if (CompileCommandsDir) {
      std::tie(Result.CDB, SentBroadcast) =
          getCDBInDirLocked(*CompileCommandsDir);
      Result.PI.SourceRoot = *CompileCommandsDir;
    } else {
      actOnAllParentDirectories(
          Request.FileName, [this, &SentBroadcast, &Result](PathRef Path) {
            std::tie(Result.CDB, SentBroadcast) = getCDBInDirLocked(Path);
            Result.PI.SourceRoot = Path;
            return Result.CDB != nullptr;
          });
    }

    if (!Result.CDB)
      return llvm::None;

    // Mark CDB as broadcasted to make sure discovery is performed once.
    if (Request.ShouldBroadcast && !SentBroadcast)
      CompilationDatabases[Result.PI.SourceRoot].SentBroadcast = true;
  }

  // FIXME: Maybe make the following part async, since this can block retrieval
  // of compile commands.
  if (Request.ShouldBroadcast && !SentBroadcast)
    broadcastCDB(Result);
  return Result;
}

void DirectoryBasedGlobalCompilationDatabase::broadcastCDB(
    CDBLookupResult Result) const {
  assert(Result.CDB && "Trying to broadcast an invalid CDB!");

  std::vector<std::string> AllFiles = Result.CDB->getAllFiles();
  // We assume CDB in CompileCommandsDir owns all of its entries, since we don't
  // perform any search in parent paths whenever it is set.
  if (CompileCommandsDir) {
    assert(*CompileCommandsDir == Result.PI.SourceRoot &&
           "Trying to broadcast a CDB outside of CompileCommandsDir!");
    OnCommandChanged.broadcast(std::move(AllFiles));
    return;
  }

  llvm::StringMap<bool> DirectoryHasCDB;
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    // Uniquify all parent directories of all files.
    for (llvm::StringRef File : AllFiles) {
      actOnAllParentDirectories(File, [&](PathRef Path) {
        auto It = DirectoryHasCDB.try_emplace(Path);
        // Already seen this path, and all of its parents.
        if (!It.second)
          return true;

        auto Res = getCDBInDirLocked(Path);
        It.first->second = Res.first != nullptr;
        return Path == Result.PI.SourceRoot;
      });
    }
  }

  std::vector<std::string> GovernedFiles;
  for (llvm::StringRef File : AllFiles) {
    // A file is governed by this CDB if lookup for the file would find it.
    // Independent of whether it has an entry for that file or not.
    actOnAllParentDirectories(File, [&](PathRef Path) {
      if (DirectoryHasCDB.lookup(Path)) {
        if (Path == Result.PI.SourceRoot)
          GovernedFiles.push_back(File);
        // Stop as soon as we hit a CDB.
        return true;
      }
      return false;
    });
  }

  OnCommandChanged.broadcast(std::move(GovernedFiles));
}

llvm::Optional<ProjectInfo>
DirectoryBasedGlobalCompilationDatabase::getProjectInfo(PathRef File) const {
  CDBLookupRequest Req;
  Req.FileName = File;
  Req.ShouldBroadcast = false;
  auto Res = lookupCDB(Req);
  if (!Res)
    return llvm::None;
  return Res->PI;
}

OverlayCDB::OverlayCDB(const GlobalCompilationDatabase *Base,
                       std::vector<std::string> FallbackFlags,
                       llvm::Optional<std::string> ResourceDir)
    : Base(Base), ResourceDir(ResourceDir ? std::move(*ResourceDir)
                                          : getStandardResourceDir()),
      FallbackFlags(std::move(FallbackFlags)) {
  if (Base)
    BaseChanged = Base->watch([this](const std::vector<std::string> Changes) {
      OnCommandChanged.broadcast(Changes);
    });
}

llvm::Optional<tooling::CompileCommand>
OverlayCDB::getCompileCommand(PathRef File) const {
  llvm::Optional<tooling::CompileCommand> Cmd;
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    auto It = Commands.find(File);
    if (It != Commands.end())
      Cmd = It->second;
  }
  if (!Cmd && Base)
    Cmd = Base->getCompileCommand(File);
  if (!Cmd)
    return llvm::None;
  adjustArguments(*Cmd, ResourceDir);
  return Cmd;
}

tooling::CompileCommand OverlayCDB::getFallbackCommand(PathRef File) const {
  auto Cmd = Base ? Base->getFallbackCommand(File)
                  : GlobalCompilationDatabase::getFallbackCommand(File);
  std::lock_guard<std::mutex> Lock(Mutex);
  Cmd.CommandLine.insert(Cmd.CommandLine.end(), FallbackFlags.begin(),
                         FallbackFlags.end());
  adjustArguments(Cmd, ResourceDir);
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

llvm::Optional<ProjectInfo> OverlayCDB::getProjectInfo(PathRef File) const {
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    auto It = Commands.find(File);
    if (It != Commands.end())
      return ProjectInfo{};
  }
  if (Base)
    return Base->getProjectInfo(File);

  return llvm::None;
}
} // namespace clangd
} // namespace clang
