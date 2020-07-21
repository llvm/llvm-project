//===--- GlobalCompilationDatabase.cpp ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GlobalCompilationDatabase.h"
#include "FS.h"
#include "support/Logger.h"
#include "support/Path.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include <string>
#include <tuple>
#include <vector>

namespace clang {
namespace clangd {
namespace {

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

tooling::CompileCommand
GlobalCompilationDatabase::getFallbackCommand(PathRef File) const {
  std::vector<std::string> Argv = {"clang"};
  // Clang treats .h files as C by default and files without extension as linker
  // input, resulting in unhelpful diagnostics.
  // Parsing as Objective C++ is friendly to more cases.
  auto FileExtension = llvm::sys::path::extension(File);
  if (FileExtension.empty() || FileExtension == ".h")
    Argv.push_back("-xobjective-c++-header");
  Argv.push_back(std::string(File));
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

// For platforms where paths are case-insensitive (but case-preserving),
// we need to do case-insensitive comparisons and use lowercase keys.
// FIXME: Make Path a real class with desired semantics instead.
//        This class is not the only place this problem exists.
// FIXME: Mac filesystems default to case-insensitive, but may be sensitive.

static std::string maybeCaseFoldPath(PathRef Path) {
#if defined(_WIN32) || defined(__APPLE__)
  return Path.lower();
#else
  return std::string(Path);
#endif
}

static bool pathEqual(PathRef A, PathRef B) {
#if defined(_WIN32) || defined(__APPLE__)
  return A.equals_lower(B);
#else
  return A == B;
#endif
}

DirectoryBasedGlobalCompilationDatabase::CachedCDB &
DirectoryBasedGlobalCompilationDatabase::getCDBInDirLocked(PathRef Dir) const {
  // FIXME(ibiryukov): Invalidate cached compilation databases on changes
  auto Key = maybeCaseFoldPath(Dir);
  auto R = CompilationDatabases.try_emplace(Key);
  if (R.second) { // Cache miss, try to load CDB.
    CachedCDB &Entry = R.first->second;
    std::string Error;
    Entry.Path = std::string(Dir);
    Entry.CDB = tooling::CompilationDatabase::loadFromDirectory(Dir, Error);
    // Check for $src/build, the conventional CMake build root.
    // Probe existence first to avoid each plugin doing IO if it doesn't exist.
    if (!CompileCommandsDir && !Entry.CDB) {
      llvm::SmallString<256> BuildDir = Dir;
      llvm::sys::path::append(BuildDir, "build");
      if (llvm::sys::fs::is_directory(BuildDir)) {
        vlog("Found candidate build directory {0}", BuildDir);
        Entry.CDB =
            tooling::CompilationDatabase::loadFromDirectory(BuildDir, Error);
      }
    }
    if (Entry.CDB)
      log("Loaded compilation database from {0}", Dir);
  }
  return R.first->second;
}

llvm::Optional<DirectoryBasedGlobalCompilationDatabase::CDBLookupResult>
DirectoryBasedGlobalCompilationDatabase::lookupCDB(
    CDBLookupRequest Request) const {
  assert(llvm::sys::path::is_absolute(Request.FileName) &&
         "path must be absolute");

  bool ShouldBroadcast = false;
  CDBLookupResult Result;

  {
    std::lock_guard<std::mutex> Lock(Mutex);
    CachedCDB *Entry = nullptr;
    if (CompileCommandsDir) {
      Entry = &getCDBInDirLocked(*CompileCommandsDir);
    } else {
      // Traverse the canonical version to prevent false positives. i.e.:
      // src/build/../a.cc can detect a CDB in /src/build if not canonicalized.
      // FIXME(sammccall): this loop is hot, use a union-find-like structure.
      actOnAllParentDirectories(removeDots(Request.FileName),
                                [&](PathRef Path) {
                                  Entry = &getCDBInDirLocked(Path);
                                  return Entry->CDB != nullptr;
                                });
    }

    if (!Entry || !Entry->CDB)
      return llvm::None;

    // Mark CDB as broadcasted to make sure discovery is performed once.
    if (Request.ShouldBroadcast && !Entry->SentBroadcast) {
      Entry->SentBroadcast = true;
      ShouldBroadcast = true;
    }

    Result.CDB = Entry->CDB.get();
    Result.PI.SourceRoot = Entry->Path;
  }

  // FIXME: Maybe make the following part async, since this can block retrieval
  // of compile commands.
  if (ShouldBroadcast)
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

        CachedCDB &Entry = getCDBInDirLocked(Path);
        It.first->second = Entry.CDB != nullptr;
        return pathEqual(Path, Result.PI.SourceRoot);
      });
    }
  }

  std::vector<std::string> GovernedFiles;
  for (llvm::StringRef File : AllFiles) {
    // A file is governed by this CDB if lookup for the file would find it.
    // Independent of whether it has an entry for that file or not.
    actOnAllParentDirectories(File, [&](PathRef Path) {
      if (DirectoryHasCDB.lookup(Path)) {
        if (pathEqual(Path, Result.PI.SourceRoot))
          // Make sure listeners always get a canonical path for the file.
          GovernedFiles.push_back(removeDots(File));
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
                       tooling::ArgumentsAdjuster Adjuster)
    : Base(Base), ArgsAdjuster(std::move(Adjuster)),
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
    auto It = Commands.find(removeDots(File));
    if (It != Commands.end())
      Cmd = It->second;
  }
  if (!Cmd && Base)
    Cmd = Base->getCompileCommand(File);
  if (!Cmd)
    return llvm::None;
  if (ArgsAdjuster)
    Cmd->CommandLine = ArgsAdjuster(Cmd->CommandLine, Cmd->Filename);
  return Cmd;
}

tooling::CompileCommand OverlayCDB::getFallbackCommand(PathRef File) const {
  auto Cmd = Base ? Base->getFallbackCommand(File)
                  : GlobalCompilationDatabase::getFallbackCommand(File);
  std::lock_guard<std::mutex> Lock(Mutex);
  Cmd.CommandLine.insert(Cmd.CommandLine.end(), FallbackFlags.begin(),
                         FallbackFlags.end());
  if (ArgsAdjuster)
    Cmd.CommandLine = ArgsAdjuster(Cmd.CommandLine, Cmd.Filename);
  return Cmd;
}

void OverlayCDB::setCompileCommand(
    PathRef File, llvm::Optional<tooling::CompileCommand> Cmd) {
  // We store a canonical version internally to prevent mismatches between set
  // and get compile commands. Also it assures clients listening to broadcasts
  // doesn't receive different names for the same file.
  std::string CanonPath = removeDots(File);
  {
    std::unique_lock<std::mutex> Lock(Mutex);
    if (Cmd)
      Commands[CanonPath] = std::move(*Cmd);
    else
      Commands.erase(CanonPath);
  }
  OnCommandChanged.broadcast({CanonPath});
}

llvm::Optional<ProjectInfo> OverlayCDB::getProjectInfo(PathRef File) const {
  // It wouldn't make much sense to treat files with overridden commands
  // specially when we can't do the same for the (unknown) local headers they
  // include or changing behavior mid-air after receiving an override.
  if (Base)
    return Base->getProjectInfo(File);
  return llvm::None;
}
} // namespace clangd
} // namespace clang
