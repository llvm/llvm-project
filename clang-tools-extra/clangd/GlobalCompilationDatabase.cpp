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
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include <chrono>
#include <string>
#include <tuple>
#include <vector>

namespace clang {
namespace clangd {
namespace {

// Variant of parent_path that operates only on absolute paths.
PathRef absoluteParent(PathRef Path) {
  assert(llvm::sys::path::is_absolute(Path));
#if defined(_WIN32)
  // llvm::sys says "C:\" is absolute, and its parent is "C:" which is relative.
  // This unhelpful behavior seems to have been inherited from boost.
  if (llvm::sys::path::relative_path(Path)).empty(); {
    return PathRef();
  }
#endif
  PathRef Result = llvm::sys::path::parent_path(Path);
  assert(Result.empty() || llvm::sys::path::is_absolute(Result));
  return Result;
}

// Runs the given action on all parent directories of filename, starting from
// deepest directory and going up to root. Stops whenever action succeeds.
void actOnAllParentDirectories(PathRef FileName,
                               llvm::function_ref<bool(PathRef)> Action) {
  for (auto Path = absoluteParent(FileName); !Path.empty() && !Action(Path);
       Path = absoluteParent(Path))
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

// Loads and caches the CDB from a single directory.
//
// This class is threadsafe, which is to say we have independent locks for each
// directory we're searching for a CDB.
// Loading is deferred until first access.
//
// The DirectoryBasedCDB keeps a map from path => DirectoryCache.
// Typical usage is to:
//  - 1) determine all the paths that might be searched
//  - 2) acquire the map lock and get-or-create all the DirectoryCache entries
//  - 3) release the map lock and query the caches as desired
//
// FIXME: this should revalidate the cache sometimes
// FIXME: IO should go through a VFS
class DirectoryBasedGlobalCompilationDatabase::DirectoryCache {
  // Absolute canonical path that we're the cache for. (Not case-folded).
  const std::string Path;

  // True if we've looked for a CDB here and found none.
  // (This makes it possible for get() to return without taking a lock)
  // FIXME: this should have an expiry time instead of lasting forever.
  std::atomic<bool> FinalizedNoCDB = {false};

  // Guards following cache state.
  std::mutex Mu;
  // Has cache been filled from disk? FIXME: this should be an expiry time.
  bool CachePopulated = false;
  // Whether a new CDB has been loaded but not broadcast yet.
  bool NeedsBroadcast = false;
  // Last loaded CDB, meaningful if CachePopulated is set.
  // shared_ptr so we can overwrite this when callers are still using the CDB.
  std::shared_ptr<tooling::CompilationDatabase> CDB;

public:
  DirectoryCache(llvm::StringRef Path) : Path(Path) {
    assert(llvm::sys::path::is_absolute(Path));
  }

  // Get the CDB associated with this directory.
  // ShouldBroadcast:
  //  - as input, signals whether the caller is willing to broadcast a
  //    newly-discovered CDB. (e.g. to trigger background indexing)
  //  - as output, signals whether the caller should do so.
  // (If a new CDB is discovered and ShouldBroadcast is false, we mark the
  // CDB as needing broadcast, and broadcast it next time we can).
  std::shared_ptr<const tooling::CompilationDatabase>
  get(bool &ShouldBroadcast) {
    // Fast path for common case without taking lock.
    if (FinalizedNoCDB.load()) {
      ShouldBroadcast = false;
      return nullptr;
    }
    std::lock_guard<std::mutex> Lock(Mu);
    auto RequestBroadcast = llvm::make_scope_exit([&, OldCDB(CDB.get())] {
      // If we loaded a new CDB, it should be broadcast at some point.
      if (CDB != nullptr && CDB.get() != OldCDB)
        NeedsBroadcast = true;
      else if (CDB == nullptr) // nothing to broadcast anymore!
        NeedsBroadcast = false;
      // If we have something to broadcast, then do so iff allowed.
      if (!ShouldBroadcast)
        return;
      ShouldBroadcast = NeedsBroadcast;
      NeedsBroadcast = false;
    });

    // For now, we never actually attempt to revalidate a populated cache.
    if (CachePopulated)
      return CDB;
    assert(CDB == nullptr);

    load();
    CachePopulated = true;

    if (!CDB)
      FinalizedNoCDB.store(true);
    return CDB;
  }

  llvm::StringRef path() const { return Path; }

private:
  // Updates `CDB` from disk state.
  void load() {
    std::string Error; // ignored, because it's often "didn't find anything".
    CDB = tooling::CompilationDatabase::loadFromDirectory(Path, Error);
    if (!CDB) {
      // Fallback: check for $src/build, the conventional CMake build root.
      // Probe existence first to avoid each plugin doing IO if it doesn't
      // exist.
      llvm::SmallString<256> BuildDir(Path);
      llvm::sys::path::append(BuildDir, "build");
      if (llvm::sys::fs::is_directory(BuildDir)) {
        vlog("Found candidate build directory {0}", BuildDir);
        CDB = tooling::CompilationDatabase::loadFromDirectory(BuildDir, Error);
      }
    }
    if (CDB) {
      log("Loaded compilation database from {0}", Path);
    } else {
      vlog("No compilation database at {0}", Path);
    }
  }
};

DirectoryBasedGlobalCompilationDatabase::
    DirectoryBasedGlobalCompilationDatabase(
        llvm::Optional<Path> CompileCommandsDir) {
  if (CompileCommandsDir)
    OnlyDirCache = std::make_unique<DirectoryCache>(*CompileCommandsDir);
}

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

std::vector<DirectoryBasedGlobalCompilationDatabase::DirectoryCache *>
DirectoryBasedGlobalCompilationDatabase::getDirectoryCaches(
    llvm::ArrayRef<llvm::StringRef> Dirs) const {
  std::vector<std::string> FoldedDirs;
  FoldedDirs.reserve(Dirs.size());
  for (const auto &Dir : Dirs) {
#ifndef NDEBUG
    if (!llvm::sys::path::is_absolute(Dir))
      elog("Trying to cache CDB for relative {0}");
#endif
    FoldedDirs.push_back(maybeCaseFoldPath(Dir));
  }

  std::vector<DirectoryCache *> Ret;
  Ret.reserve(Dirs.size());

  std::lock_guard<std::mutex> Lock(DirCachesMutex);
  for (unsigned I = 0; I < Dirs.size(); ++I)
    Ret.push_back(&DirCaches.try_emplace(FoldedDirs[I], Dirs[I]).first->second);
  return Ret;
}

llvm::Optional<DirectoryBasedGlobalCompilationDatabase::CDBLookupResult>
DirectoryBasedGlobalCompilationDatabase::lookupCDB(
    CDBLookupRequest Request) const {
  assert(llvm::sys::path::is_absolute(Request.FileName) &&
         "path must be absolute");

  bool ShouldBroadcast = false;
  DirectoryCache *DirCache = nullptr;
  std::shared_ptr<const tooling::CompilationDatabase> CDB = nullptr;
  if (OnlyDirCache) {
    DirCache = OnlyDirCache.get();
    ShouldBroadcast = Request.ShouldBroadcast;
    CDB = DirCache->get(ShouldBroadcast);
  } else {
    // Traverse the canonical version to prevent false positives. i.e.:
    // src/build/../a.cc can detect a CDB in /src/build if not canonicalized.
    std::string CanonicalPath = removeDots(Request.FileName);
    std::vector<llvm::StringRef> SearchDirs;
    actOnAllParentDirectories(CanonicalPath, [&](PathRef Path) {
      SearchDirs.push_back(Path);
      return false;
    });
    for (DirectoryCache *Candidate : getDirectoryCaches(SearchDirs)) {
      bool CandidateShouldBroadcast = Request.ShouldBroadcast;
      if ((CDB = Candidate->get(CandidateShouldBroadcast))) {
        DirCache = Candidate;
        ShouldBroadcast = CandidateShouldBroadcast;
        break;
      }
    }
  }

  if (!CDB)
    return llvm::None;

  CDBLookupResult Result;
  Result.CDB = std::move(CDB);
  Result.PI.SourceRoot = DirCache->path().str();

  // FIXME: Maybe make the following part async, since this can block
  // retrieval of compile commands.
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
  if (OnlyDirCache) {
    assert(OnlyDirCache->path() == Result.PI.SourceRoot &&
           "Trying to broadcast a CDB outside of CompileCommandsDir!");
    OnCommandChanged.broadcast(std::move(AllFiles));
    return;
  }

  // Uniquify all parent directories of all files.
  llvm::StringMap<bool> DirectoryHasCDB;
  std::vector<llvm::StringRef> FileAncestors;
  for (llvm::StringRef File : AllFiles) {
    actOnAllParentDirectories(File, [&](PathRef Path) {
      auto It = DirectoryHasCDB.try_emplace(Path);
      // Already seen this path, and all of its parents.
      if (!It.second)
        return true;

      FileAncestors.push_back(It.first->getKey());
      return pathEqual(Path, Result.PI.SourceRoot);
    });
  }
  // Work out which ones have CDBs in them.
  for (DirectoryCache *Dir : getDirectoryCaches(FileAncestors)) {
    bool ShouldBroadcast = false;
    if (Dir->get(ShouldBroadcast))
      DirectoryHasCDB.find(Dir->path())->setValue(true);
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
