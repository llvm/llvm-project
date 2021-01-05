//===--- GlobalCompilationDatabase.cpp ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GlobalCompilationDatabase.h"
#include "FS.h"
#include "SourceCode.h"
#include "support/Logger.h"
#include "support/Path.h"
#include "support/ThreadsafeFS.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/CompilationDatabasePluginRegistry.h"
#include "clang/Tooling/JSONCompilationDatabase.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/VirtualFileSystem.h"
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
  if (llvm::sys::path::relative_path(Path).empty()) {
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
class DirectoryBasedGlobalCompilationDatabase::DirectoryCache {
  using stopwatch = std::chrono::steady_clock;

  // CachedFile is used to read a CDB file on disk (e.g. compile_commands.json).
  // It specializes in being able to quickly bail out if the file is unchanged,
  // which is the common case.
  // Internally, it stores file metadata so a stat() can verify it's unchanged.
  // We don't actually cache the content as it's not needed - if the file is
  // unchanged then the previous CDB is valid.
  struct CachedFile {
    CachedFile(llvm::StringRef Parent, llvm::StringRef Rel) {
      llvm::SmallString<256> Path = Parent;
      llvm::sys::path::append(Path, Rel);
      this->Path = Path.str().str();
    }
    std::string Path;
    size_t Size = NoFileCached;
    llvm::sys::TimePoint<> ModifiedTime;
    FileDigest ContentHash;

    static constexpr size_t NoFileCached = -1;

    struct LoadResult {
      enum {
        FileNotFound,
        TransientError,
        FoundSameData,
        FoundNewData,
      } Result;
      std::unique_ptr<llvm::MemoryBuffer> Buffer; // Set only if FoundNewData
    };

    LoadResult load(llvm::vfs::FileSystem &FS, bool HasOldData);
  };

  // If we've looked for a CDB here and found none, the time when that happened.
  // (Atomics make it possible for get() to return without taking a lock)
  std::atomic<stopwatch::rep> NoCDBAt = {
      stopwatch::time_point::min().time_since_epoch().count()};

  // Guards the following cache state.
  std::mutex Mu;
  // When was the cache last known to be in sync with disk state?
  stopwatch::time_point CachePopulatedAt = stopwatch::time_point::min();
  // Whether a new CDB has been loaded but not broadcast yet.
  bool NeedsBroadcast = false;
  // Last loaded CDB, meaningful if CachePopulatedAt was ever set.
  // shared_ptr so we can overwrite this when callers are still using the CDB.
  std::shared_ptr<tooling::CompilationDatabase> CDB;
  // File metadata for the CDB files we support tracking directly.
  CachedFile CompileCommandsJson;
  CachedFile BuildCompileCommandsJson;
  CachedFile CompileFlagsTxt;
  // CachedFile member corresponding to CDB.
  //   CDB  | ACF  | Scenario
  //   null | null | no CDB found, or initial empty cache
  //   set  | null | CDB was loaded via generic plugin interface
  //   null | set  | found known CDB file, but parsing it failed
  //   set  | set  | CDB was parsed from a known file
  CachedFile *ActiveCachedFile = nullptr;

public:
  DirectoryCache(llvm::StringRef Path)
      : CompileCommandsJson(Path, "compile_commands.json"),
        BuildCompileCommandsJson(Path, "build/compile_commands.json"),
        CompileFlagsTxt(Path, "compile_flags.txt"), Path(Path) {
    assert(llvm::sys::path::is_absolute(Path));
  }

  // Absolute canonical path that we're the cache for. (Not case-folded).
  const std::string Path;

  // Get the CDB associated with this directory.
  // ShouldBroadcast:
  //  - as input, signals whether the caller is willing to broadcast a
  //    newly-discovered CDB. (e.g. to trigger background indexing)
  //  - as output, signals whether the caller should do so.
  // (If a new CDB is discovered and ShouldBroadcast is false, we mark the
  // CDB as needing broadcast, and broadcast it next time we can).
  std::shared_ptr<const tooling::CompilationDatabase>
  get(const ThreadsafeFS &TFS, bool &ShouldBroadcast,
      stopwatch::time_point FreshTime, stopwatch::time_point FreshTimeMissing) {
    // Fast path for common case without taking lock.
    if (stopwatch::time_point(stopwatch::duration(NoCDBAt.load())) >
        FreshTimeMissing) {
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

    // If our cache is valid, serve from it.
    if (CachePopulatedAt > FreshTime)
      return CDB;

    if (/*MayCache=*/load(*TFS.view(/*CWD=*/llvm::None))) {
      // Use new timestamp, as loading may be slow.
      CachePopulatedAt = stopwatch::now();
      NoCDBAt.store((CDB ? stopwatch::time_point::min() : CachePopulatedAt)
                        .time_since_epoch()
                        .count());
    }

    return CDB;
  }

private:
  // Updates `CDB` from disk state. Returns false on failure.
  bool load(llvm::vfs::FileSystem &FS);
};

DirectoryBasedGlobalCompilationDatabase::DirectoryCache::CachedFile::LoadResult
DirectoryBasedGlobalCompilationDatabase::DirectoryCache::CachedFile::load(
    llvm::vfs::FileSystem &FS, bool HasOldData) {
  auto Stat = FS.status(Path);
  if (!Stat || !Stat->isRegularFile()) {
    Size = NoFileCached;
    ContentHash = {};
    return {LoadResult::FileNotFound, nullptr};
  }
  // If both the size and mtime match, presume unchanged without reading.
  if (HasOldData && Stat->getLastModificationTime() == ModifiedTime &&
      Stat->getSize() == Size)
    return {LoadResult::FoundSameData, nullptr};
  auto Buf = FS.getBufferForFile(Path);
  if (!Buf || (*Buf)->getBufferSize() != Stat->getSize()) {
    // Don't clear the cache - possible we're seeing inconsistent size as the
    // file is being recreated. If it ends up identical later, great!
    //
    // This isn't a complete solution: if we see a partial file but stat/read
    // agree on its size, we're ultimately going to have spurious CDB reloads.
    // May be worth fixing if generators don't write atomically (CMake does).
    elog("Failed to read {0}: {1}", Path,
         Buf ? "size changed" : Buf.getError().message());
    return {LoadResult::TransientError, nullptr};
  }

  FileDigest NewContentHash = digest((*Buf)->getBuffer());
  if (HasOldData && NewContentHash == ContentHash) {
    // mtime changed but data is the same: avoid rebuilding the CDB.
    ModifiedTime = Stat->getLastModificationTime();
    return {LoadResult::FoundSameData, nullptr};
  }

  Size = (*Buf)->getBufferSize();
  ModifiedTime = Stat->getLastModificationTime();
  ContentHash = NewContentHash;
  return {LoadResult::FoundNewData, std::move(*Buf)};
}

// Adapt CDB-loading functions to a common interface for DirectoryCache::load().
static std::unique_ptr<tooling::CompilationDatabase>
parseJSON(PathRef Path, llvm::StringRef Data, std::string &Error) {
  if (auto CDB = tooling::JSONCompilationDatabase::loadFromBuffer(
          Data, Error, tooling::JSONCommandLineSyntax::AutoDetect)) {
    // FS used for expanding response files.
    // FIXME: ExpandResponseFilesDatabase appears not to provide the usual
    // thread-safety guarantees, as the access to FS is not locked!
    // For now, use the real FS, which is known to be threadsafe (if we don't
    // use/change working directory, which ExpandResponseFilesDatabase doesn't).
    auto FS = llvm::vfs::getRealFileSystem();
    return tooling::inferTargetAndDriverMode(
        tooling::inferMissingCompileCommands(
            expandResponseFiles(std::move(CDB), std::move(FS))));
  }
  return nullptr;
}
static std::unique_ptr<tooling::CompilationDatabase>
parseFixed(PathRef Path, llvm::StringRef Data, std::string &Error) {
  return tooling::FixedCompilationDatabase::loadFromBuffer(Path, Data, Error);
}

bool DirectoryBasedGlobalCompilationDatabase::DirectoryCache::load(
    llvm::vfs::FileSystem &FS) {
  dlog("Probing directory {0}", Path);
  std::string Error;

  // Load from the specially-supported compilation databases (JSON + Fixed).
  // For these, we know the files they read and cache their metadata so we can
  // cheaply validate whether they've changed, and hot-reload if they have.
  // (As a bonus, these are also VFS-clean)!
  struct CDBFile {
    CachedFile *File;
    // Wrapper for {Fixed,JSON}CompilationDatabase::loadFromBuffer.
    llvm::function_ref<std::unique_ptr<tooling::CompilationDatabase>(
        PathRef,
        /*Data*/ llvm::StringRef,
        /*ErrorMsg*/ std::string &)>
        Parser;
  };
  for (const auto &Entry : {CDBFile{&CompileCommandsJson, parseJSON},
                            CDBFile{&BuildCompileCommandsJson, parseJSON},
                            CDBFile{&CompileFlagsTxt, parseFixed}}) {
    bool Active = ActiveCachedFile == Entry.File;
    auto Loaded = Entry.File->load(FS, Active);
    switch (Loaded.Result) {
    case CachedFile::LoadResult::FileNotFound:
      if (Active) {
        log("Unloaded compilation database from {0}", Entry.File->Path);
        ActiveCachedFile = nullptr;
        CDB = nullptr;
      }
      // Continue looking at other candidates.
      break;
    case CachedFile::LoadResult::TransientError:
      // File existed but we couldn't read it. Reuse the cache, retry later.
      return false; // Load again next time.
    case CachedFile::LoadResult::FoundSameData:
      assert(Active && "CachedFile may not return 'same data' if !HasOldData");
      // This is the critical file, and it hasn't changed.
      return true;
    case CachedFile::LoadResult::FoundNewData:
      // We have a new CDB!
      CDB = Entry.Parser(Entry.File->Path, Loaded.Buffer->getBuffer(), Error);
      if (CDB)
        log("{0} compilation database from {1}", Active ? "Reloaded" : "Loaded",
            Entry.File->Path);
      else
        elog("Failed to load compilation database from {0}: {1}",
             Entry.File->Path, Error);
      ActiveCachedFile = Entry.File;
      return true;
    }
  }

  // Fall back to generic handling of compilation databases.
  // We don't know what files they read, so can't efficiently check whether
  // they need to be reloaded. So we never do that.
  // FIXME: the interface doesn't provide a way to virtualize FS access.

  // Don't try these more than once. If we've scanned before, we're done.
  if (CachePopulatedAt > stopwatch::time_point::min())
    return true;
  for (const auto &Entry :
       tooling::CompilationDatabasePluginRegistry::entries()) {
    // Avoid duplicating the special cases handled above.
    if (Entry.getName() == "fixed-compilation-database" ||
        Entry.getName() == "json-compilation-database")
      continue;
    auto Plugin = Entry.instantiate();
    if (auto CDB = Plugin->loadFromDirectory(Path, Error)) {
      log("Loaded compilation database from {0} with plugin {1}", Path,
          Entry.getName());
      this->CDB = std::move(CDB);
      return true;
    }
    // Don't log Error here, it's usually just "couldn't find <file>".
  }
  vlog("No compilation database at {0}", Path);
  return true;
}

DirectoryBasedGlobalCompilationDatabase::
    DirectoryBasedGlobalCompilationDatabase(const Options &Opts)
    : Opts(Opts) {
  if (Opts.CompileCommandsDir)
    OnlyDirCache = std::make_unique<DirectoryCache>(*Opts.CompileCommandsDir);
}

DirectoryBasedGlobalCompilationDatabase::
    ~DirectoryBasedGlobalCompilationDatabase() = default;

llvm::Optional<tooling::CompileCommand>
DirectoryBasedGlobalCompilationDatabase::getCompileCommand(PathRef File) const {
  CDBLookupRequest Req;
  Req.FileName = File;
  Req.ShouldBroadcast = true;
  auto Now = std::chrono::steady_clock::now();
  Req.FreshTime = Now - Opts.RevalidateAfter;
  Req.FreshTimeMissing = Now - Opts.RevalidateMissingAfter;

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
    CDB = DirCache->get(Opts.TFS, ShouldBroadcast, Request.FreshTime,
                        Request.FreshTimeMissing);
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
      if ((CDB = Candidate->get(Opts.TFS, CandidateShouldBroadcast,
                                Request.FreshTime, Request.FreshTimeMissing))) {
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
  Result.PI.SourceRoot = DirCache->Path;

  // FIXME: Maybe make the following part async, since this can block
  // retrieval of compile commands.
  if (ShouldBroadcast)
    broadcastCDB(Result);
  return Result;
}

void DirectoryBasedGlobalCompilationDatabase::broadcastCDB(
    CDBLookupResult Result) const {
  vlog("Broadcasting compilation database from {0}", Result.PI.SourceRoot);
  assert(Result.CDB && "Trying to broadcast an invalid CDB!");

  std::vector<std::string> AllFiles = Result.CDB->getAllFiles();
  // We assume CDB in CompileCommandsDir owns all of its entries, since we don't
  // perform any search in parent paths whenever it is set.
  if (OnlyDirCache) {
    assert(OnlyDirCache->Path == Result.PI.SourceRoot &&
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
  // Given that we know that CDBs have been moved/generated, don't trust caches.
  // (This should be rare, so it's OK to add a little latency).
  constexpr auto IgnoreCache = std::chrono::steady_clock::time_point::max();
  auto DirectoryCaches = getDirectoryCaches(FileAncestors);
  assert(DirectoryCaches.size() == FileAncestors.size());
  for (unsigned I = 0; I < DirectoryCaches.size(); ++I) {
    bool ShouldBroadcast = false;
    if (DirectoryCaches[I]->get(Opts.TFS, ShouldBroadcast,
                                /*FreshTime=*/IgnoreCache,
                                /*FreshTimeMissing=*/IgnoreCache))
      DirectoryHasCDB.find(FileAncestors[I])->setValue(true);
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
  Req.FreshTime = Req.FreshTimeMissing =
      std::chrono::steady_clock::time_point::min();
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
