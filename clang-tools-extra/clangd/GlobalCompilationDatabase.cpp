//===--- GlobalCompilationDatabase.cpp ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GlobalCompilationDatabase.h"
#include "Config.h"
#include "FS.h"
#include "SourceCode.h"
#include "support/Logger.h"
#include "support/Path.h"
#include "support/Threading.h"
#include "support/ThreadsafeFS.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/CompilationDatabasePluginRegistry.h"
#include "clang/Tooling/JSONCompilationDatabase.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
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

    if (/*MayCache=*/load(*TFS.view(/*CWD=*/std::nullopt))) {
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
  return tooling::FixedCompilationDatabase::loadFromBuffer(
      llvm::sys::path::parent_path(Path), Data, Error);
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
    std::unique_ptr<tooling::CompilationDatabase> (*Parser)(
        PathRef,
        /*Data*/ llvm::StringRef,
        /*ErrorMsg*/ std::string &);
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
  dlog("No compilation database at {0}", Path);
  return true;
}

DirectoryBasedGlobalCompilationDatabase::
    DirectoryBasedGlobalCompilationDatabase(const Options &Opts)
    : Opts(Opts), Broadcaster(std::make_unique<BroadcastThread>(*this)) {
  if (!this->Opts.ContextProvider)
    this->Opts.ContextProvider = [](llvm::StringRef) {
      return Context::current().clone();
    };
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
    return std::nullopt;
  }

  auto Candidates = Res->CDB->getCompileCommands(File);
  if (!Candidates.empty())
    return std::move(Candidates.front());

  return std::nullopt;
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

  std::string Storage;
  std::vector<llvm::StringRef> SearchDirs;
  if (Opts.CompileCommandsDir) // FIXME: unify this case with config.
    SearchDirs = {*Opts.CompileCommandsDir};
  else {
    WithContext WithProvidedContext(Opts.ContextProvider(Request.FileName));
    const auto &Spec = Config::current().CompileFlags.CDBSearch;
    switch (Spec.Policy) {
    case Config::CDBSearchSpec::NoCDBSearch:
      return std::nullopt;
    case Config::CDBSearchSpec::FixedDir:
      Storage = *Spec.FixedCDBPath;
      SearchDirs = {Storage};
      break;
    case Config::CDBSearchSpec::Ancestors:
      // Traverse the canonical version to prevent false positives. i.e.:
      // src/build/../a.cc can detect a CDB in /src/build if not
      // canonicalized.
      Storage = removeDots(Request.FileName);
      actOnAllParentDirectories(Storage, [&](llvm::StringRef Dir) {
        SearchDirs.push_back(Dir);
        return false;
      });
    }
  }

  std::shared_ptr<const tooling::CompilationDatabase> CDB = nullptr;
  bool ShouldBroadcast = false;
  DirectoryCache *DirCache = nullptr;
  for (DirectoryCache *Candidate : getDirectoryCaches(SearchDirs)) {
    bool CandidateShouldBroadcast = Request.ShouldBroadcast;
    if ((CDB = Candidate->get(Opts.TFS, CandidateShouldBroadcast,
                              Request.FreshTime, Request.FreshTimeMissing))) {
      DirCache = Candidate;
      ShouldBroadcast = CandidateShouldBroadcast;
      break;
    }
  }

  if (!CDB)
    return std::nullopt;

  CDBLookupResult Result;
  Result.CDB = std::move(CDB);
  Result.PI.SourceRoot = DirCache->Path;

  if (ShouldBroadcast)
    broadcastCDB(Result);
  return Result;
}

// The broadcast thread announces files with new compile commands to the world.
// Primarily this is used to enqueue them for background indexing.
//
// It's on a separate thread because:
//  - otherwise it would block the first parse of the initial file
//  - we need to enumerate all files in the CDB, of which there are many
//  - we (will) have to evaluate config for every file in the CDB, which is slow
class DirectoryBasedGlobalCompilationDatabase::BroadcastThread {
  class Filter;
  DirectoryBasedGlobalCompilationDatabase &Parent;

  std::mutex Mu;
  std::condition_variable CV;
  // Shutdown flag (CV is notified after writing).
  // This is atomic so that broadcasts can also observe it and abort early.
  std::atomic<bool> ShouldStop = {false};
  struct Task {
    CDBLookupResult Lookup;
    Context Ctx;
  };
  std::deque<Task> Queue;
  llvm::Optional<Task> ActiveTask;
  std::thread Thread; // Must be last member.

  // Thread body: this is just the basic queue procesing boilerplate.
  void run() {
    std::unique_lock<std::mutex> Lock(Mu);
    while (true) {
      bool Stopping = false;
      CV.wait(Lock, [&] {
        return (Stopping = ShouldStop.load(std::memory_order_acquire)) ||
               !Queue.empty();
      });
      if (Stopping) {
        Queue.clear();
        CV.notify_all();
        return;
      }
      ActiveTask = std::move(Queue.front());
      Queue.pop_front();

      Lock.unlock();
      {
        WithContext WithCtx(std::move(ActiveTask->Ctx));
        process(ActiveTask->Lookup);
      }
      Lock.lock();
      ActiveTask.reset();
      CV.notify_all();
    }
  }

  // Inspects a new CDB and broadcasts the files it owns.
  void process(const CDBLookupResult &T);

public:
  BroadcastThread(DirectoryBasedGlobalCompilationDatabase &Parent)
      : Parent(Parent), Thread([this] { run(); }) {}

  void enqueue(CDBLookupResult Lookup) {
    {
      assert(!Lookup.PI.SourceRoot.empty());
      std::lock_guard<std::mutex> Lock(Mu);
      // New CDB takes precedence over any queued one for the same directory.
      llvm::erase_if(Queue, [&](const Task &T) {
        return T.Lookup.PI.SourceRoot == Lookup.PI.SourceRoot;
      });
      Queue.push_back({std::move(Lookup), Context::current().clone()});
    }
    CV.notify_all();
  }

  bool blockUntilIdle(Deadline Timeout) {
    std::unique_lock<std::mutex> Lock(Mu);
    return wait(Lock, CV, Timeout,
                [&] { return Queue.empty() && !ActiveTask; });
  }

  ~BroadcastThread() {
    {
      std::lock_guard<std::mutex> Lock(Mu);
      ShouldStop.store(true, std::memory_order_release);
    }
    CV.notify_all();
    Thread.join();
  }
};

// The DirBasedCDB associates each file with a specific CDB.
// When a CDB is discovered, it may claim to describe files that we associate
// with a different CDB. We do not want to broadcast discovery of these, and
// trigger background indexing of them.
//
// We must filter the list, and check whether they are associated with this CDB.
// This class attempts to do so efficiently.
//
// Roughly, it:
//  - loads the config for each file, and determines the relevant search path
//  - gathers all directories that are part of any search path
//  - (lazily) checks for a CDB in each such directory at most once
//  - walks the search path for each file and determines whether to include it.
class DirectoryBasedGlobalCompilationDatabase::BroadcastThread::Filter {
  llvm::StringRef ThisDir;
  DirectoryBasedGlobalCompilationDatabase &Parent;

  // Keep track of all directories we might check for CDBs.
  struct DirInfo {
    DirectoryCache *Cache = nullptr;
    enum { Unknown, Missing, TargetCDB, OtherCDB } State = Unknown;
    DirInfo *Parent = nullptr;
  };
  llvm::StringMap<DirInfo> Dirs;

  // A search path starts at a directory, and either includes ancestors or not.
  using SearchPath = llvm::PointerIntPair<DirInfo *, 1>;

  // Add all ancestor directories of FilePath to the tracked set.
  // Returns the immediate parent of the file.
  DirInfo *addParents(llvm::StringRef FilePath) {
    DirInfo *Leaf = nullptr;
    DirInfo *Child = nullptr;
    actOnAllParentDirectories(FilePath, [&](llvm::StringRef Dir) {
      auto &Info = Dirs[Dir];
      // If this is the first iteration, then this node is the overall result.
      if (!Leaf)
        Leaf = &Info;
      // Fill in the parent link from the previous iteration to this parent.
      if (Child)
        Child->Parent = &Info;
      // Keep walking, whether we inserted or not, if parent link is missing.
      // (If it's present, parent links must be present up to the root, so stop)
      Child = &Info;
      return Info.Parent != nullptr;
    });
    return Leaf;
  }

  // Populates DirInfo::Cache (and State, if it is TargetCDB).
  void grabCaches() {
    // Fast path out if there were no files, or CDB loading is off.
    if (Dirs.empty())
      return;

    std::vector<llvm::StringRef> DirKeys;
    std::vector<DirInfo *> DirValues;
    DirKeys.reserve(Dirs.size() + 1);
    DirValues.reserve(Dirs.size());
    for (auto &E : Dirs) {
      DirKeys.push_back(E.first());
      DirValues.push_back(&E.second);
    }

    // Also look up the cache entry for the CDB we're broadcasting.
    // Comparing DirectoryCache pointers is more robust than checking string
    // equality, e.g. reuses the case-sensitivity handling.
    DirKeys.push_back(ThisDir);
    auto DirCaches = Parent.getDirectoryCaches(DirKeys);
    const DirectoryCache *ThisCache = DirCaches.back();
    DirCaches.pop_back();
    DirKeys.pop_back();

    for (unsigned I = 0; I < DirKeys.size(); ++I) {
      DirValues[I]->Cache = DirCaches[I];
      if (DirCaches[I] == ThisCache)
        DirValues[I]->State = DirInfo::TargetCDB;
    }
  }

  // Should we include a file from this search path?
  bool shouldInclude(SearchPath P) {
    DirInfo *Info = P.getPointer();
    if (!Info)
      return false;
    if (Info->State == DirInfo::Unknown) {
      assert(Info->Cache && "grabCaches() should have filled this");
      // Given that we know that CDBs have been moved/generated, don't trust
      // caches. (This should be rare, so it's OK to add a little latency).
      constexpr auto IgnoreCache = std::chrono::steady_clock::time_point::max();
      // Don't broadcast CDBs discovered while broadcasting!
      bool ShouldBroadcast = false;
      bool Exists =
          nullptr != Info->Cache->get(Parent.Opts.TFS, ShouldBroadcast,
                                      /*FreshTime=*/IgnoreCache,
                                      /*FreshTimeMissing=*/IgnoreCache);
      Info->State = Exists ? DirInfo::OtherCDB : DirInfo::Missing;
    }
    // If we have a CDB, include the file if it's the target CDB only.
    if (Info->State != DirInfo::Missing)
      return Info->State == DirInfo::TargetCDB;
    // If we have no CDB and no relevant parent, don't include the file.
    if (!P.getInt() || !Info->Parent)
      return false;
    // Walk up to the next parent.
    return shouldInclude(SearchPath(Info->Parent, 1));
  }

public:
  Filter(llvm::StringRef ThisDir,
         DirectoryBasedGlobalCompilationDatabase &Parent)
      : ThisDir(ThisDir), Parent(Parent) {}

  std::vector<std::string> filter(std::vector<std::string> AllFiles,
                                  std::atomic<bool> &ShouldStop) {
    std::vector<std::string> Filtered;
    // Allow for clean early-exit of the slow parts.
    auto ExitEarly = [&] {
      if (ShouldStop.load(std::memory_order_acquire)) {
        log("Giving up on broadcasting CDB, as we're shutting down");
        Filtered.clear();
        return true;
      }
      return false;
    };
    // Compute search path for each file.
    std::vector<SearchPath> SearchPaths(AllFiles.size());
    for (unsigned I = 0; I < AllFiles.size(); ++I) {
      if (Parent.Opts.CompileCommandsDir) { // FIXME: unify with config
        SearchPaths[I].setPointer(&Dirs[*Parent.Opts.CompileCommandsDir]);
        continue;
      }
      if (ExitEarly()) // loading config may be slow
        return Filtered;
      WithContext WithProvidedContent(Parent.Opts.ContextProvider(AllFiles[I]));
      const Config::CDBSearchSpec &Spec =
          Config::current().CompileFlags.CDBSearch;
      switch (Spec.Policy) {
      case Config::CDBSearchSpec::NoCDBSearch:
        break;
      case Config::CDBSearchSpec::Ancestors:
        SearchPaths[I].setInt(/*Recursive=*/1);
        SearchPaths[I].setPointer(addParents(AllFiles[I]));
        break;
      case Config::CDBSearchSpec::FixedDir:
        SearchPaths[I].setPointer(&Dirs[*Spec.FixedCDBPath]);
        break;
      }
    }
    // Get the CDB cache for each dir on the search path, but don't load yet.
    grabCaches();
    // Now work out which files we want to keep, loading CDBs where needed.
    for (unsigned I = 0; I < AllFiles.size(); ++I) {
      if (ExitEarly()) // loading CDBs may be slow
        return Filtered;
      if (shouldInclude(SearchPaths[I]))
        Filtered.push_back(std::move(AllFiles[I]));
    }
    return Filtered;
  }
};

void DirectoryBasedGlobalCompilationDatabase::BroadcastThread::process(
    const CDBLookupResult &T) {
  vlog("Broadcasting compilation database from {0}", T.PI.SourceRoot);
  std::vector<std::string> GovernedFiles =
      Filter(T.PI.SourceRoot, Parent).filter(T.CDB->getAllFiles(), ShouldStop);
  if (!GovernedFiles.empty())
    Parent.OnCommandChanged.broadcast(std::move(GovernedFiles));
}

void DirectoryBasedGlobalCompilationDatabase::broadcastCDB(
    CDBLookupResult Result) const {
  assert(Result.CDB && "Trying to broadcast an invalid CDB!");
  Broadcaster->enqueue(Result);
}

bool DirectoryBasedGlobalCompilationDatabase::blockUntilIdle(
    Deadline Timeout) const {
  return Broadcaster->blockUntilIdle(Timeout);
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
    return std::nullopt;
  return Res->PI;
}

OverlayCDB::OverlayCDB(const GlobalCompilationDatabase *Base,
                       std::vector<std::string> FallbackFlags,
                       CommandMangler Mangler)
    : DelegatingCDB(Base), Mangler(std::move(Mangler)),
      FallbackFlags(std::move(FallbackFlags)) {}

llvm::Optional<tooling::CompileCommand>
OverlayCDB::getCompileCommand(PathRef File) const {
  llvm::Optional<tooling::CompileCommand> Cmd;
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    auto It = Commands.find(removeDots(File));
    if (It != Commands.end())
      Cmd = It->second;
  }
  if (!Cmd)
    Cmd = DelegatingCDB::getCompileCommand(File);
  if (!Cmd)
    return std::nullopt;
  if (Mangler)
    Mangler(*Cmd, File);
  return Cmd;
}

tooling::CompileCommand OverlayCDB::getFallbackCommand(PathRef File) const {
  auto Cmd = DelegatingCDB::getFallbackCommand(File);
  std::lock_guard<std::mutex> Lock(Mutex);
  Cmd.CommandLine.insert(Cmd.CommandLine.end(), FallbackFlags.begin(),
                         FallbackFlags.end());
  if (Mangler)
    Mangler(Cmd, File);
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

DelegatingCDB::DelegatingCDB(const GlobalCompilationDatabase *Base)
    : Base(Base) {
  if (Base)
    BaseChanged = Base->watch([this](const std::vector<std::string> Changes) {
      OnCommandChanged.broadcast(Changes);
    });
}

DelegatingCDB::DelegatingCDB(std::unique_ptr<GlobalCompilationDatabase> Base)
    : DelegatingCDB(Base.get()) {
  BaseOwner = std::move(Base);
}

llvm::Optional<tooling::CompileCommand>
DelegatingCDB::getCompileCommand(PathRef File) const {
  if (!Base)
    return std::nullopt;
  return Base->getCompileCommand(File);
}

llvm::Optional<ProjectInfo> DelegatingCDB::getProjectInfo(PathRef File) const {
  if (!Base)
    return std::nullopt;
  return Base->getProjectInfo(File);
}

tooling::CompileCommand DelegatingCDB::getFallbackCommand(PathRef File) const {
  if (!Base)
    return GlobalCompilationDatabase::getFallbackCommand(File);
  return Base->getFallbackCommand(File);
}

bool DelegatingCDB::blockUntilIdle(Deadline D) const {
  if (!Base)
    return true;
  return Base->blockUntilIdle(D);
}

} // namespace clangd
} // namespace clang
