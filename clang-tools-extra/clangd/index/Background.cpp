//===-- Background.cpp - Build an index in a background thread ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "index/Background.h"
#include "ClangdUnit.h"
#include "Compiler.h"
#include "Logger.h"
#include "SourceCode.h"
#include "Threading.h"
#include "Trace.h"
#include "URI.h"
#include "index/IndexAction.h"
#include "index/MemIndex.h"
#include "index/Serialization.h"
#include "index/SymbolCollector.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA1.h"

#include <chrono>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <thread>

using namespace llvm;
namespace clang {
namespace clangd {
namespace {
// Resolves URI to file paths with cache.
class URIToFileCache {
public:
  URIToFileCache(llvm::StringRef HintPath) : HintPath(HintPath) {}

  llvm::StringRef resolve(llvm::StringRef FileURI) {
    auto I = URIToPathCache.try_emplace(FileURI);
    if (I.second) {
      auto U = URI::parse(FileURI);
      if (!U) {
        elog("Failed to parse URI {0}: {1}", FileURI, U.takeError());
        assert(false && "Failed to parse URI");
        return "";
      }
      auto Path = URI::resolve(*U, HintPath);
      if (!Path) {
        elog("Failed to resolve URI {0}: {1}", FileURI, Path.takeError());
        assert(false && "Failed to resolve URI");
        return "";
      }
      I.first->second = *Path;
    }
    return I.first->second;
  }

private:
  std::string HintPath;
  llvm::StringMap<std::string> URIToPathCache;
};

// We keep only the node "U" and its edges. Any node other than "U" will be
// empty in the resultant graph.
IncludeGraph getSubGraph(const URI &U, const IncludeGraph &FullGraph) {
  IncludeGraph IG;

  std::string FileURI = U.toString();
  auto Entry = IG.try_emplace(FileURI).first;
  auto &Node = Entry->getValue();
  Node = FullGraph.lookup(Entry->getKey());
  Node.URI = Entry->getKey();

  // URIs inside nodes must point into the keys of the same IncludeGraph.
  for (auto &Include : Node.DirectIncludes) {
    auto I = IG.try_emplace(Include).first;
    I->getValue().URI = I->getKey();
    Include = I->getKey();
  }

  return IG;
}

// Creates a filter to not collect index results from files with unchanged
// digests.
// \p FileDigests contains file digests for the current indexed files, and all
// changed files will be added to \p FilesToUpdate.
decltype(SymbolCollector::Options::FileFilter)
createFileFilter(const llvm::StringMap<FileDigest> &FileDigests,
                 llvm::StringMap<FileDigest> &FilesToUpdate) {
  return [&FileDigests, &FilesToUpdate](const SourceManager &SM, FileID FID) {
    StringRef Path;
    if (const auto *F = SM.getFileEntryForID(FID))
      Path = F->getName();
    if (Path.empty())
      return false; // Skip invalid files.
    SmallString<128> AbsPath(Path);
    if (std::error_code EC =
            SM.getFileManager().getVirtualFileSystem()->makeAbsolute(AbsPath)) {
      elog("Warning: could not make absolute file: {0}", EC.message());
      return false; // Skip files without absolute path.
    }
    sys::path::remove_dots(AbsPath, /*remove_dot_dot=*/true);
    auto Digest = digestFile(SM, FID);
    if (!Digest)
      return false;
    auto D = FileDigests.find(AbsPath);
    if (D != FileDigests.end() && D->second == Digest)
      return false; // Skip files that haven't changed.

    FilesToUpdate[AbsPath] = *Digest;
    return true;
  };
}

} // namespace

BackgroundIndex::BackgroundIndex(
    Context BackgroundContext, StringRef ResourceDir,
    const FileSystemProvider &FSProvider, const GlobalCompilationDatabase &CDB,
    BackgroundIndexStorage::Factory IndexStorageFactory,
    size_t BuildIndexPeriodMs, size_t ThreadPoolSize)
    : SwapIndex(make_unique<MemIndex>()), ResourceDir(ResourceDir),
      FSProvider(FSProvider), CDB(CDB),
      BackgroundContext(std::move(BackgroundContext)),
      BuildIndexPeriodMs(BuildIndexPeriodMs),
      SymbolsUpdatedSinceLastIndex(false),
      IndexStorageFactory(std::move(IndexStorageFactory)),
      CommandsChanged(
          CDB.watch([&](const std::vector<std::string> &ChangedFiles) {
            enqueue(ChangedFiles);
          })) {
  assert(ThreadPoolSize > 0 && "Thread pool size can't be zero.");
  assert(this->IndexStorageFactory && "Storage factory can not be null!");
  while (ThreadPoolSize--)
    ThreadPool.emplace_back([this] { run(); });
  if (BuildIndexPeriodMs > 0) {
    log("BackgroundIndex: build symbol index periodically every {0} ms.",
        BuildIndexPeriodMs);
    ThreadPool.emplace_back([this] { buildIndex(); });
  }
}

BackgroundIndex::~BackgroundIndex() {
  stop();
  for (auto &Thread : ThreadPool)
    Thread.join();
}

void BackgroundIndex::stop() {
  {
    std::lock_guard<std::mutex> QueueLock(QueueMu);
    std::lock_guard<std::mutex> IndexLock(IndexMu);
    ShouldStop = true;
  }
  QueueCV.notify_all();
  IndexCV.notify_all();
}

void BackgroundIndex::run() {
  WithContext Background(BackgroundContext.clone());
  while (true) {
    Optional<Task> Task;
    ThreadPriority Priority;
    {
      std::unique_lock<std::mutex> Lock(QueueMu);
      QueueCV.wait(Lock, [&] { return ShouldStop || !Queue.empty(); });
      if (ShouldStop) {
        Queue.clear();
        QueueCV.notify_all();
        return;
      }
      ++NumActiveTasks;
      std::tie(Task, Priority) = std::move(Queue.front());
      Queue.pop_front();
    }

    if (Priority != ThreadPriority::Normal)
      setCurrentThreadPriority(Priority);
    (*Task)();
    if (Priority != ThreadPriority::Normal)
      setCurrentThreadPriority(ThreadPriority::Normal);

    {
      std::unique_lock<std::mutex> Lock(QueueMu);
      assert(NumActiveTasks > 0 && "before decrementing");
      --NumActiveTasks;
    }
    QueueCV.notify_all();
  }
}

bool BackgroundIndex::blockUntilIdleForTest(
    llvm::Optional<double> TimeoutSeconds) {
  std::unique_lock<std::mutex> Lock(QueueMu);
  return wait(Lock, QueueCV, timeoutSeconds(TimeoutSeconds),
              [&] { return Queue.empty() && NumActiveTasks == 0; });
}

void BackgroundIndex::enqueue(const std::vector<std::string> &ChangedFiles) {
  enqueueTask(
      [this, ChangedFiles] {
        trace::Span Tracer("BackgroundIndexEnqueue");
        // We're doing this asynchronously, because we'll read shards here too.
        // FIXME: read shards here too.

        log("Enqueueing {0} commands for indexing", ChangedFiles.size());
        SPAN_ATTACH(Tracer, "files", int64_t(ChangedFiles.size()));

        // We shuffle the files because processing them in a random order should
        // quickly give us good coverage of headers in the project.
        std::vector<unsigned> Permutation(ChangedFiles.size());
        std::iota(Permutation.begin(), Permutation.end(), 0);
        std::mt19937 Generator(std::random_device{}());
        std::shuffle(Permutation.begin(), Permutation.end(), Generator);

        for (const unsigned I : Permutation)
          enqueue(ChangedFiles[I]);
      },
      ThreadPriority::Normal);
}

void BackgroundIndex::enqueue(const std::string &File) {
  ProjectInfo Project;
  if (auto Cmd = CDB.getCompileCommand(File, &Project)) {
    auto *Storage = IndexStorageFactory(Project.SourceRoot);
    // Set priority to low, since background indexing is a long running
    // task we do not want to eat up cpu when there are any other high
    // priority threads.
    enqueueTask(Bind(
                    [this, File, Storage](tooling::CompileCommand Cmd) {
                      Cmd.CommandLine.push_back("-resource-dir=" + ResourceDir);
                      if (auto Error = index(std::move(Cmd), Storage))
                        log("Indexing {0} failed: {1}", File, std::move(Error));
                    },
                    std::move(*Cmd)),
                ThreadPriority::Low);
  }
}

void BackgroundIndex::enqueueTask(Task T, ThreadPriority Priority) {
  {
    std::lock_guard<std::mutex> Lock(QueueMu);
    auto I = Queue.end();
    // We first store the tasks with Normal priority in the front of the queue.
    // Then we store low priority tasks. Normal priority tasks are pretty rare,
    // they should not grow beyond single-digit numbers, so it is OK to do
    // linear search and insert after that.
    if (Priority == ThreadPriority::Normal) {
      I = llvm::find_if(Queue, [](const std::pair<Task, ThreadPriority> &Elem) {
        return Elem.second == ThreadPriority::Low;
      });
    }
    Queue.insert(I, {std::move(T), Priority});
  }
  QueueCV.notify_all();
}

/// Given index results from a TU, only update files in \p FilesToUpdate.
void BackgroundIndex::update(StringRef MainFile, IndexFileIn Index,
                             const StringMap<FileDigest> &FilesToUpdate,
                             BackgroundIndexStorage *IndexStorage) {
  // Partition symbols/references into files.
  struct File {
    DenseSet<const Symbol *> Symbols;
    DenseSet<const Ref *> Refs;
  };
  StringMap<File> Files;
  URIToFileCache URICache(MainFile);
  for (const auto &Sym : *Index.Symbols) {
    if (Sym.CanonicalDeclaration) {
      auto DeclPath = URICache.resolve(Sym.CanonicalDeclaration.FileURI);
      if (FilesToUpdate.count(DeclPath) != 0)
        Files[DeclPath].Symbols.insert(&Sym);
    }
    // For symbols with different declaration and definition locations, we store
    // the full symbol in both the header file and the implementation file, so
    // that merging can tell the preferred symbols (from canonical headers) from
    // other symbols (e.g. forward declarations).
    if (Sym.Definition &&
        Sym.Definition.FileURI != Sym.CanonicalDeclaration.FileURI) {
      auto DefPath = URICache.resolve(Sym.Definition.FileURI);
      if (FilesToUpdate.count(DefPath) != 0)
        Files[DefPath].Symbols.insert(&Sym);
    }
  }
  DenseMap<const Ref *, SymbolID> RefToIDs;
  for (const auto &SymRefs : *Index.Refs) {
    for (const auto &R : SymRefs.second) {
      auto Path = URICache.resolve(R.Location.FileURI);
      if (FilesToUpdate.count(Path) != 0) {
        auto &F = Files[Path];
        RefToIDs[&R] = SymRefs.first;
        F.Refs.insert(&R);
      }
    }
  }

  // Build and store new slabs for each updated file.
  for (const auto &F : Files) {
    StringRef Path = F.first();
    vlog("Update symbols in {0}", Path);
    SymbolSlab::Builder Syms;
    RefSlab::Builder Refs;
    for (const auto *S : F.second.Symbols)
      Syms.insert(*S);
    for (const auto *R : F.second.Refs)
      Refs.insert(RefToIDs[R], *R);

    auto SS = llvm::make_unique<SymbolSlab>(std::move(Syms).build());
    auto RS = llvm::make_unique<RefSlab>(std::move(Refs).build());
    auto IG = llvm::make_unique<IncludeGraph>(
        getSubGraph(URI::create(Path), Index.Sources.getValue()));

    auto Hash = FilesToUpdate.lookup(Path);
    // We need to store shards before updating the index, since the latter
    // consumes slabs.
    if (IndexStorage) {
      IndexFileOut Shard;
      Shard.Symbols = SS.get();
      Shard.Refs = RS.get();
      Shard.Sources = IG.get();

      if (auto Error = IndexStorage->storeShard(Path, Shard))
        elog("Failed to write background-index shard for file {0}: {1}", Path,
             std::move(Error));
    }

    std::lock_guard<std::mutex> Lock(DigestsMu);
    // This can override a newer version that is added in another thread,
    // if this thread sees the older version but finishes later. This should be
    // rare in practice.
    IndexedFileDigests[Path] = Hash;
    IndexedSymbols.update(Path, std::move(SS), std::move(RS));
  }
}

void BackgroundIndex::buildIndex() {
  assert(BuildIndexPeriodMs > 0);
  while (true) {
    {
      std::unique_lock<std::mutex> Lock(IndexMu);
      if (ShouldStop)  // Avoid waiting if stopped.
        break;
      // Wait until this is notified to stop or `BuildIndexPeriodMs` has past.
      IndexCV.wait_for(Lock, std::chrono::milliseconds(BuildIndexPeriodMs));
      if (ShouldStop)  // Avoid rebuilding index if stopped.
        break;
    }
    if (!SymbolsUpdatedSinceLastIndex.exchange(false))
      continue;
    // There can be symbol update right after the flag is reset above and before
    // index is rebuilt below. The new index would contain the updated symbols
    // but the flag would still be true. This is fine as we would simply run an
    // extra index build.
    reset(
        IndexedSymbols.buildIndex(IndexType::Heavy, DuplicateHandling::Merge));
    log("BackgroundIndex: rebuilt symbol index.");
  }
}

Error BackgroundIndex::index(tooling::CompileCommand Cmd,
                             BackgroundIndexStorage *IndexStorage) {
  trace::Span Tracer("BackgroundIndex");
  SPAN_ATTACH(Tracer, "file", Cmd.Filename);
  SmallString<128> AbsolutePath;
  if (sys::path::is_absolute(Cmd.Filename)) {
    AbsolutePath = Cmd.Filename;
  } else {
    AbsolutePath = Cmd.Directory;
    sys::path::append(AbsolutePath, Cmd.Filename);
  }

  auto FS = FSProvider.getFileSystem();
  auto Buf = FS->getBufferForFile(AbsolutePath);
  if (!Buf)
    return errorCodeToError(Buf.getError());
  auto Hash = digest(Buf->get()->getBuffer());

  // Take a snapshot of the digests to avoid locking for each file in the TU.
  llvm::StringMap<FileDigest> DigestsSnapshot;
  {
    std::lock_guard<std::mutex> Lock(DigestsMu);
    if (IndexedFileDigests.lookup(AbsolutePath) == Hash) {
      vlog("No need to index {0}, already up to date", AbsolutePath);
      return Error::success();
    }

    DigestsSnapshot = IndexedFileDigests;
  }

  log("Indexing {0} (digest:={1})", Cmd.Filename, toHex(Hash));
  ParseInputs Inputs;
  Inputs.FS = std::move(FS);
  Inputs.FS->setCurrentWorkingDirectory(Cmd.Directory);
  Inputs.CompileCommand = std::move(Cmd);
  auto CI = buildCompilerInvocation(Inputs);
  if (!CI)
    return createStringError(inconvertibleErrorCode(),
                             "Couldn't build compiler invocation");
  IgnoreDiagnostics IgnoreDiags;
  auto Clang = prepareCompilerInstance(
      std::move(CI), /*Preamble=*/nullptr, std::move(*Buf),
      std::make_shared<PCHContainerOperations>(), Inputs.FS, IgnoreDiags);
  if (!Clang)
    return createStringError(inconvertibleErrorCode(),
                             "Couldn't build compiler instance");

  SymbolCollector::Options IndexOpts;
  StringMap<FileDigest> FilesToUpdate;
  IndexOpts.FileFilter = createFileFilter(DigestsSnapshot, FilesToUpdate);
  IndexFileIn Index;
  auto Action = createStaticIndexingAction(
      IndexOpts, [&](SymbolSlab S) { Index.Symbols = std::move(S); },
      [&](RefSlab R) { Index.Refs = std::move(R); },
      [&](IncludeGraph IG) { Index.Sources = std::move(IG); });

  // We're going to run clang here, and it could potentially crash.
  // We could use CrashRecoveryContext to try to make indexing crashes nonfatal,
  // but the leaky "recovery" is pretty scary too in a long-running process.
  // If crashes are a real problem, maybe we should fork a child process.

  const FrontendInputFile &Input = Clang->getFrontendOpts().Inputs.front();
  if (!Action->BeginSourceFile(*Clang, Input))
    return createStringError(inconvertibleErrorCode(),
                             "BeginSourceFile() failed");
  if (!Action->Execute())
    return createStringError(inconvertibleErrorCode(), "Execute() failed");
  Action->EndSourceFile();
  if (Clang->hasDiagnostics() &&
      Clang->getDiagnostics().hasUncompilableErrorOccurred()) {
    return createStringError(inconvertibleErrorCode(),
                             "IndexingAction failed: has uncompilable errors");
  }

  assert(Index.Symbols && Index.Refs && Index.Sources
     && "Symbols, Refs and Sources must be set.");

  log("Indexed {0} ({1} symbols, {2} refs, {3} files)",
      Inputs.CompileCommand.Filename, Index.Symbols->size(),
      Index.Refs->numRefs(), Index.Sources->size());
  SPAN_ATTACH(Tracer, "symbols", int(Index.Symbols->size()));
  SPAN_ATTACH(Tracer, "refs", int(Index.Refs->numRefs()));
  SPAN_ATTACH(Tracer, "sources", int(Index.Sources->size()));

  update(AbsolutePath, std::move(Index), FilesToUpdate, IndexStorage);
  {
    // Make sure hash for the main file is always updated even if there is no
    // index data in it.
    std::lock_guard<std::mutex> Lock(DigestsMu);
    IndexedFileDigests[AbsolutePath] = Hash;
  }

  if (BuildIndexPeriodMs > 0)
    SymbolsUpdatedSinceLastIndex = true;
  else
    reset(
        IndexedSymbols.buildIndex(IndexType::Light, DuplicateHandling::Merge));

  return Error::success();
}

} // namespace clangd
} // namespace clang
