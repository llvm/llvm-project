//===-- Background.cpp - Build an index in a background thread ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "index/Background.h"
#include "Compiler.h"
#include "Config.h"
#include "FS.h"
#include "Headers.h"
#include "SourceCode.h"
#include "URI.h"
#include "index/BackgroundIndexLoader.h"
#include "index/FileIndex.h"
#include "index/Index.h"
#include "index/IndexAction.h"
#include "index/MemIndex.h"
#include "index/Ref.h"
#include "index/Relation.h"
#include "index/Serialization.h"
#include "index/Symbol.h"
#include "index/SymbolCollector.h"
#include "support/Context.h"
#include "support/Logger.h"
#include "support/Path.h"
#include "support/Threading.h"
#include "support/ThreadsafeFS.h"
#include "support/Trace.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Stack.h"
#include "clang/Frontend/FrontendAction.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/xxhash.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace clang {
namespace clangd {
namespace {

// We cannot use vfs->makeAbsolute because Cmd.FileName is either absolute or
// relative to Cmd.Directory, which might not be the same as current working
// directory.
llvm::SmallString<128> getAbsolutePath(const tooling::CompileCommand &Cmd) {
  llvm::SmallString<128> AbsolutePath;
  if (llvm::sys::path::is_absolute(Cmd.Filename)) {
    AbsolutePath = Cmd.Filename;
  } else {
    AbsolutePath = Cmd.Directory;
    llvm::sys::path::append(AbsolutePath, Cmd.Filename);
    llvm::sys::path::remove_dots(AbsolutePath, true);
  }
  return AbsolutePath;
}

bool shardIsStale(const LoadedShard &LS, llvm::vfs::FileSystem *FS) {
  auto Buf = FS->getBufferForFile(LS.AbsolutePath);
  if (!Buf) {
    vlog("Background-index: Couldn't read {0} to validate stored index: {1}",
         LS.AbsolutePath, Buf.getError().message());
    // There is no point in indexing an unreadable file.
    return false;
  }
  return digest(Buf->get()->getBuffer()) != LS.Digest;
}

llvm::Expected<BackgroundIndex::IndexedFile>
indexedFile(PathRef File, const IncludeGraphNode &Source, PathRef HintPath) {
  BackgroundIndex::IndexedFile Result;
  Result.File = File.str();
  Result.Digest = Source.Digest;
  Result.Flags = Source.Flags;
  for (llvm::StringRef IncludedURI : Source.DirectIncludes) {
    auto Included = URI::resolve(IncludedURI, HintPath);
    if (!Included)
      return error("cannot resolve include URI {0}: {1}", IncludedURI,
                   Included.takeError());
    Result.DirectIncludes.push_back(std::move(*Included));
  }
  return Result;
}

} // namespace

BackgroundIndex::BackgroundIndex(
    const ThreadsafeFS &TFS, const GlobalCompilationDatabase &CDB,
    BackgroundIndexStorage::Factory IndexStorageFactory, Options Opts)
    : SwapIndex(std::make_unique<MemIndex>()), TFS(TFS), CDB(CDB),
      IndexingPriority(Opts.IndexingPriority),
      ContextProvider(std::move(Opts.ContextProvider)),
      IndexedSymbols(IndexContents::All, Opts.SupportContainedRefs),
      Rebuilder(this, &IndexedSymbols, Opts.ThreadPoolSize),
      IndexStorageFactory(std::move(IndexStorageFactory)),
      Queue(std::move(Opts.OnProgress)),
      CommandsChanged(
          CDB.watch([&](const std::vector<std::string> &ChangedFiles) {
            enqueue(ChangedFiles);
          })) {
  assert(Opts.ThreadPoolSize > 0 && "Thread pool size can't be zero.");
  assert(this->IndexStorageFactory && "Storage factory can not be null!");
  for (unsigned I = 0; I < Opts.ThreadPoolSize; ++I) {
    ThreadPool.runAsync("background-worker-" + llvm::Twine(I + 1),
                        [this, Ctx(Context::current().clone())]() mutable {
                          clang::noteBottomOfStack();
                          WithContext BGContext(std::move(Ctx));
                          Queue.work([&] { Rebuilder.idle(); });
                        });
  }
}

BackgroundIndex::~BackgroundIndex() {
  stop();
  ThreadPool.wait();
}

void BackgroundIndex::enqueue(const std::vector<std::string> &ChangedFiles) {
  {
    std::lock_guard<std::mutex> Lock(ShardVersionsMu);
    for (PathRef File : ChangedFiles)
      KnownTUs.insert(removeDots(File));
  }
  Queue.push(changedFilesTask(ChangedFiles));
}

llvm::Expected<std::vector<BackgroundIndex::IndexedFile>>
BackgroundIndex::includeGraphSnapshot() const {
  std::lock_guard<std::mutex> Lock(ShardVersionsMu);
  if (IncludeGraphError)
    return error("background include graph is incomplete: {0}",
                 *IncludeGraphError);
  if (!IndexFailures.empty())
    return error("background indexing failed for {0}: {1}",
                 IndexFailures.begin()->first(),
                 IndexFailures.begin()->getValue());
  for (llvm::StringRef TU : KnownTUs.keys()) {
    auto It = IndexedFiles.find(TU);
    if (It == IndexedFiles.end() ||
        !(It->getValue().Flags & IncludeGraphNode::SourceFlag::IsTU))
      return error("background include graph has no translation unit {0}", TU);
  }
  std::vector<IndexedFile> Result;
  Result.reserve(IndexedFiles.size());
  for (const auto &Entry : IndexedFiles)
    Result.push_back(Entry.getValue());
  return Result;
}

llvm::Error BackgroundIndex::invalidateAfterFileRenames(
    llvm::ArrayRef<std::pair<Path, Path>> Renames) {
  for (const auto &Rename : Renames)
    if (!llvm::sys::path::is_absolute(Rename.first) ||
        !llvm::sys::path::is_absolute(Rename.second))
      return error("file rename paths must be absolute: {0} -> {1}",
                   Rename.first, Rename.second);
  auto AffectedByRename = [&](PathRef File) {
    return llvm::any_of(Renames, [&](const auto &Rename) {
      return pathEqual(Rename.first, File) ||
             pathStartsWith(Rename.first, File) ||
             pathEqual(Rename.second, File) ||
             pathStartsWith(Rename.second, File);
    });
  };

  llvm::StringSet<> NewKnownTUs;
  llvm::StringSet<> InvalidatedFiles;
  std::vector<Path> RemovedPaths;
  llvm::StringSet<> TranslationUnits;
  {
    std::lock_guard<std::mutex> Lock(ShardVersionsMu);

    // Validate every path projection before changing any state.
    auto ValidatePath = [&](PathRef File) -> llvm::Error {
      auto Projected = mapPathAfterRenames(File, Renames);
      if (!Projected)
        return Projected.takeError();
      return llvm::Error::success();
    };
    for (const auto &Entry : IndexedFiles) {
      if (auto Err = ValidatePath(Entry.getValue().File))
        return Err;
      for (PathRef Included : Entry.getValue().DirectIncludes)
        if (auto Err = ValidatePath(Included))
          return Err;
    }
    for (const auto &Entry : ShardVersions)
      if (auto Err = ValidatePath(Entry.first()))
        return Err;
    for (const auto &Entry : IndexFailures)
      if (auto Err = ValidatePath(Entry.first()))
        return Err;

    llvm::StringMap<Path> ProjectedTUs;
    for (llvm::StringRef TU : KnownTUs.keys()) {
      auto NewTU = mapPathAfterRenames(TU, Renames);
      if (!NewTU)
        return NewTU.takeError();
      ProjectedTUs.try_emplace(TU, std::move(*NewTU));
    }

    // Seed invalidation with files and include edges in either renamed
    // namespace. Destination entries are stale when a rename overwrites them.
    for (const auto &Entry : IndexedFiles) {
      const IndexedFile &File = Entry.getValue();
      if (AffectedByRename(File.File) ||
          llvm::any_of(File.DirectIncludes, AffectedByRename))
        InvalidatedFiles.insert(File.File);
    }

    // A TU must be reindexed if any file in its transitive include graph was
    // invalidated. Compute that reverse closure from the persisted graph.
    bool Changed;
    do {
      Changed = false;
      for (const auto &Entry : IndexedFiles) {
        const IndexedFile &File = Entry.getValue();
        if (InvalidatedFiles.contains(File.File))
          continue;
        if (llvm::any_of(File.DirectIncludes, [&](PathRef Included) {
              return InvalidatedFiles.contains(Included);
            })) {
          InvalidatedFiles.insert(File.File);
          Changed = true;
        }
      }
    } while (Changed);

    for (const auto &Entry : ShardVersions)
      if (AffectedByRename(Entry.first()) ||
          InvalidatedFiles.contains(Entry.first()))
        InvalidatedFiles.insert(Entry.first());

    for (llvm::StringRef File : InvalidatedFiles.keys()) {
      RemovedPaths.push_back(File.str());
      IndexedFiles.erase(File);
      ShardVersions.erase(File);
      IndexFailures.erase(File);
    }

    std::vector<Path> InvalidatedFailures;
    for (const auto &Entry : IndexFailures)
      if (AffectedByRename(Entry.first()) ||
          InvalidatedFiles.contains(Entry.first()))
        InvalidatedFailures.push_back(Entry.first().str());
    for (PathRef File : InvalidatedFailures)
      IndexFailures.erase(File);

    for (llvm::StringRef TU : KnownTUs.keys()) {
      const Path &NewTU = ProjectedTUs.find(TU)->getValue();
      NewKnownTUs.insert(NewTU);
      if (AffectedByRename(TU) || InvalidatedFiles.contains(TU))
        TranslationUnits.insert(NewTU);
    }

    KnownTUs = std::move(NewKnownTUs);
    if (!InvalidatedFiles.empty() || !TranslationUnits.empty())
      IncludeGraphError.reset();
  }

  llvm::Error RemoveErrors = llvm::Error::success();
  for (PathRef OldPath : RemovedPaths) {
    RemoveErrors =
        llvm::joinErrors(std::move(RemoveErrors),
                         IndexStorageFactory(OldPath)->removeShard(OldPath));
    IndexedSymbols.update(URI::create(OldPath).toString(),
                          /*Symbols=*/nullptr, /*Refs=*/nullptr,
                          /*Relations=*/nullptr, /*CountReferences=*/false);
    Rebuilder.indexedTU();
  }
  std::vector<BackgroundQueue::Task> Tasks;
  Tasks.reserve(TranslationUnits.size());
  for (llvm::StringRef TU : TranslationUnits.keys())
    Tasks.push_back(indexFileTask(TU.str(),
                                  /*BypassDuplicateSuppression=*/true));
  Queue.append(std::move(Tasks));
  return RemoveErrors;
}

BackgroundQueue::Task BackgroundIndex::changedFilesTask(
    const std::vector<std::string> &ChangedFiles) {
  BackgroundQueue::Task T([this, ChangedFiles] {
    trace::Span Tracer("BackgroundIndexEnqueue");

    std::optional<WithContext> WithProvidedContext;
    if (ContextProvider)
      WithProvidedContext.emplace(ContextProvider(/*Path=*/""));

    // We're doing this asynchronously, because we'll read shards here too.
    log("Enqueueing {0} commands for indexing", ChangedFiles.size());
    SPAN_ATTACH(Tracer, "files", int64_t(ChangedFiles.size()));

    auto NeedsReIndexing = loadProject(std::move(ChangedFiles));
    // Run indexing for files that need to be updated.
    std::shuffle(NeedsReIndexing.begin(), NeedsReIndexing.end(),
                 std::mt19937(std::random_device{}()));
    std::vector<BackgroundQueue::Task> Tasks;
    Tasks.reserve(NeedsReIndexing.size());
    for (const auto &File : NeedsReIndexing)
      Tasks.push_back(indexFileTask(std::move(File)));
    Queue.append(std::move(Tasks));
  });

  T.QueuePri = LoadShards;
  T.ThreadPri = llvm::ThreadPriority::Default;
  return T;
}

static llvm::StringRef filenameWithoutExtension(llvm::StringRef Path) {
  Path = llvm::sys::path::filename(Path);
  return Path.drop_back(llvm::sys::path::extension(Path).size());
}

BackgroundQueue::Task
BackgroundIndex::indexFileTask(std::string Path,
                               bool BypassDuplicateSuppression) {
  std::string Tag = filenameWithoutExtension(Path).str();
  uint64_t Key = BypassDuplicateSuppression ? 0 : llvm::xxh3_64bits(Path);
  BackgroundQueue::Task T([this, Path(std::move(Path))] {
    std::optional<WithContext> WithProvidedContext;
    if (ContextProvider)
      WithProvidedContext.emplace(ContextProvider(Path));
    auto Cmd = CDB.getCompileCommand(Path);
    if (!Cmd) {
      std::lock_guard<std::mutex> Lock(ShardVersionsMu);
      IndexFailures[Path] = "no compilation command is available";
      return;
    }
    if (auto Error = index(std::move(*Cmd))) {
      std::string Message = llvm::toString(std::move(Error));
      {
        std::lock_guard<std::mutex> Lock(ShardVersionsMu);
        IndexFailures[Path] = Message;
      }
      elog("Indexing {0} failed: {1}", Path, Message);
      return;
    }
    std::lock_guard<std::mutex> Lock(ShardVersionsMu);
    IndexFailures.erase(Path);
  });
  T.QueuePri = IndexFile;
  T.ThreadPri = IndexingPriority;
  T.Tag = std::move(Tag);
  T.Key = Key;
  return T;
}

void BackgroundIndex::boostRelated(llvm::StringRef Path) {
  if (isHeaderFile(Path))
    Queue.boost(filenameWithoutExtension(Path), IndexBoostedFile);
}

/// Given index results from a TU, only update symbols coming from files that
/// are different or missing from than \p ShardVersionsSnapshot. Also stores new
/// index information on IndexStorage.
void BackgroundIndex::update(
    llvm::StringRef MainFile, IndexFileIn Index,
    const llvm::StringMap<ShardVersion> &ShardVersionsSnapshot,
    bool HadErrors) {
  // Keys are URIs.
  llvm::StringMap<std::pair<Path, FileDigest>> FilesToUpdate;
  // Note that sources do not contain any information regarding missing headers,
  // since we don't even know what absolute path they should fall in.
  for (const auto &IndexIt : *Index.Sources) {
    const auto &IGN = IndexIt.getValue();
    auto AbsPath = URI::resolve(IGN.URI, MainFile);
    if (!AbsPath) {
      elog("Failed to resolve URI: {0}", AbsPath.takeError());
      continue;
    }
    const auto DigestIt = ShardVersionsSnapshot.find(*AbsPath);
    // File has different contents, or indexing was successful this time.
    if (DigestIt == ShardVersionsSnapshot.end() ||
        DigestIt->getValue().Digest != IGN.Digest ||
        (DigestIt->getValue().HadErrors && !HadErrors))
      FilesToUpdate[IGN.URI] = {std::move(*AbsPath), IGN.Digest};
  }

  // Shard slabs into files.
  FileShardedIndex ShardedIndex(std::move(Index));

  // Build and store new slabs for each updated file.
  for (const auto &FileIt : FilesToUpdate) {
    auto Uri = FileIt.first();
    auto IF = ShardedIndex.getShard(Uri);
    assert(IF && "no shard for file in Index.Sources?");
    PathRef Path = FileIt.getValue().first;

    // Only store command line hash for main files of the TU, since our
    // current model keeps only one version of a header file.
    if (Path != MainFile)
      IF->Cmd.reset();

    // We need to store shards before updating the index, since the latter
    // consumes slabs.
    // FIXME: Also skip serializing the shard if it is already up-to-date.
    if (auto Error = IndexStorageFactory(Path)->storeShard(Path, *IF))
      elog("Failed to write background-index shard for file {0}: {1}", Path,
           std::move(Error));

    {
      std::lock_guard<std::mutex> Lock(ShardVersionsMu);
      const auto &Hash = FileIt.getValue().second;
      auto DigestIt = ShardVersions.try_emplace(Path);
      ShardVersion &SV = DigestIt.first->second;
      // Skip if file is already up to date, unless previous index was broken
      // and this one is not.
      if (!DigestIt.second && SV.Digest == Hash && SV.HadErrors && !HadErrors)
        continue;
      SV.Digest = Hash;
      SV.HadErrors = HadErrors;

      const auto &Sources = *IF->Sources;
      auto Source = Sources.find(Uri);
      if (Source == Sources.end()) {
        IncludeGraphError =
            llvm::formatv("missing source node for {0}", Path).str();
      } else {
        auto File = indexedFile(Path, Source->getValue(), MainFile);
        if (!File)
          IncludeGraphError = llvm::toString(File.takeError());
        else
          IndexedFiles[Path] = std::move(*File);
      }

      // This can override a newer version that is added in another thread, if
      // this thread sees the older version but finishes later. This should be
      // rare in practice.
      IndexedSymbols.update(
          Uri, std::make_unique<SymbolSlab>(std::move(*IF->Symbols)),
          std::make_unique<RefSlab>(std::move(*IF->Refs)),
          std::make_unique<RelationSlab>(std::move(*IF->Relations)),
          Path == MainFile);
    }
  }
}

llvm::Error BackgroundIndex::index(tooling::CompileCommand Cmd) {
  trace::Span Tracer("BackgroundIndex");
  SPAN_ATTACH(Tracer, "file", Cmd.Filename);
  auto AbsolutePath = getAbsolutePath(Cmd);

  auto FS = TFS.view(Cmd.Directory);
  auto Buf = FS->getBufferForFile(AbsolutePath);
  if (!Buf)
    return llvm::errorCodeToError(Buf.getError());
  auto Hash = digest(Buf->get()->getBuffer());

  // Take a snapshot of the versions to avoid locking for each file in the TU.
  llvm::StringMap<ShardVersion> ShardVersionsSnapshot;
  {
    std::lock_guard<std::mutex> Lock(ShardVersionsMu);
    ShardVersionsSnapshot = ShardVersions;
  }

  vlog("Indexing {0} (digest:={1})", Cmd.Filename, llvm::toHex(Hash));
  ParseInputs Inputs;
  Inputs.TFS = &TFS;
  Inputs.CompileCommand = std::move(Cmd);
  IgnoreDiagnostics IgnoreDiags;
  auto CI = buildCompilerInvocation(Inputs, IgnoreDiags);
  if (!CI)
    return error("Couldn't build compiler invocation");

  auto Clang =
      prepareCompilerInstance(std::move(CI), /*Preamble=*/nullptr,
                              std::move(*Buf), std::move(FS), IgnoreDiags);
  if (!Clang)
    return error("Couldn't build compiler instance");

  SymbolCollector::Options IndexOpts;
  // Creates a filter to not collect index results from files with unchanged
  // digests.
  IndexOpts.FileFilter = [&ShardVersionsSnapshot](const SourceManager &SM,
                                                  FileID FID) {
    const auto F = SM.getFileEntryRefForID(FID);
    if (!F)
      return false; // Skip invalid files.
    auto AbsPath = getCanonicalPath(*F, SM.getFileManager());
    if (!AbsPath)
      return false; // Skip files without absolute path.
    auto Digest = digestFile(SM, FID);
    if (!Digest)
      return false;
    auto D = ShardVersionsSnapshot.find(*AbsPath);
    if (D != ShardVersionsSnapshot.end() && D->second.Digest == Digest &&
        !D->second.HadErrors)
      return false; // Skip files that haven't changed, without errors.
    return true;
  };
  IndexOpts.CollectMainFileRefs = true;

  IndexFileIn Index;
  auto Action = createStaticIndexingAction(
      IndexOpts, [&](IndexFileIn Result) { Index = std::move(Result); });

  // We're going to run clang here, and it could potentially crash.
  // We could use CrashRecoveryContext to try to make indexing crashes nonfatal,
  // but the leaky "recovery" is pretty scary too in a long-running process.
  // If crashes are a real problem, maybe we should fork a child process.

  const FrontendInputFile &Input = Clang->getFrontendOpts().Inputs.front();
  if (!Action->BeginSourceFile(*Clang, Input))
    return error("BeginSourceFile() failed");
  if (llvm::Error Err = Action->Execute())
    return Err;

  Action->EndSourceFile();

  Index.Cmd = Inputs.CompileCommand;
  assert(Index.Symbols && Index.Refs && Index.Sources &&
         "Symbols, Refs and Sources must be set.");

  log("Indexed {0} ({1} symbols, {2} refs, {3} files)",
      Inputs.CompileCommand.Filename, Index.Symbols->size(),
      Index.Refs->numRefs(), Index.Sources->size());
  SPAN_ATTACH(Tracer, "symbols", int(Index.Symbols->size()));
  SPAN_ATTACH(Tracer, "refs", int(Index.Refs->numRefs()));
  SPAN_ATTACH(Tracer, "sources", int(Index.Sources->size()));

  bool HadErrors = Clang->hasDiagnostics() &&
                   Clang->getDiagnostics().hasUncompilableErrorOccurred();
  if (HadErrors) {
    log("Failed to compile {0}, index may be incomplete", AbsolutePath);
    for (auto &It : *Index.Sources)
      It.second.Flags |= IncludeGraphNode::SourceFlag::HadErrors;
  }
  update(AbsolutePath, std::move(Index), ShardVersionsSnapshot, HadErrors);

  Rebuilder.indexedTU();
  return llvm::Error::success();
}

// Restores shards for \p MainFiles from index storage. Then checks staleness of
// those shards and returns a list of TUs that needs to be indexed to update
// staleness.
std::vector<std::string>
BackgroundIndex::loadProject(std::vector<std::string> MainFiles) {
  // Drop files where background indexing is disabled in config.
  if (ContextProvider)
    llvm::erase_if(MainFiles, [&](const std::string &TU) {
      // Load the config for each TU, as indexing may be selectively enabled.
      WithContext WithProvidedContext(ContextProvider(TU));
      return Config::current().Index.Background ==
             Config::BackgroundPolicy::Skip;
    });
  Rebuilder.startLoading();
  // Load shards for all of the mainfiles.
  const std::vector<LoadedShard> Result =
      loadIndexShards(MainFiles, IndexStorageFactory, CDB);
  size_t LoadedShards = 0;
  {
    // Update in-memory state.
    std::lock_guard<std::mutex> Lock(ShardVersionsMu);
    for (auto &LS : Result) {
      if (!LS.Shard)
        continue;
      auto SS =
          LS.Shard->Symbols
              ? std::make_unique<SymbolSlab>(std::move(*LS.Shard->Symbols))
              : nullptr;
      auto RS = LS.Shard->Refs
                    ? std::make_unique<RefSlab>(std::move(*LS.Shard->Refs))
                    : nullptr;
      auto RelS =
          LS.Shard->Relations
              ? std::make_unique<RelationSlab>(std::move(*LS.Shard->Relations))
              : nullptr;
      ShardVersion &SV = ShardVersions[LS.AbsolutePath];
      SV.Digest = LS.Digest;
      SV.HadErrors = LS.HadErrors;
      ++LoadedShards;

      if (LS.Shard->Sources) {
        std::string FileURI = URI::create(LS.AbsolutePath).toString();
        auto Source = LS.Shard->Sources->find(FileURI);
        if (Source == LS.Shard->Sources->end()) {
          IncludeGraphError =
              llvm::formatv("missing loaded source node for {0}",
                            LS.AbsolutePath)
                  .str();
        } else {
          auto File =
              indexedFile(LS.AbsolutePath, Source->getValue(), LS.AbsolutePath);
          if (!File)
            IncludeGraphError = llvm::toString(File.takeError());
          else
            IndexedFiles[LS.AbsolutePath] = std::move(*File);
        }
      }

      IndexedSymbols.update(URI::create(LS.AbsolutePath).toString(),
                            std::move(SS), std::move(RS), std::move(RelS),
                            LS.CountReferences);
    }
  }
  Rebuilder.loadedShard(LoadedShards);
  Rebuilder.doneLoading();

  auto FS = TFS.view(/*CWD=*/std::nullopt);
  llvm::DenseSet<PathRef> TUsToIndex;
  // We'll accept data from stale shards, but ensure the files get reindexed
  // soon.
  for (auto &LS : Result) {
    if (!shardIsStale(LS, FS.get()))
      continue;
    PathRef TUForFile = LS.DependentTU;
    assert(!TUForFile.empty() && "File without a TU!");

    // FIXME: Currently, we simply schedule indexing on a TU whenever any of
    // its dependencies needs re-indexing. We might do it smarter by figuring
    // out a minimal set of TUs that will cover all the stale dependencies.
    // FIXME: Try looking at other TUs if no compile commands are available
    // for this TU, i.e TU was deleted after we performed indexing.
    TUsToIndex.insert(TUForFile);
  }

  return {TUsToIndex.begin(), TUsToIndex.end()};
}

void BackgroundIndex::profile(MemoryTree &MT) const {
  IndexedSymbols.profile(MT.child("slabs"));
  // We don't want to mix memory used by index and symbols, so call base class.
  MT.child("index").addUsage(SwapIndex::estimateMemoryUsage());
}
} // namespace clangd
} // namespace clang
