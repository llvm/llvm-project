//===-- Background.cpp - Build an index in a background thread ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "index/Background.h"
#include "Compiler.h"
#include "Context.h"
#include "FSProvider.h"
#include "Headers.h"
#include "Logger.h"
#include "ParsedAST.h"
#include "Path.h"
#include "SourceCode.h"
#include "Symbol.h"
#include "Threading.h"
#include "Trace.h"
#include "URI.h"
#include "index/BackgroundIndexLoader.h"
#include "index/FileIndex.h"
#include "index/IndexAction.h"
#include "index/MemIndex.h"
#include "index/Ref.h"
#include "index/Relation.h"
#include "index/Serialization.h"
#include "index/SymbolCollector.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Driver/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Threading.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

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
      auto Path = URI::resolve(FileURI, HintPath);
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
    elog("Background-index: Couldn't read {0} to validate stored index: {1}",
         LS.AbsolutePath, Buf.getError().message());
    // There is no point in indexing an unreadable file.
    return false;
  }
  return digest(Buf->get()->getBuffer()) != LS.Digest;
}

} // namespace

BackgroundIndex::BackgroundIndex(
    Context BackgroundContext, const FileSystemProvider &FSProvider,
    const GlobalCompilationDatabase &CDB,
    BackgroundIndexStorage::Factory IndexStorageFactory, size_t ThreadPoolSize)
    : SwapIndex(std::make_unique<MemIndex>()), FSProvider(FSProvider),
      CDB(CDB), BackgroundContext(std::move(BackgroundContext)),
      Rebuilder(this, &IndexedSymbols, ThreadPoolSize),
      IndexStorageFactory(std::move(IndexStorageFactory)),
      CommandsChanged(
          CDB.watch([&](const std::vector<std::string> &ChangedFiles) {
            enqueue(ChangedFiles);
          })) {
  assert(ThreadPoolSize > 0 && "Thread pool size can't be zero.");
  assert(this->IndexStorageFactory && "Storage factory can not be null!");
  for (unsigned I = 0; I < ThreadPoolSize; ++I) {
    ThreadPool.runAsync("background-worker-" + llvm::Twine(I + 1), [this] {
      WithContext Ctx(this->BackgroundContext.clone());
      Queue.work([&] { Rebuilder.idle(); });
    });
  }
}

BackgroundIndex::~BackgroundIndex() {
  stop();
  ThreadPool.wait();
}

BackgroundQueue::Task BackgroundIndex::changedFilesTask(
    const std::vector<std::string> &ChangedFiles) {
  BackgroundQueue::Task T([this, ChangedFiles] {
    trace::Span Tracer("BackgroundIndexEnqueue");
    // We're doing this asynchronously, because we'll read shards here too.
    log("Enqueueing {0} commands for indexing", ChangedFiles.size());
    SPAN_ATTACH(Tracer, "files", int64_t(ChangedFiles.size()));

    auto NeedsReIndexing = loadProject(std::move(ChangedFiles));
    // Run indexing for files that need to be updated.
    std::shuffle(NeedsReIndexing.begin(), NeedsReIndexing.end(),
                 std::mt19937(std::random_device{}()));
    std::vector<BackgroundQueue::Task> Tasks;
    Tasks.reserve(NeedsReIndexing.size());
    for (auto &Cmd : NeedsReIndexing)
      Tasks.push_back(indexFileTask(std::move(Cmd)));
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
BackgroundIndex::indexFileTask(tooling::CompileCommand Cmd) {
  BackgroundQueue::Task T([this, Cmd] {
    // We can't use llvm::StringRef here since we are going to
    // move from Cmd during the call below.
    const std::string FileName = Cmd.Filename;
    if (auto Error = index(std::move(Cmd)))
      elog("Indexing {0} failed: {1}", FileName, std::move(Error));
  });
  T.QueuePri = IndexFile;
  T.Tag = filenameWithoutExtension(Cmd.Filename);
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
  // Partition symbols/references into files.
  struct File {
    llvm::DenseSet<const Symbol *> Symbols;
    llvm::DenseSet<const Ref *> Refs;
    llvm::DenseSet<const Relation *> Relations;
    FileDigest Digest;
  };
  llvm::StringMap<File> Files;
  URIToFileCache URICache(MainFile);
  for (const auto &IndexIt : *Index.Sources) {
    const auto &IGN = IndexIt.getValue();
    // Note that sources do not contain any information regarding missing
    // headers, since we don't even know what absolute path they should fall in.
    const auto AbsPath = URICache.resolve(IGN.URI);
    const auto DigestIt = ShardVersionsSnapshot.find(AbsPath);
    // File has different contents, or indexing was successfull this time.
    if (DigestIt == ShardVersionsSnapshot.end() ||
        DigestIt->getValue().Digest != IGN.Digest ||
        (DigestIt->getValue().HadErrors && !HadErrors))
      Files.try_emplace(AbsPath).first->getValue().Digest = IGN.Digest;
  }
  // This map is used to figure out where to store relations.
  llvm::DenseMap<SymbolID, File *> SymbolIDToFile;
  for (const auto &Sym : *Index.Symbols) {
    if (Sym.CanonicalDeclaration) {
      auto DeclPath = URICache.resolve(Sym.CanonicalDeclaration.FileURI);
      const auto FileIt = Files.find(DeclPath);
      if (FileIt != Files.end()) {
        FileIt->second.Symbols.insert(&Sym);
        SymbolIDToFile[Sym.ID] = &FileIt->second;
      }
    }
    // For symbols with different declaration and definition locations, we store
    // the full symbol in both the header file and the implementation file, so
    // that merging can tell the preferred symbols (from canonical headers) from
    // other symbols (e.g. forward declarations).
    if (Sym.Definition &&
        Sym.Definition.FileURI != Sym.CanonicalDeclaration.FileURI) {
      auto DefPath = URICache.resolve(Sym.Definition.FileURI);
      const auto FileIt = Files.find(DefPath);
      if (FileIt != Files.end())
        FileIt->second.Symbols.insert(&Sym);
    }
  }
  llvm::DenseMap<const Ref *, SymbolID> RefToIDs;
  for (const auto &SymRefs : *Index.Refs) {
    for (const auto &R : SymRefs.second) {
      auto Path = URICache.resolve(R.Location.FileURI);
      const auto FileIt = Files.find(Path);
      if (FileIt != Files.end()) {
        auto &F = FileIt->getValue();
        RefToIDs[&R] = SymRefs.first;
        F.Refs.insert(&R);
      }
    }
  }
  for (const auto &Rel : *Index.Relations) {
    const auto FileIt = SymbolIDToFile.find(Rel.Subject);
    if (FileIt != SymbolIDToFile.end())
      FileIt->second->Relations.insert(&Rel);
  }

  // Build and store new slabs for each updated file.
  for (const auto &FileIt : Files) {
    llvm::StringRef Path = FileIt.getKey();
    SymbolSlab::Builder Syms;
    RefSlab::Builder Refs;
    RelationSlab::Builder Relations;
    for (const auto *S : FileIt.second.Symbols)
      Syms.insert(*S);
    for (const auto *R : FileIt.second.Refs)
      Refs.insert(RefToIDs[R], *R);
    for (const auto *Rel : FileIt.second.Relations)
      Relations.insert(*Rel);
    auto SS = std::make_unique<SymbolSlab>(std::move(Syms).build());
    auto RS = std::make_unique<RefSlab>(std::move(Refs).build());
    auto RelS = std::make_unique<RelationSlab>(std::move(Relations).build());
    auto IG = std::make_unique<IncludeGraph>(
        getSubGraph(URI::create(Path), Index.Sources.getValue()));

    // We need to store shards before updating the index, since the latter
    // consumes slabs.
    // FIXME: Also skip serializing the shard if it is already up-to-date.
    BackgroundIndexStorage *IndexStorage = IndexStorageFactory(Path);
    IndexFileOut Shard;
    Shard.Symbols = SS.get();
    Shard.Refs = RS.get();
    Shard.Relations = RelS.get();
    Shard.Sources = IG.get();

    // Only store command line hash for main files of the TU, since our
    // current model keeps only one version of a header file.
    if (Path == MainFile)
      Shard.Cmd = Index.Cmd.getPointer();

    if (auto Error = IndexStorage->storeShard(Path, Shard))
      elog("Failed to write background-index shard for file {0}: {1}", Path,
           std::move(Error));

    {
      std::lock_guard<std::mutex> Lock(ShardVersionsMu);
      auto Hash = FileIt.second.Digest;
      auto DigestIt = ShardVersions.try_emplace(Path);
      ShardVersion &SV = DigestIt.first->second;
      // Skip if file is already up to date, unless previous index was broken
      // and this one is not.
      if (!DigestIt.second && SV.Digest == Hash && SV.HadErrors && !HadErrors)
        continue;
      SV.Digest = Hash;
      SV.HadErrors = HadErrors;

      // This can override a newer version that is added in another thread, if
      // this thread sees the older version but finishes later. This should be
      // rare in practice.
      IndexedSymbols.update(Path, std::move(SS), std::move(RS), std::move(RelS),
                            Path == MainFile);
    }
  }
}

llvm::Error BackgroundIndex::index(tooling::CompileCommand Cmd) {
  trace::Span Tracer("BackgroundIndex");
  SPAN_ATTACH(Tracer, "file", Cmd.Filename);
  auto AbsolutePath = getAbsolutePath(Cmd);

  auto FS = FSProvider.getFileSystem();
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
  Inputs.FS = std::move(FS);
  Inputs.FS->setCurrentWorkingDirectory(Cmd.Directory);
  Inputs.CompileCommand = std::move(Cmd);
  IgnoreDiagnostics IgnoreDiags;
  auto CI = buildCompilerInvocation(Inputs, IgnoreDiags);
  if (!CI)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Couldn't build compiler invocation");
  auto Clang = prepareCompilerInstance(std::move(CI), /*Preamble=*/nullptr,
                                       std::move(*Buf), Inputs.FS, IgnoreDiags);
  if (!Clang)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Couldn't build compiler instance");

  SymbolCollector::Options IndexOpts;
  // Creates a filter to not collect index results from files with unchanged
  // digests.
  IndexOpts.FileFilter = [&ShardVersionsSnapshot](const SourceManager &SM,
                                                  FileID FID) {
    const auto *F = SM.getFileEntryForID(FID);
    if (!F)
      return false; // Skip invalid files.
    auto AbsPath = getCanonicalPath(F, SM);
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

  IndexFileIn Index;
  auto Action = createStaticIndexingAction(
      IndexOpts, [&](SymbolSlab S) { Index.Symbols = std::move(S); },
      [&](RefSlab R) { Index.Refs = std::move(R); },
      [&](RelationSlab R) { Index.Relations = std::move(R); },
      [&](IncludeGraph IG) { Index.Sources = std::move(IG); });

  // We're going to run clang here, and it could potentially crash.
  // We could use CrashRecoveryContext to try to make indexing crashes nonfatal,
  // but the leaky "recovery" is pretty scary too in a long-running process.
  // If crashes are a real problem, maybe we should fork a child process.

  const FrontendInputFile &Input = Clang->getFrontendOpts().Inputs.front();
  if (!Action->BeginSourceFile(*Clang, Input))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "BeginSourceFile() failed");
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
std::vector<tooling::CompileCommand>
BackgroundIndex::loadProject(std::vector<std::string> MainFiles) {
  std::vector<tooling::CompileCommand> NeedsReIndexing;

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

      IndexedSymbols.update(LS.AbsolutePath, std::move(SS), std::move(RS),
                            std::move(RelS), LS.CountReferences);
    }
  }
  Rebuilder.loadedShard(LoadedShards);
  Rebuilder.doneLoading();

  auto FS = FSProvider.getFileSystem();
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

  for (PathRef TU : TUsToIndex) {
    auto Cmd = CDB.getCompileCommand(TU);
    if (!Cmd)
      continue;
    NeedsReIndexing.emplace_back(std::move(*Cmd));
  }

  return NeedsReIndexing;
}

} // namespace clangd
} // namespace clang
