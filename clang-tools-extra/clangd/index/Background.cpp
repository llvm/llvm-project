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
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA1.h"
#include <random>
#include <string>

using namespace llvm;
namespace clang {
namespace clangd {

BackgroundIndex::BackgroundIndex(Context BackgroundContext,
                                 StringRef ResourceDir,
                                 const FileSystemProvider &FSProvider,
                                 ArrayRef<std::string> URISchemes,
                                 size_t ThreadPoolSize)
    : SwapIndex(make_unique<MemIndex>()), ResourceDir(ResourceDir),
      FSProvider(FSProvider), BackgroundContext(std::move(BackgroundContext)),
      URISchemes(URISchemes) {
  assert(ThreadPoolSize > 0 && "Thread pool size can't be zero.");
  while (ThreadPoolSize--) {
    ThreadPool.emplace_back([this] { run(); });
    // Set priority to low, since background indexing is a long running task we
    // do not want to eat up cpu when there are any other high priority threads.
    // FIXME: In the future we might want a more general way of handling this to
    // support a tasks with various priorities.
    setThreadPriority(ThreadPool.back(), ThreadPriority::Low);
  }
}

BackgroundIndex::~BackgroundIndex() {
  stop();
  for (auto &Thread : ThreadPool)
    Thread.join();
}

void BackgroundIndex::stop() {
  {
    std::lock_guard<std::mutex> Lock(QueueMu);
    ShouldStop = true;
  }
  QueueCV.notify_all();
}

void BackgroundIndex::run() {
  WithContext Background(BackgroundContext.clone());
  while (true) {
    Optional<Task> Task;
    {
      std::unique_lock<std::mutex> Lock(QueueMu);
      QueueCV.wait(Lock, [&] { return ShouldStop || !Queue.empty(); });
      if (ShouldStop) {
        Queue.clear();
        QueueCV.notify_all();
        return;
      }
      ++NumActiveTasks;
      Task = std::move(Queue.front());
      Queue.pop_front();
    }
    (*Task)();
    {
      std::unique_lock<std::mutex> Lock(QueueMu);
      assert(NumActiveTasks > 0 && "before decrementing");
      --NumActiveTasks;
    }
    QueueCV.notify_all();
  }
}

void BackgroundIndex::blockUntilIdleForTest() {
  std::unique_lock<std::mutex> Lock(QueueMu);
  QueueCV.wait(Lock, [&] { return Queue.empty() && NumActiveTasks == 0; });
}

void BackgroundIndex::enqueue(StringRef Directory,
                              tooling::CompileCommand Cmd) {
  {
    std::lock_guard<std::mutex> Lock(QueueMu);
    enqueueLocked(std::move(Cmd));
  }
  QueueCV.notify_all();
}

void BackgroundIndex::enqueueAll(StringRef Directory,
                                 const tooling::CompilationDatabase &CDB) {
  trace::Span Tracer("BackgroundIndexEnqueueCDB");
  // FIXME: this function may be slow. Perhaps enqueue a task to re-read the CDB
  // from disk and enqueue the commands asynchronously?
  auto Cmds = CDB.getAllCompileCommands();
  SPAN_ATTACH(Tracer, "commands", int64_t(Cmds.size()));
  std::mt19937 Generator(std::random_device{}());
  std::shuffle(Cmds.begin(), Cmds.end(), Generator);
  log("Enqueueing {0} commands for indexing from {1}", Cmds.size(), Directory);
  {
    std::lock_guard<std::mutex> Lock(QueueMu);
    for (auto &Cmd : Cmds)
      enqueueLocked(std::move(Cmd));
  }
  QueueCV.notify_all();
}

void BackgroundIndex::enqueueLocked(tooling::CompileCommand Cmd) {
  Queue.push_back(Bind(
      [this](tooling::CompileCommand Cmd) {
        std::string Filename = Cmd.Filename;
        Cmd.CommandLine.push_back("-resource-dir=" + ResourceDir);
        if (auto Error = index(std::move(Cmd)))
          log("Indexing {0} failed: {1}", Filename, std::move(Error));
      },
      std::move(Cmd)));
}

static BackgroundIndex::FileDigest digest(StringRef Content) {
  return SHA1::hash({(const uint8_t *)Content.data(), Content.size()});
}

static Optional<BackgroundIndex::FileDigest> digestFile(const SourceManager &SM,
                                                        FileID FID) {
  bool Invalid = false;
  StringRef Content = SM.getBufferData(FID, &Invalid);
  if (Invalid)
    return None;
  return digest(Content);
}

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

/// Given index results from a TU, only update files in \p FilesToUpdate.
void BackgroundIndex::update(StringRef MainFile, SymbolSlab Symbols,
                             RefSlab Refs,
                             const StringMap<FileDigest> &FilesToUpdate) {
  // Partition symbols/references into files.
  struct File {
    DenseSet<const Symbol *> Symbols;
    DenseSet<const Ref *> Refs;
  };
  StringMap<File> Files;
  URIToFileCache URICache(MainFile);
  for (const auto &Sym : Symbols) {
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
  for (const auto &SymRefs : Refs) {
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

    std::lock_guard<std::mutex> Lock(DigestsMu);
    // This can override a newer version that is added in another thread,
    // if this thread sees the older version but finishes later. This should be
    // rare in practice.
    IndexedFileDigests[Path] = FilesToUpdate.lookup(Path);
    IndexedSymbols.update(Path,
                          make_unique<SymbolSlab>(std::move(Syms).build()),
                          make_unique<RefSlab>(std::move(Refs).build()));
  }
}

// Creates a filter to not collect index results from files with unchanged
// digests.
// \p FileDigests contains file digests for the current indexed files, and all changed files will be added to \p FilesToUpdate.
decltype(SymbolCollector::Options::FileFilter) createFileFilter(
    const llvm::StringMap<BackgroundIndex::FileDigest> &FileDigests,
    llvm::StringMap<BackgroundIndex::FileDigest> &FilesToUpdate) {
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

Error BackgroundIndex::index(tooling::CompileCommand Cmd) {
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

  log("Indexing {0}", Cmd.Filename, toHex(Hash));
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
  IndexOpts.URISchemes = URISchemes;
  StringMap<FileDigest> FilesToUpdate;
  IndexOpts.FileFilter = createFileFilter(DigestsSnapshot, FilesToUpdate);
  SymbolSlab Symbols;
  RefSlab Refs;
  auto Action = createStaticIndexingAction(
      IndexOpts, [&](SymbolSlab S) { Symbols = std::move(S); },
      [&](RefSlab R) { Refs = std::move(R); });

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

  log("Indexed {0} ({1} symbols, {2} refs)", Inputs.CompileCommand.Filename,
      Symbols.size(), Refs.numRefs());
  SPAN_ATTACH(Tracer, "symbols", int(Symbols.size()));
  SPAN_ATTACH(Tracer, "refs", int(Refs.numRefs()));
  update(AbsolutePath, std::move(Symbols), std::move(Refs), FilesToUpdate);
  {
    // Make sure hash for the main file is always updated even if there is no
    // index data in it.
    std::lock_guard<std::mutex> Lock(DigestsMu);
    IndexedFileDigests[AbsolutePath] = Hash;
  }

  // FIXME: this should rebuild once-in-a-while, not after every file.
  //       At that point we should use Dex, too.
  vlog("Rebuilding automatic index");
  reset(IndexedSymbols.buildIndex(IndexType::Light, DuplicateHandling::Merge,
                                  URISchemes));

  return Error::success();
}

} // namespace clangd
} // namespace clang
