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
#include "Trace.h"
#include "index/IndexAction.h"
#include "index/MemIndex.h"
#include "index/Serialization.h"
#include "llvm/Support/SHA1.h"
#include <random>

using namespace llvm;
namespace clang {
namespace clangd {

BackgroundIndex::BackgroundIndex(Context BackgroundContext,
                                 StringRef ResourceDir,
                                 const FileSystemProvider &FSProvider)
    : SwapIndex(llvm::make_unique<MemIndex>()), ResourceDir(ResourceDir),
      FSProvider(FSProvider), BackgroundContext(std::move(BackgroundContext)),
      Thread([this] { run(); }) {}

BackgroundIndex::~BackgroundIndex() {
  stop();
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
  WithContext Background(std::move(BackgroundContext));
  while (true) {
    llvm::Optional<Task> Task;
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

llvm::Error BackgroundIndex::index(tooling::CompileCommand Cmd) {
  trace::Span Tracer("BackgroundIndex");
  SPAN_ATTACH(Tracer, "file", Cmd.Filename);
  SmallString<128> AbsolutePath;
  if (llvm::sys::path::is_absolute(Cmd.Filename)) {
    AbsolutePath = Cmd.Filename;
  } else {
    AbsolutePath = Cmd.Directory;
    llvm::sys::path::append(AbsolutePath, Cmd.Filename);
  }

  auto FS = FSProvider.getFileSystem();
  auto Buf = FS->getBufferForFile(AbsolutePath);
  if (!Buf)
    return errorCodeToError(Buf.getError());
  StringRef Contents = Buf->get()->getBuffer();
  auto Hash = SHA1::hash({(const uint8_t *)Contents.data(), Contents.size()});

  if (FileHash.lookup(AbsolutePath) == Hash) {
    vlog("No need to index {0}, already up to date", AbsolutePath);
    return Error::success();
  }

  log("Indexing {0}", Cmd.Filename, toHex(Hash));
  ParseInputs Inputs;
  Inputs.FS = std::move(FS);
  Inputs.FS->setCurrentWorkingDirectory(Cmd.Directory);
  Inputs.CompileCommand = std::move(Cmd);
  auto CI = buildCompilerInvocation(Inputs);
  if (!CI)
    return createStringError(llvm::inconvertibleErrorCode(),
                             "Couldn't build compiler invocation");
  IgnoreDiagnostics IgnoreDiags;
  auto Clang = prepareCompilerInstance(
      std::move(CI), /*Preamble=*/nullptr, std::move(*Buf),
      std::make_shared<PCHContainerOperations>(), Inputs.FS, IgnoreDiags);
  if (!Clang)
    return createStringError(llvm::inconvertibleErrorCode(),
                             "Couldn't build compiler instance");

  SymbolCollector::Options IndexOpts;
  SymbolSlab Symbols;
  RefSlab Refs;
  IndexFileIn IndexData;
  auto Action = createStaticIndexingAction(
      IndexOpts, [&](SymbolSlab S) { Symbols = std::move(S); },
      [&](RefSlab R) { Refs = std::move(R); });

  // We're going to run clang here, and it could potentially crash.
  // We could use CrashRecoveryContext to try to make indexing crashes nonfatal,
  // but the leaky "recovery" is pretty scary too in a long-running process.
  // If crashes are a real problem, maybe we should fork a child process.

  const FrontendInputFile &Input = Clang->getFrontendOpts().Inputs.front();
  if (!Action->BeginSourceFile(*Clang, Input))
    return createStringError(llvm::inconvertibleErrorCode(),
                             "BeginSourceFile() failed");
  if (!Action->Execute())
    return createStringError(llvm::inconvertibleErrorCode(),
                             "Execute() failed");
  Action->EndSourceFile();

  log("Indexed {0} ({1} symbols, {2} refs)", Inputs.CompileCommand.Filename,
      Symbols.size(), Refs.numRefs());
  SPAN_ATTACH(Tracer, "symbols", int(Symbols.size()));
  SPAN_ATTACH(Tracer, "refs", int(Refs.numRefs()));
  // FIXME: partition the symbols by file rather than TU, to avoid duplication.
  IndexedSymbols.update(AbsolutePath,
                        llvm::make_unique<SymbolSlab>(std::move(Symbols)),
                        llvm::make_unique<RefSlab>(std::move(Refs)));
  FileHash[AbsolutePath] = Hash;

  // FIXME: this should rebuild once-in-a-while, not after every file.
  //       At that point we should use Dex, too.
  vlog("Rebuilding automatic index");
  reset(IndexedSymbols.buildIndex(IndexType::Light));
  return Error::success();
}

} // namespace clangd
} // namespace clang
