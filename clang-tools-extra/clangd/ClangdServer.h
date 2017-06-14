//===--- ClangdServer.h - Main clangd server code ----------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDSERVER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDSERVER_H

#include "ClangdUnitStore.h"
#include "DraftStore.h"
#include "GlobalCompilationDatabase.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"

#include "ClangdUnit.h"
#include "Protocol.h"

#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

namespace clang {
class PCHContainerOperations;

namespace clangd {

/// Turn a [line, column] pair into an offset in Code.
size_t positionToOffset(StringRef Code, Position P);

/// Turn an offset in Code into a [line, column] pair.
Position offsetToPosition(StringRef Code, size_t Offset);

/// A tag supplied by the FileSytemProvider.
typedef std::string VFSTag;

/// A value of an arbitrary type and VFSTag that was supplied by the
/// FileSystemProvider when this value was computed.
template <class T> class Tagged {
public:
  template <class U>
  Tagged(U &&Value, VFSTag Tag)
      : Value(std::forward<U>(Value)), Tag(std::move(Tag)) {}

  template <class U>
  Tagged(const Tagged<U> &Other) : Value(Other.Value), Tag(Other.Tag) {}

  template <class U>
  Tagged(Tagged<U> &&Other)
      : Value(std::move(Other.Value)), Tag(std::move(Other.Tag)) {}

  T Value;
  VFSTag Tag;
};

template <class T>
Tagged<typename std::decay<T>::type> make_tagged(T &&Value, VFSTag Tag) {
  return Tagged<T>(std::forward<T>(Value), Tag);
}

class DiagnosticsConsumer {
public:
  virtual ~DiagnosticsConsumer() = default;

  /// Called by ClangdServer when \p Diagnostics for \p File are ready.
  virtual void
  onDiagnosticsReady(PathRef File,
                     Tagged<std::vector<DiagWithFixIts>> Diagnostics) = 0;
};

class FileSystemProvider {
public:
  virtual ~FileSystemProvider() = default;
  /// Called by ClangdServer to obtain a vfs::FileSystem to be used for parsing.
  /// Name of the file that will be parsed is passed in \p File.
  ///
  /// \return A filesystem that will be used for all file accesses in clangd.
  /// A Tag returned by this method will be propagated to all results of clangd
  /// that will use this filesystem.
  virtual Tagged<IntrusiveRefCntPtr<vfs::FileSystem>>
  getTaggedFileSystem(PathRef File) = 0;
};

class RealFileSystemProvider : public FileSystemProvider {
public:
  /// \return getRealFileSystem() tagged with default tag, i.e. VFSTag()
  Tagged<IntrusiveRefCntPtr<vfs::FileSystem>>
  getTaggedFileSystem(PathRef File) override;
};

class ClangdServer;

/// Handles running WorkerRequests of ClangdServer on a separate threads.
/// Currently runs only one worker thread.
class ClangdScheduler {
public:
  ClangdScheduler(bool RunSynchronously);
  ~ClangdScheduler();

  /// Add \p Request to the start of the queue. \p Request will be run on a
  /// separate worker thread.
  /// \p Request is scheduled to be executed before all currently added
  /// requests.
  void addToFront(std::function<void()> Request);
  /// Add \p Request to the end of the queue. \p Request will be run on a
  /// separate worker thread.
  /// \p Request is scheduled to be executed after all currently added
  /// requests.
  void addToEnd(std::function<void()> Request);

private:
  bool RunSynchronously;
  std::mutex Mutex;
  /// We run some tasks on a separate thread(parsing, ClangdUnit cleanup).
  /// This thread looks into RequestQueue to find requests to handle and
  /// terminates when Done is set to true.
  std::thread Worker;
  /// Setting Done to true will make the worker thread terminate.
  bool Done = false;
  /// A queue of requests.
  /// FIXME(krasimir): code completion should always have priority over parsing
  /// for diagnostics.
  std::deque<std::function<void()>> RequestQueue;
  /// Condition variable to wake up the worker thread.
  std::condition_variable RequestCV;
};

/// Provides API to manage ASTs for a collection of C++ files and request
/// various language features(currently, only codeCompletion and async
/// diagnostics for tracked files).
class ClangdServer {
public:
  /// Creates a new ClangdServer. If \p RunSynchronously is false, no worker
  /// thread will be created and all requests will be completed synchronously on
  /// the calling thread (this is mostly used for tests). If \p RunSynchronously
  /// is true, a worker thread will be created to parse files in the background
  /// and provide diagnostics results via DiagConsumer.onDiagnosticsReady
  /// callback. File accesses for each instance of parsing will be conducted via
  /// a vfs::FileSystem provided by \p FSProvider. Results of code
  /// completion/diagnostics also include a tag, that \p FSProvider returns
  /// along with the vfs::FileSystem.
  ClangdServer(GlobalCompilationDatabase &CDB,
               DiagnosticsConsumer &DiagConsumer,
               FileSystemProvider &FSProvider, bool RunSynchronously);

  /// Add a \p File to the list of tracked C++ files or update the contents if
  /// \p File is already tracked. Also schedules parsing of the AST for it on a
  /// separate thread. When the parsing is complete, DiagConsumer passed in
  /// constructor will receive onDiagnosticsReady callback.
  void addDocument(PathRef File, StringRef Contents);
  /// Remove \p File from list of tracked files, schedule a request to free
  /// resources associated with it.
  void removeDocument(PathRef File);
  /// Force \p File to be reparsed using the latest contents.
  void forceReparse(PathRef File);

  /// Run code completion for \p File at \p Pos. If \p OverridenContents is not
  /// None, they will used only for code completion, i.e. no diagnostics update
  /// will be scheduled and a draft for \p File will not be updated.
  /// If \p OverridenContents is None, contents of the current draft for \p File
  /// will be used.
  /// This method should only be called for currently tracked files.
  Tagged<std::vector<CompletionItem>>
  codeComplete(PathRef File, Position Pos,
               llvm::Optional<StringRef> OverridenContents = llvm::None);

  /// Run formatting for \p Rng inside \p File.
  std::vector<tooling::Replacement> formatRange(PathRef File, Range Rng);
  /// Run formatting for the whole \p File.
  std::vector<tooling::Replacement> formatFile(PathRef File);
  /// Run formatting after a character was typed at \p Pos in \p File.
  std::vector<tooling::Replacement> formatOnType(PathRef File, Position Pos);

  /// Gets current document contents for \p File. \p File must point to a
  /// currently tracked file.
  /// FIXME(ibiryukov): This function is here to allow offset-to-Position
  /// conversions in outside code, maybe there's a way to get rid of it.
  std::string getDocument(PathRef File);

  /// Only for testing purposes.
  /// Waits until all requests to worker thread are finished and dumps AST for
  /// \p File. \p File must be in the list of added documents.
  std::string dumpAST(PathRef File);

private:
  GlobalCompilationDatabase &CDB;
  DiagnosticsConsumer &DiagConsumer;
  FileSystemProvider &FSProvider;
  DraftStore DraftMgr;
  ClangdUnitStore Units;
  std::shared_ptr<PCHContainerOperations> PCHs;
  // WorkScheduler has to be the last member, because its destructor has to be
  // called before all other members to stop the worker thread that references
  // ClangdServer
  ClangdScheduler WorkScheduler;
};

} // namespace clangd
} // namespace clang

#endif
