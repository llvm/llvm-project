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

#include "Cancellation.h"
#include "ClangdUnit.h"
#include "CodeComplete.h"
#include "FSProvider.h"
#include "Function.h"
#include "GlobalCompilationDatabase.h"
#include "Protocol.h"
#include "TUScheduler.h"
#include "index/FileIndex.h"
#include "index/Index.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <future>
#include <string>
#include <type_traits>
#include <utility>

namespace clang {
class PCHContainerOperations;

namespace clangd {

class DiagnosticsConsumer {
public:
  virtual ~DiagnosticsConsumer() = default;

  /// Called by ClangdServer when \p Diagnostics for \p File are ready.
  virtual void onDiagnosticsReady(PathRef File,
                                  std::vector<Diag> Diagnostics) = 0;
};

/// Manages a collection of source files and derived data (ASTs, indexes),
/// and provides language-aware features such as code completion.
///
/// The primary client is ClangdLSPServer which exposes these features via
/// the Language Server protocol. ClangdServer may also be embedded directly,
/// though its API is not stable over time.
///
/// ClangdServer should be used from a single thread. Many potentially-slow
/// operations have asynchronous APIs and deliver their results on another
/// thread.
/// Such operations support cancellation: if the caller sets up a cancelable
/// context, many operations will notice cancellation and fail early.
/// (ClangdLSPServer uses this to implement $/cancelRequest).
class ClangdServer {
public:
  struct Options {
    /// To process requests asynchronously, ClangdServer spawns worker threads.
    /// If this is zero, no threads are spawned. All work is done on the calling
    /// thread, and callbacks are invoked before "async" functions return.
    unsigned AsyncThreadsCount = getDefaultAsyncThreadsCount();

    /// AST caching policy. The default is to keep up to 3 ASTs in memory.
    ASTRetentionPolicy RetentionPolicy;

    /// Cached preambles are potentially large. If false, store them on disk.
    bool StorePreamblesInMemory = true;

    /// If true, ClangdServer builds a dynamic in-memory index for symbols in
    /// opened files and uses the index to augment code completion results.
    bool BuildDynamicSymbolIndex = false;

    /// URI schemes to use when building the dynamic index.
    /// If empty, the default schemes in SymbolCollector will be used.
    std::vector<std::string> URISchemes;

    /// If set, use this index to augment code completion results.
    SymbolIndex *StaticIndex = nullptr;

    /// The resource directory is used to find internal headers, overriding
    /// defaults and -resource-dir compiler flag).
    /// If None, ClangdServer calls CompilerInvocation::GetResourcePath() to
    /// obtain the standard resource directory.
    llvm::Optional<StringRef> ResourceDir = llvm::None;

    /// Time to wait after a new file version before computing diagnostics.
    std::chrono::steady_clock::duration UpdateDebounce =
        std::chrono::milliseconds(500);
  };
  // Sensible default options for use in tests.
  // Features like indexing must be enabled if desired.
  static Options optsForTest();

  /// Creates a new ClangdServer instance.
  ///
  /// ClangdServer uses \p CDB to obtain compilation arguments for parsing. Note
  /// that ClangdServer only obtains compilation arguments once for each newly
  /// added file (i.e., when processing a first call to addDocument) and reuses
  /// those arguments for subsequent reparses. However, ClangdServer will check
  /// if compilation arguments changed on calls to forceReparse().
  ///
  /// After each parsing request finishes, ClangdServer reports diagnostics to
  /// \p DiagConsumer. Note that a callback to \p DiagConsumer happens on a
  /// worker thread. Therefore, instances of \p DiagConsumer must properly
  /// synchronize access to shared state.
  ClangdServer(GlobalCompilationDatabase &CDB, FileSystemProvider &FSProvider,
               DiagnosticsConsumer &DiagConsumer, const Options &Opts);

  /// Set the root path of the workspace.
  void setRootPath(PathRef RootPath);

  /// Add a \p File to the list of tracked C++ files or update the contents if
  /// \p File is already tracked. Also schedules parsing of the AST for it on a
  /// separate thread. When the parsing is complete, DiagConsumer passed in
  /// constructor will receive onDiagnosticsReady callback.
  void addDocument(PathRef File, StringRef Contents,
                   WantDiagnostics WD = WantDiagnostics::Auto);

  /// Remove \p File from list of tracked files, schedule a request to free
  /// resources associated with it.
  void removeDocument(PathRef File);

  /// Run code completion for \p File at \p Pos.
  /// Request is processed asynchronously.
  ///
  /// This method should only be called for currently tracked files. However, it
  /// is safe to call removeDocument for \p File after this method returns, even
  /// while returned future is not yet ready.
  /// A version of `codeComplete` that runs \p Callback on the processing thread
  /// when codeComplete results become available.
  void codeComplete(PathRef File, Position Pos,
                    const clangd::CodeCompleteOptions &Opts,
                    Callback<CodeCompleteResult> CB);

  /// Provide signature help for \p File at \p Pos.  This method should only be
  /// called for tracked files.
  void signatureHelp(PathRef File, Position Pos, Callback<SignatureHelp> CB);

  /// Get definition of symbol at a specified \p Line and \p Column in \p File.
  void findDefinitions(PathRef File, Position Pos,
                       Callback<std::vector<Location>> CB);

  /// Helper function that returns a path to the corresponding source file when
  /// given a header file and vice versa.
  llvm::Optional<Path> switchSourceHeader(PathRef Path);

  /// Get document highlights for a given position.
  void findDocumentHighlights(PathRef File, Position Pos,
                              Callback<std::vector<DocumentHighlight>> CB);

  /// Get code hover for a given position.
  void findHover(PathRef File, Position Pos,
                 Callback<llvm::Optional<Hover>> CB);

  /// Retrieve the top symbols from the workspace matching a query.
  void workspaceSymbols(StringRef Query, int Limit,
                        Callback<std::vector<SymbolInformation>> CB);

  /// Retrieve the symbols within the specified file.
  void documentSymbols(StringRef File,
                       Callback<std::vector<SymbolInformation>> CB);

  /// Retrieve locations for symbol references.
  void findReferences(PathRef File, Position Pos,
                      Callback<std::vector<Location>> CB);

  /// Run formatting for \p Rng inside \p File with content \p Code.
  llvm::Expected<tooling::Replacements> formatRange(StringRef Code,
                                                    PathRef File, Range Rng);

  /// Run formatting for the whole \p File with content \p Code.
  llvm::Expected<tooling::Replacements> formatFile(StringRef Code,
                                                   PathRef File);

  /// Run formatting after a character was typed at \p Pos in \p File with
  /// content \p Code.
  llvm::Expected<tooling::Replacements>
  formatOnType(StringRef Code, PathRef File, Position Pos);

  /// Rename all occurrences of the symbol at the \p Pos in \p File to
  /// \p NewName.
  void rename(PathRef File, Position Pos, llvm::StringRef NewName,
              Callback<std::vector<tooling::Replacement>> CB);

  /// Only for testing purposes.
  /// Waits until all requests to worker thread are finished and dumps AST for
  /// \p File. \p File must be in the list of added documents.
  void dumpAST(PathRef File, llvm::unique_function<void(std::string)> Callback);
  /// Called when an event occurs for a watched file in the workspace.
  void onFileEvent(const DidChangeWatchedFilesParams &Params);

  /// Returns estimated memory usage for each of the currently open files.
  /// The order of results is unspecified.
  /// Overall memory usage of clangd may be significantly more than reported
  /// here, as this metric does not account (at least) for:
  ///   - memory occupied by static and dynamic index,
  ///   - memory required for in-flight requests,
  /// FIXME: those metrics might be useful too, we should add them.
  std::vector<std::pair<Path, std::size_t>> getUsedBytesPerFile() const;

  /// Returns the active dynamic index if one was built.
  /// This can be useful for testing, debugging, or observing memory usage.
  const SymbolIndex *dynamicIndex() const;

  // Blocks the main thread until the server is idle. Only for use in tests.
  // Returns false if the timeout expires.
  LLVM_NODISCARD bool
  blockUntilIdleForTest(llvm::Optional<double> TimeoutSeconds = 10);

private:
  /// FIXME: This stats several files to find a .clang-format file. I/O can be
  /// slow. Think of a way to cache this.
  llvm::Expected<tooling::Replacements>
  formatCode(llvm::StringRef Code, PathRef File,
             ArrayRef<tooling::Range> Ranges);

  typedef uint64_t DocVersion;

  void consumeDiagnostics(PathRef File, DocVersion Version,
                          std::vector<Diag> Diags);

  tooling::CompileCommand getCompileCommand(PathRef File);

  GlobalCompilationDatabase &CDB;
  DiagnosticsConsumer &DiagConsumer;
  FileSystemProvider &FSProvider;

  /// Used to synchronize diagnostic responses for added and removed files.
  llvm::StringMap<DocVersion> InternalVersion;

  Path ResourceDir;
  // The index used to look up symbols. This could be:
  //   - null (all index functionality is optional)
  //   - the dynamic index owned by ClangdServer (DynamicIdx)
  //   - the static index passed to the constructor
  //   - a merged view of a static and dynamic index (MergedIndex)
  const SymbolIndex *Index;
  // If present, an index of symbols in open files. Read via *Index.
  std::unique_ptr<FileIndex> DynamicIdx;
  // If present, storage for the merged static/dynamic index. Read via *Index.
  std::unique_ptr<SymbolIndex> MergedIndex;

  // GUARDED_BY(CachedCompletionFuzzyFindRequestMutex)
  llvm::StringMap<llvm::Optional<FuzzyFindRequest>>
      CachedCompletionFuzzyFindRequestByFile;
  mutable std::mutex CachedCompletionFuzzyFindRequestMutex;

  // If set, this represents the workspace path.
  llvm::Optional<std::string> RootPath;
  std::shared_ptr<PCHContainerOperations> PCHs;
  /// Used to serialize diagnostic callbacks.
  /// FIXME(ibiryukov): get rid of an extra map and put all version counters
  /// into CppFile.
  std::mutex DiagnosticsMutex;
  /// Maps from a filename to the latest version of reported diagnostics.
  llvm::StringMap<DocVersion> ReportedDiagnosticVersions;
  // WorkScheduler has to be the last member, because its destructor has to be
  // called before all other members to stop the worker thread that references
  // ClangdServer.
  TUScheduler WorkScheduler;
};

} // namespace clangd
} // namespace clang

#endif
