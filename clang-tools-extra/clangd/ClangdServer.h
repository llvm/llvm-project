//===--- ClangdServer.h - Main clangd server code ----------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDSERVER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDSERVER_H

#include "../clang-tidy/ClangTidyOptions.h"
#include "CodeComplete.h"
#include "GlobalCompilationDatabase.h"
#include "Hover.h"
#include "Protocol.h"
#include "SemanticHighlighting.h"
#include "TUScheduler.h"
#include "XRefs.h"
#include "index/Background.h"
#include "index/FileIndex.h"
#include "index/Index.h"
#include "refactor/Rename.h"
#include "refactor/Tweak.h"
#include "support/Cancellation.h"
#include "support/FSProvider.h"
#include "support/Function.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <future>
#include <string>
#include <type_traits>
#include <utility>

namespace clang {
namespace clangd {

/// When set, used by ClangdServer to get clang-tidy options for each particular
/// file. Must be thread-safe. We use this instead of ClangTidyOptionsProvider
/// to allow reading tidy configs from the VFS used for parsing.
using ClangTidyOptionsBuilder = std::function<tidy::ClangTidyOptions(
    llvm::vfs::FileSystem &, llvm::StringRef /*File*/)>;

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
  /// Interface with hooks for users of ClangdServer to be notified of events.
  class Callbacks {
  public:
    virtual ~Callbacks() = default;

    /// Called by ClangdServer when \p Diagnostics for \p File are ready.
    /// May be called concurrently for separate files, not for a single file.
    virtual void onDiagnosticsReady(PathRef File, llvm::StringRef Version,
                                    std::vector<Diag> Diagnostics) {}
    /// Called whenever the file status is updated.
    /// May be called concurrently for separate files, not for a single file.
    virtual void onFileUpdated(PathRef File, const TUStatus &Status) {}

    /// Called by ClangdServer when some \p Highlightings for \p File are ready.
    /// May be called concurrently for separate files, not for a single file.
    virtual void
    onHighlightingsReady(PathRef File, llvm::StringRef Version,
                         std::vector<HighlightingToken> Highlightings) {}

    /// Called when background indexing tasks are enqueued/started/completed.
    /// Not called concurrently.
    virtual void
    onBackgroundIndexProgress(const BackgroundQueue::Stats &Stats) {}
  };

  struct Options {
    /// To process requests asynchronously, ClangdServer spawns worker threads.
    /// If this is zero, no threads are spawned. All work is done on the calling
    /// thread, and callbacks are invoked before "async" functions return.
    unsigned AsyncThreadsCount = getDefaultAsyncThreadsCount();

    /// AST caching policy. The default is to keep up to 3 ASTs in memory.
    ASTRetentionPolicy RetentionPolicy;

    /// Cached preambles are potentially large. If false, store them on disk.
    bool StorePreamblesInMemory = true;
    /// Reuse even stale preambles, and rebuild them in the background.
    /// This improves latency at the cost of accuracy.
    bool AsyncPreambleBuilds = false;

    /// If true, ClangdServer builds a dynamic in-memory index for symbols in
    /// opened files and uses the index to augment code completion results.
    bool BuildDynamicSymbolIndex = false;
    /// Use a heavier and faster in-memory index implementation.
    bool HeavyweightDynamicSymbolIndex = true;
    /// If true, ClangdServer automatically indexes files in the current project
    /// on background threads. The index is stored in the project root.
    bool BackgroundIndex = false;

    /// If set, use this index to augment code completion results.
    SymbolIndex *StaticIndex = nullptr;

    /// If set, enable clang-tidy in clangd and use to it get clang-tidy
    /// configurations for a particular file.
    /// Clangd supports only a small subset of ClangTidyOptions, these options
    /// (Checks, CheckOptions) are about which clang-tidy checks will be
    /// enabled.
    ClangTidyOptionsBuilder GetClangTidyOptions;

    /// If true, turn on the `-frecovery-ast` clang flag.
    bool BuildRecoveryAST = false;

    /// If true, turn on the `-frecovery-ast-type` clang flag.
    bool PreserveRecoveryASTType = false;

    /// Clangd's workspace root. Relevant for "workspace" operations not bound
    /// to a particular file.
    /// FIXME: If not set, should use the current working directory.
    llvm::Optional<std::string> WorkspaceRoot;

    /// The resource directory is used to find internal headers, overriding
    /// defaults and -resource-dir compiler flag).
    /// If None, ClangdServer calls CompilerInvocation::GetResourcePath() to
    /// obtain the standard resource directory.
    llvm::Optional<std::string> ResourceDir = llvm::None;

    /// Time to wait after a new file version before computing diagnostics.
    DebouncePolicy UpdateDebounce = DebouncePolicy{
        /*Min=*/std::chrono::milliseconds(50),
        /*Max=*/std::chrono::milliseconds(500),
        /*RebuildRatio=*/1,
    };

    bool SuggestMissingIncludes = false;

    /// Clangd will execute compiler drivers matching one of these globs to
    /// fetch system include path.
    std::vector<std::string> QueryDriverGlobs;

    /// Enable notification-based semantic highlighting.
    bool TheiaSemanticHighlighting = false;

    /// Returns true if the tweak should be enabled.
    std::function<bool(const Tweak &)> TweakFilter = [](const Tweak &T) {
      return !T.hidden(); // only enable non-hidden tweaks.
    };

    explicit operator TUScheduler::Options() const;
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
  ClangdServer(const GlobalCompilationDatabase &CDB,
               const FileSystemProvider &FSProvider, const Options &Opts,
               Callbacks *Callbacks = nullptr);

  /// Add a \p File to the list of tracked C++ files or update the contents if
  /// \p File is already tracked. Also schedules parsing of the AST for it on a
  /// separate thread. When the parsing is complete, DiagConsumer passed in
  /// constructor will receive onDiagnosticsReady callback.
  /// Version identifies this snapshot and is propagated to ASTs, preambles,
  /// diagnostics etc built from it.
  void addDocument(PathRef File, StringRef Contents,
                   llvm::StringRef Version = "null",
                   WantDiagnostics WD = WantDiagnostics::Auto,
                   bool ForceRebuild = false);

  /// Remove \p File from list of tracked files, schedule a request to free
  /// resources associated with it. Pending diagnostics for closed files may not
  /// be delivered, even if requested with WantDiags::Auto or WantDiags::Yes.
  /// An empty set of diagnostics will be delivered, with Version = "".
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

  /// Find declaration/definition locations of symbol at a specified position.
  void locateSymbolAt(PathRef File, Position Pos,
                      Callback<std::vector<LocatedSymbol>> CB);

  /// Switch to a corresponding source file when given a header file, and vice
  /// versa.
  void switchSourceHeader(PathRef Path,
                          Callback<llvm::Optional<clangd::Path>> CB);

  /// Get document highlights for a given position.
  void findDocumentHighlights(PathRef File, Position Pos,
                              Callback<std::vector<DocumentHighlight>> CB);

  /// Get code hover for a given position.
  void findHover(PathRef File, Position Pos,
                 Callback<llvm::Optional<HoverInfo>> CB);

  /// Get information about type hierarchy for a given position.
  void typeHierarchy(PathRef File, Position Pos, int Resolve,
                     TypeHierarchyDirection Direction,
                     Callback<llvm::Optional<TypeHierarchyItem>> CB);

  /// Resolve type hierarchy item in the given direction.
  void resolveTypeHierarchy(TypeHierarchyItem Item, int Resolve,
                            TypeHierarchyDirection Direction,
                            Callback<llvm::Optional<TypeHierarchyItem>> CB);

  /// Retrieve the top symbols from the workspace matching a query.
  void workspaceSymbols(StringRef Query, int Limit,
                        Callback<std::vector<SymbolInformation>> CB);

  /// Retrieve the symbols within the specified file.
  void documentSymbols(StringRef File,
                       Callback<std::vector<DocumentSymbol>> CB);

  /// Retrieve locations for symbol references.
  void findReferences(PathRef File, Position Pos, uint32_t Limit,
                      Callback<ReferencesResult> CB);

  /// Run formatting for \p Rng inside \p File with content \p Code.
  llvm::Expected<tooling::Replacements> formatRange(StringRef Code,
                                                    PathRef File, Range Rng);

  /// Run formatting for the whole \p File with content \p Code.
  llvm::Expected<tooling::Replacements> formatFile(StringRef Code,
                                                   PathRef File);

  /// Run formatting after \p TriggerText was typed at \p Pos in \p File with
  /// content \p Code.
  llvm::Expected<std::vector<TextEdit>> formatOnType(StringRef Code,
                                                     PathRef File, Position Pos,
                                                     StringRef TriggerText);

  /// Test the validity of a rename operation.
  void prepareRename(PathRef File, Position Pos,
                     const RenameOptions &RenameOpts,
                     Callback<llvm::Optional<Range>> CB);

  /// Rename all occurrences of the symbol at the \p Pos in \p File to
  /// \p NewName.
  /// If WantFormat is false, the final TextEdit will be not formatted,
  /// embedders could use this method to get all occurrences of the symbol (e.g.
  /// highlighting them in prepare stage).
  void rename(PathRef File, Position Pos, llvm::StringRef NewName,
              const RenameOptions &Opts, Callback<FileEdits> CB);

  struct TweakRef {
    std::string ID;    /// ID to pass for applyTweak.
    std::string Title; /// A single-line message to show in the UI.
    Tweak::Intent Intent;
  };
  /// Enumerate the code tweaks available to the user at a specified point.
  void enumerateTweaks(PathRef File, Range Sel,
                       Callback<std::vector<TweakRef>> CB);

  /// Apply the code tweak with a specified \p ID.
  void applyTweak(PathRef File, Range Sel, StringRef ID,
                  Callback<Tweak::Effect> CB);

  /// Only for testing purposes.
  /// Waits until all requests to worker thread are finished and dumps AST for
  /// \p File. \p File must be in the list of added documents.
  void dumpAST(PathRef File, llvm::unique_function<void(std::string)> Callback);
  /// Called when an event occurs for a watched file in the workspace.
  void onFileEvent(const DidChangeWatchedFilesParams &Params);

  /// Get symbol info for given position.
  /// Clangd extension - not part of official LSP.
  void symbolInfo(PathRef File, Position Pos,
                  Callback<std::vector<SymbolDetails>> CB);

  /// Get semantic ranges around a specified position in a file.
  void semanticRanges(PathRef File, const std::vector<Position> &Pos,
                      Callback<std::vector<SelectionRange>> CB);

  /// Get all document links in a file.
  void documentLinks(PathRef File, Callback<std::vector<DocumentLink>> CB);

  void semanticHighlights(PathRef File,
                          Callback<std::vector<HighlightingToken>>);

  /// Returns estimated memory usage and other statistics for each of the
  /// currently open files.
  /// Overall memory usage of clangd may be significantly more than reported
  /// here, as this metric does not account (at least) for:
  ///   - memory occupied by static and dynamic index,
  ///   - memory required for in-flight requests,
  /// FIXME: those metrics might be useful too, we should add them.
  llvm::StringMap<TUScheduler::FileStats> fileStats() const;

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

  const FileSystemProvider &FSProvider;

  Path ResourceDir;
  // The index used to look up symbols. This could be:
  //   - null (all index functionality is optional)
  //   - the dynamic index owned by ClangdServer (DynamicIdx)
  //   - the static index passed to the constructor
  //   - a merged view of a static and dynamic index (MergedIndex)
  const SymbolIndex *Index = nullptr;
  // If present, an index of symbols in open files. Read via *Index.
  std::unique_ptr<FileIndex> DynamicIdx;
  // If present, the new "auto-index" maintained in background threads.
  std::unique_ptr<BackgroundIndex> BackgroundIdx;
  // Storage for merged views of the various indexes.
  std::vector<std::unique_ptr<SymbolIndex>> MergedIdx;

  // When set, provides clang-tidy options for a specific file.
  ClangTidyOptionsBuilder GetClangTidyOptions;

  // If this is true, suggest include insertion fixes for diagnostic errors that
  // can be caused by missing includes (e.g. member access in incomplete type).
  bool SuggestMissingIncludes = false;

  // If true, preserve expressions in AST for broken code.
  bool BuildRecoveryAST = false;
  // If true, preserve the type for recovery AST.
  bool PreserveRecoveryASTType = false;

  std::function<bool(const Tweak &)> TweakFilter;

  // GUARDED_BY(CachedCompletionFuzzyFindRequestMutex)
  llvm::StringMap<llvm::Optional<FuzzyFindRequest>>
      CachedCompletionFuzzyFindRequestByFile;
  mutable std::mutex CachedCompletionFuzzyFindRequestMutex;

  llvm::Optional<std::string> WorkspaceRoot;
  // WorkScheduler has to be the last member, because its destructor has to be
  // called before all other members to stop the worker thread that references
  // ClangdServer.
  TUScheduler WorkScheduler;
};

} // namespace clangd
} // namespace clang

#endif
