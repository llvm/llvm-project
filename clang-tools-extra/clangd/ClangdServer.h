//===--- ClangdServer.h - Main clangd server code ----------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDSERVER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDSERVER_H

#include "CodeComplete.h"
#include "ConfigProvider.h"
#include "Diagnostics.h"
#include "DraftStore.h"
#include "FeatureModule.h"
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
#include "support/Function.h"
#include "support/MemoryTree.h"
#include "support/Path.h"
#include "support/ThreadsafeFS.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace clang {
namespace clangd {
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
    /// These pushed diagnostics might correspond to an older version of the
    /// file, they do not interfere with "pull-based" ClangdServer::diagnostics.
    /// May be called concurrently for separate files, not for a single file.
    virtual void onDiagnosticsReady(PathRef File, llvm::StringRef Version,
                                    std::vector<Diag> Diagnostics) {}
    /// Called whenever the file status is updated.
    /// May be called concurrently for separate files, not for a single file.
    virtual void onFileUpdated(PathRef File, const TUStatus &Status) {}

    /// Called when background indexing tasks are enqueued/started/completed.
    /// Not called concurrently.
    virtual void
    onBackgroundIndexProgress(const BackgroundQueue::Stats &Stats) {}

    /// Called when the meaning of a source code may have changed without an
    /// edit. Usually clients assume that responses to requests are valid until
    /// they next edit the file. If they're invalidated at other times, we
    /// should tell the client. In particular, when an asynchronous preamble
    /// build finishes, we can provide more accurate semantic tokens, so we
    /// should tell the client to refresh.
    virtual void onSemanticsMaybeChanged(PathRef File) {}
  };
  /// Creates a context provider that loads and installs config.
  /// Errors in loading config are reported as diagnostics via Callbacks.
  /// (This is typically used as ClangdServer::Options::ContextProvider).
  static std::function<Context(PathRef)>
  createConfiguredContextProvider(const config::Provider *Provider,
                                  ClangdServer::Callbacks *);

  struct Options {
    /// To process requests asynchronously, ClangdServer spawns worker threads.
    /// If this is zero, no threads are spawned. All work is done on the calling
    /// thread, and callbacks are invoked before "async" functions return.
    unsigned AsyncThreadsCount = getDefaultAsyncThreadsCount();

    /// AST caching policy. The default is to keep up to 3 ASTs in memory.
    ASTRetentionPolicy RetentionPolicy;

    /// Cached preambles are potentially large. If false, store them on disk.
    bool StorePreamblesInMemory = true;

    /// This throttler controls which preambles may be built at a given time.
    clangd::PreambleThrottler *PreambleThrottler = nullptr;

    /// If true, ClangdServer builds a dynamic in-memory index for symbols in
    /// opened files and uses the index to augment code completion results.
    bool BuildDynamicSymbolIndex = false;
    /// If true, ClangdServer automatically indexes files in the current project
    /// on background threads. The index is stored in the project root.
    bool BackgroundIndex = false;
    llvm::ThreadPriority BackgroundIndexPriority = llvm::ThreadPriority::Low;

    /// If set, use this index to augment code completion results.
    SymbolIndex *StaticIndex = nullptr;

    /// If set, queried to derive a processing context for some work.
    /// Usually used to inject Config (see createConfiguredContextProvider).
    ///
    /// When the provider is called, the active context will be that inherited
    /// from the request (e.g. addDocument()), or from the ClangdServer
    /// constructor if there is no such request (e.g. background indexing).
    ///
    /// The path is an absolute path of the file being processed.
    /// If there is no particular file (e.g. project loading) then it is empty.
    std::function<Context(PathRef)> ContextProvider;

    /// The Options provider to use when running clang-tidy. If null, clang-tidy
    /// checks will be disabled.
    TidyProviderRef ClangTidyProvider;

    /// Clangd's workspace root. Relevant for "workspace" operations not bound
    /// to a particular file.
    /// FIXME: If not set, should use the current working directory.
    llvm::Optional<std::string> WorkspaceRoot;

    /// The resource directory is used to find internal headers, overriding
    /// defaults and -resource-dir compiler flag).
    /// If None, ClangdServer calls CompilerInvocation::GetResourcePath() to
    /// obtain the standard resource directory.
    llvm::Optional<std::string> ResourceDir = std::nullopt;

    /// Time to wait after a new file version before computing diagnostics.
    DebouncePolicy UpdateDebounce = DebouncePolicy{
        /*Min=*/std::chrono::milliseconds(50),
        /*Max=*/std::chrono::milliseconds(500),
        /*RebuildRatio=*/1,
    };

    /// Cancel certain requests if the file changes before they begin running.
    /// This is useful for "transient" actions like enumerateTweaks that were
    /// likely implicitly generated, and avoids redundant work if clients forget
    /// to cancel. Clients that always cancel stale requests should clear this.
    bool ImplicitCancellation = true;

    /// Clangd will execute compiler drivers matching one of these globs to
    /// fetch system include path.
    std::vector<std::string> QueryDriverGlobs;

    // Whether the client supports folding only complete lines.
    bool LineFoldingOnly = false;

    FeatureModuleSet *FeatureModules = nullptr;
    /// If true, use the dirty buffer contents when building Preambles.
    bool UseDirtyHeaders = false;

    // If true, parse emplace-like functions in the preamble.
    bool PreambleParseForwardingFunctions = false;

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
  ClangdServer(const GlobalCompilationDatabase &CDB, const ThreadsafeFS &TFS,
               const Options &Opts, Callbacks *Callbacks = nullptr);
  ~ClangdServer();

  /// Gets the installed feature module of a given type, if any.
  /// This exposes access the public interface of feature modules that have one.
  template <typename Mod> Mod *featureModule() {
    return FeatureModules ? FeatureModules->get<Mod>() : nullptr;
  }
  template <typename Mod> const Mod *featureModule() const {
    return FeatureModules ? FeatureModules->get<Mod>() : nullptr;
  }

  /// Add a \p File to the list of tracked C++ files or update the contents if
  /// \p File is already tracked. Also schedules parsing of the AST for it on a
  /// separate thread. When the parsing is complete, DiagConsumer passed in
  /// constructor will receive onDiagnosticsReady callback.
  /// Version identifies this snapshot and is propagated to ASTs, preambles,
  /// diagnostics etc built from it. If empty, a version number is generated.
  void addDocument(PathRef File, StringRef Contents,
                   llvm::StringRef Version = "null",
                   WantDiagnostics WD = WantDiagnostics::Auto,
                   bool ForceRebuild = false);

  /// Remove \p File from list of tracked files, schedule a request to free
  /// resources associated with it. Pending diagnostics for closed files may not
  /// be delivered, even if requested with WantDiags::Auto or WantDiags::Yes.
  /// An empty set of diagnostics will be delivered, with Version = "".
  void removeDocument(PathRef File);

  /// Requests a reparse of currently opened files using their latest source.
  /// This will typically only rebuild if something other than the source has
  /// changed (e.g. the CDB yields different flags, or files included in the
  /// preamble have been modified).
  void reparseOpenFilesIfNeeded(
      llvm::function_ref<bool(llvm::StringRef File)> Filter);

  /// Run code completion for \p File at \p Pos.
  ///
  /// This method should only be called for currently tracked files.
  void codeComplete(PathRef File, Position Pos,
                    const clangd::CodeCompleteOptions &Opts,
                    Callback<CodeCompleteResult> CB);

  /// Provide signature help for \p File at \p Pos.  This method should only be
  /// called for tracked files.
  void signatureHelp(PathRef File, Position Pos, MarkupKind DocumentationFormat,
                     Callback<SignatureHelp> CB);

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
                     Callback<std::vector<TypeHierarchyItem>> CB);
  /// Get direct parents of a type hierarchy item.
  void superTypes(const TypeHierarchyItem &Item,
                  Callback<llvm::Optional<std::vector<TypeHierarchyItem>>> CB);
  /// Get direct children of a type hierarchy item.
  void subTypes(const TypeHierarchyItem &Item,
                Callback<std::vector<TypeHierarchyItem>> CB);

  /// Resolve type hierarchy item in the given direction.
  void resolveTypeHierarchy(TypeHierarchyItem Item, int Resolve,
                            TypeHierarchyDirection Direction,
                            Callback<llvm::Optional<TypeHierarchyItem>> CB);

  /// Get information about call hierarchy for a given position.
  void prepareCallHierarchy(PathRef File, Position Pos,
                            Callback<std::vector<CallHierarchyItem>> CB);

  /// Resolve incoming calls for a given call hierarchy item.
  void incomingCalls(const CallHierarchyItem &Item,
                     Callback<std::vector<CallHierarchyIncomingCall>>);

  /// Resolve inlay hints for a given document.
  void inlayHints(PathRef File, llvm::Optional<Range> RestrictRange,
                  Callback<std::vector<InlayHint>>);

  /// Retrieve the top symbols from the workspace matching a query.
  void workspaceSymbols(StringRef Query, int Limit,
                        Callback<std::vector<SymbolInformation>> CB);

  /// Retrieve the symbols within the specified file.
  void documentSymbols(StringRef File,
                       Callback<std::vector<DocumentSymbol>> CB);

  /// Retrieve ranges that can be used to fold code within the specified file.
  void foldingRanges(StringRef File, Callback<std::vector<FoldingRange>> CB);

  /// Retrieve implementations for virtual method.
  void findImplementations(PathRef File, Position Pos,
                           Callback<std::vector<LocatedSymbol>> CB);

  /// Retrieve symbols for types referenced at \p Pos.
  void findType(PathRef File, Position Pos,
                Callback<std::vector<LocatedSymbol>> CB);

  /// Retrieve locations for symbol references.
  void findReferences(PathRef File, Position Pos, uint32_t Limit,
                      Callback<ReferencesResult> CB);

  /// Run formatting for the \p File with content \p Code.
  /// If \p Rng is non-null, formats only that region.
  void formatFile(PathRef File, llvm::Optional<Range> Rng,
                  Callback<tooling::Replacements> CB);

  /// Run formatting after \p TriggerText was typed at \p Pos in \p File with
  /// content \p Code.
  void formatOnType(PathRef File, Position Pos, StringRef TriggerText,
                    Callback<std::vector<TextEdit>> CB);

  /// Test the validity of a rename operation.
  ///
  /// If NewName is provided, it performs a name validation.
  void prepareRename(PathRef File, Position Pos,
                     llvm::Optional<std::string> NewName,
                     const RenameOptions &RenameOpts,
                     Callback<RenameResult> CB);

  /// Rename all occurrences of the symbol at the \p Pos in \p File to
  /// \p NewName.
  /// If WantFormat is false, the final TextEdit will be not formatted,
  /// embedders could use this method to get all occurrences of the symbol (e.g.
  /// highlighting them in prepare stage).
  void rename(PathRef File, Position Pos, llvm::StringRef NewName,
              const RenameOptions &Opts, Callback<RenameResult> CB);

  struct TweakRef {
    std::string ID;    /// ID to pass for applyTweak.
    std::string Title; /// A single-line message to show in the UI.
    llvm::StringLiteral Kind;
  };
  /// Enumerate the code tweaks available to the user at a specified point.
  /// Tweaks where Filter returns false will not be checked or included.
  void enumerateTweaks(PathRef File, Range Sel,
                       llvm::unique_function<bool(const Tweak &)> Filter,
                       Callback<std::vector<TweakRef>> CB);

  /// Apply the code tweak with a specified \p ID.
  void applyTweak(PathRef File, Range Sel, StringRef ID,
                  Callback<Tweak::Effect> CB);

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

  /// Describe the AST subtree for a piece of code.
  void getAST(PathRef File, llvm::Optional<Range> R,
              Callback<llvm::Optional<ASTNode>> CB);

  /// Runs an arbitrary action that has access to the AST of the specified file.
  /// The action will execute on one of ClangdServer's internal threads.
  /// The AST is only valid for the duration of the callback.
  /// As with other actions, the file must have been opened.
  void customAction(PathRef File, llvm::StringRef Name,
                    Callback<InputsAndAST> Action);

  /// Fetches diagnostics for current version of the \p File. This might fail if
  /// server is busy (building a preamble) and would require a long time to
  /// prepare diagnostics. If it fails, clients should wait for
  /// onSemanticsMaybeChanged and then retry.
  /// These 'pulled' diagnostics do not interfere with the diagnostics 'pushed'
  /// to Callbacks::onDiagnosticsReady, and clients may use either or both.
  void diagnostics(PathRef File, Callback<std::vector<Diag>> CB);

  /// Returns estimated memory usage and other statistics for each of the
  /// currently open files.
  /// Overall memory usage of clangd may be significantly more than reported
  /// here, as this metric does not account (at least) for:
  ///   - memory occupied by static and dynamic index,
  ///   - memory required for in-flight requests,
  /// FIXME: those metrics might be useful too, we should add them.
  llvm::StringMap<TUScheduler::FileStats> fileStats() const;

  /// Gets the contents of a currently tracked file. Returns nullptr if the file
  /// isn't being tracked.
  std::shared_ptr<const std::string> getDraft(PathRef File) const;

  // Blocks the main thread until the server is idle. Only for use in tests.
  // Returns false if the timeout expires.
  // FIXME: various subcomponents each get the full timeout, so it's more of
  // an order of magnitude than a hard deadline.
  [[nodiscard]] bool
  blockUntilIdleForTest(llvm::Optional<double> TimeoutSeconds = 10);

  /// Builds a nested representation of memory used by components.
  void profile(MemoryTree &MT) const;

private:
  FeatureModuleSet *FeatureModules;
  const GlobalCompilationDatabase &CDB;
  const ThreadsafeFS &getHeaderFS() const {
    return UseDirtyHeaders ? *DirtyFS : TFS;
  }
  const ThreadsafeFS &TFS;

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
  TidyProviderRef ClangTidyProvider;

  bool UseDirtyHeaders = false;

  // Whether the client supports folding only complete lines.
  bool LineFoldingOnly = false;

  bool PreambleParseForwardingFunctions = false;

  // GUARDED_BY(CachedCompletionFuzzyFindRequestMutex)
  llvm::StringMap<llvm::Optional<FuzzyFindRequest>>
      CachedCompletionFuzzyFindRequestByFile;
  mutable std::mutex CachedCompletionFuzzyFindRequestMutex;

  llvm::Optional<std::string> WorkspaceRoot;
  llvm::Optional<AsyncTaskRunner> IndexTasks; // for stdlib indexing.
  llvm::Optional<TUScheduler> WorkScheduler;
  // Invalidation policy used for actions that we assume are "transient".
  TUScheduler::ASTActionInvalidation Transient;

  // Store of the current versions of the open documents.
  // Only written from the main thread (despite being threadsafe).
  DraftStore DraftMgr;

  std::unique_ptr<ThreadsafeFS> DirtyFS;
};

} // namespace clangd
} // namespace clang

#endif
