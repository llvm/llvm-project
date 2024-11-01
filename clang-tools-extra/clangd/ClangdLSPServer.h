//===--- ClangdLSPServer.h - LSP server --------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDLSPSERVER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDLSPSERVER_H

#include "ClangdServer.h"
#include "Diagnostics.h"
#include "GlobalCompilationDatabase.h"
#include "LSPBinder.h"
#include "Protocol.h"
#include "Transport.h"
#include "support/Context.h"
#include "support/MemoryTree.h"
#include "support/Path.h"
#include "support/Threading.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/JSON.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace clang {
namespace clangd {

/// This class exposes ClangdServer's capabilities via Language Server Protocol.
///
/// MessageHandler binds the implemented LSP methods (e.g. onInitialize) to
/// corresponding JSON-RPC methods ("initialize").
/// The server also supports $/cancelRequest (MessageHandler provides this).
class ClangdLSPServer : private ClangdServer::Callbacks,
                        private LSPBinder::RawOutgoing {
public:
  struct Options : ClangdServer::Options {
    /// Supplies configuration (overrides ClangdServer::ContextProvider).
    config::Provider *ConfigProvider = nullptr;
    /// Look for compilation databases, rather than using compile commands
    /// set via LSP (extensions) only.
    bool UseDirBasedCDB = true;
    /// The offset-encoding to use, or std::nullopt to negotiate it over LSP.
    std::optional<OffsetEncoding> Encoding;
    /// If set, periodically called to release memory.
    /// Consider malloc_trim(3)
    std::function<void()> MemoryCleanup = nullptr;

    /// Per-feature options. Generally ClangdServer lets these vary
    /// per-request, but LSP allows limited/no customizations.
    clangd::CodeCompleteOptions CodeComplete;
    MarkupKind SignatureHelpDocumentationFormat = MarkupKind::PlainText;
    clangd::RenameOptions Rename;
    /// Returns true if the tweak should be enabled.
    std::function<bool(const Tweak &)> TweakFilter = [](const Tweak &T) {
      return !T.hidden(); // only enable non-hidden tweaks.
    };

    /// Limit the number of references returned (0 means no limit).
    size_t ReferencesLimit = 0;
  };

  ClangdLSPServer(Transport &Transp, const ThreadsafeFS &TFS,
                  const ClangdLSPServer::Options &Opts);
  /// The destructor blocks on any outstanding background tasks.
  ~ClangdLSPServer();

  /// Run LSP server loop, communicating with the Transport provided in the
  /// constructor. This method must not be executed more than once.
  ///
  /// \return Whether we shut down cleanly with a 'shutdown' -> 'exit' sequence.
  bool run();

  /// Profiles resource-usage.
  void profile(MemoryTree &MT) const;

private:
  // Implement ClangdServer::Callbacks.
  void onDiagnosticsReady(PathRef File, llvm::StringRef Version,
                          llvm::ArrayRef<Diag> Diagnostics) override;
  void onFileUpdated(PathRef File, const TUStatus &Status) override;
  void onBackgroundIndexProgress(const BackgroundQueue::Stats &Stats) override;
  void onSemanticsMaybeChanged(PathRef File) override;
  void onInactiveRegionsReady(PathRef File,
                              std::vector<Range> InactiveRegions) override;

  // LSP methods. Notifications have signature void(const Params&).
  // Calls have signature void(const Params&, Callback<Response>).
  void onInitialize(const InitializeParams &, Callback<llvm::json::Value>);
  void onInitialized(const InitializedParams &);
  void onShutdown(const NoParams &, Callback<std::nullptr_t>);
  void onSync(const NoParams &, Callback<std::nullptr_t>);
  void onDocumentDidOpen(const DidOpenTextDocumentParams &);
  void onDocumentDidChange(const DidChangeTextDocumentParams &);
  void onDocumentDidClose(const DidCloseTextDocumentParams &);
  void onDocumentDidSave(const DidSaveTextDocumentParams &);
  void onAST(const ASTParams &, Callback<std::optional<ASTNode>>);
  void onDocumentOnTypeFormatting(const DocumentOnTypeFormattingParams &,
                                  Callback<std::vector<TextEdit>>);
  void onDocumentRangeFormatting(const DocumentRangeFormattingParams &,
                                 Callback<std::vector<TextEdit>>);
  void onDocumentFormatting(const DocumentFormattingParams &,
                            Callback<std::vector<TextEdit>>);
  // The results are serialized 'vector<DocumentSymbol>' if
  // SupportsHierarchicalDocumentSymbol is true and 'vector<SymbolInformation>'
  // otherwise.
  void onDocumentSymbol(const DocumentSymbolParams &,
                        Callback<llvm::json::Value>);
  void onFoldingRange(const FoldingRangeParams &,
                      Callback<std::vector<FoldingRange>>);
  void onCodeAction(const CodeActionParams &, Callback<llvm::json::Value>);
  void onCompletion(const CompletionParams &, Callback<CompletionList>);
  void onSignatureHelp(const TextDocumentPositionParams &,
                       Callback<SignatureHelp>);
  void onGoToDeclaration(const TextDocumentPositionParams &,
                         Callback<std::vector<Location>>);
  void onGoToDefinition(const TextDocumentPositionParams &,
                        Callback<std::vector<Location>>);
  void onGoToType(const TextDocumentPositionParams &,
                  Callback<std::vector<Location>>);
  void onGoToImplementation(const TextDocumentPositionParams &,
                            Callback<std::vector<Location>>);
  void onReference(const ReferenceParams &, Callback<std::vector<ReferenceLocation>>);
  void onSwitchSourceHeader(const TextDocumentIdentifier &,
                            Callback<std::optional<URIForFile>>);
  void onDocumentHighlight(const TextDocumentPositionParams &,
                           Callback<std::vector<DocumentHighlight>>);
  void onFileEvent(const DidChangeWatchedFilesParams &);
  void onWorkspaceSymbol(const WorkspaceSymbolParams &,
                         Callback<std::vector<SymbolInformation>>);
  void onPrepareRename(const TextDocumentPositionParams &,
                       Callback<std::optional<Range>>);
  void onRename(const RenameParams &, Callback<WorkspaceEdit>);
  void onHover(const TextDocumentPositionParams &,
               Callback<std::optional<Hover>>);
  void onPrepareTypeHierarchy(const TypeHierarchyPrepareParams &,
                              Callback<std::vector<TypeHierarchyItem>>);
  void onSuperTypes(const ResolveTypeHierarchyItemParams &,
                    Callback<std::optional<std::vector<TypeHierarchyItem>>>);
  void onSubTypes(const ResolveTypeHierarchyItemParams &,
                  Callback<std::vector<TypeHierarchyItem>>);
  void onTypeHierarchy(const TypeHierarchyPrepareParams &,
                       Callback<llvm::json::Value>);
  void onResolveTypeHierarchy(const ResolveTypeHierarchyItemParams &,
                              Callback<llvm::json::Value>);
  void onPrepareCallHierarchy(const CallHierarchyPrepareParams &,
                              Callback<std::vector<CallHierarchyItem>>);
  void onCallHierarchyIncomingCalls(
      const CallHierarchyIncomingCallsParams &,
      Callback<std::vector<CallHierarchyIncomingCall>>);
  void onClangdInlayHints(const InlayHintsParams &,
                          Callback<llvm::json::Value>);
  void onInlayHint(const InlayHintsParams &, Callback<std::vector<InlayHint>>);
  void onChangeConfiguration(const DidChangeConfigurationParams &);
  void onSymbolInfo(const TextDocumentPositionParams &,
                    Callback<std::vector<SymbolDetails>>);
  void onSelectionRange(const SelectionRangeParams &,
                        Callback<std::vector<SelectionRange>>);
  void onDocumentLink(const DocumentLinkParams &,
                      Callback<std::vector<DocumentLink>>);
  void onSemanticTokens(const SemanticTokensParams &, Callback<SemanticTokens>);
  void onSemanticTokensDelta(const SemanticTokensDeltaParams &,
                             Callback<SemanticTokensOrDelta>);
  /// This is a clangd extension. Provides a json tree representing memory usage
  /// hierarchy.
  void onMemoryUsage(const NoParams &, Callback<MemoryTree>);
  void onCommand(const ExecuteCommandParams &, Callback<llvm::json::Value>);

  /// Implement commands.
  void onCommandApplyEdit(const WorkspaceEdit &, Callback<llvm::json::Value>);
  void onCommandApplyTweak(const TweakArgs &, Callback<llvm::json::Value>);

  /// Outgoing LSP calls.
  LSPBinder::OutgoingMethod<ApplyWorkspaceEditParams,
                            ApplyWorkspaceEditResponse>
      ApplyWorkspaceEdit;
  LSPBinder::OutgoingNotification<ShowMessageParams> ShowMessage;
  LSPBinder::OutgoingNotification<PublishDiagnosticsParams> PublishDiagnostics;
  LSPBinder::OutgoingNotification<FileStatus> NotifyFileStatus;
  LSPBinder::OutgoingNotification<InactiveRegionsParams> PublishInactiveRegions;
  LSPBinder::OutgoingMethod<WorkDoneProgressCreateParams, std::nullptr_t>
      CreateWorkDoneProgress;
  LSPBinder::OutgoingNotification<ProgressParams<WorkDoneProgressBegin>>
      BeginWorkDoneProgress;
  LSPBinder::OutgoingNotification<ProgressParams<WorkDoneProgressReport>>
      ReportWorkDoneProgress;
  LSPBinder::OutgoingNotification<ProgressParams<WorkDoneProgressEnd>>
      EndWorkDoneProgress;
  LSPBinder::OutgoingMethod<NoParams, std::nullptr_t> SemanticTokensRefresh;

  void applyEdit(WorkspaceEdit WE, llvm::json::Value Success,
                 Callback<llvm::json::Value> Reply);

  void bindMethods(LSPBinder &, const ClientCapabilities &Caps);
  std::optional<ClangdServer::DiagRef> getDiagRef(StringRef File,
                                                  const clangd::Diagnostic &D);

  /// Checks if completion request should be ignored. We need this due to the
  /// limitation of the LSP. Per LSP, a client sends requests for all "trigger
  /// character" we specify, but for '>' and ':' we need to check they actually
  /// produce '->' and '::', respectively.
  bool shouldRunCompletion(const CompletionParams &Params) const;

  void applyConfiguration(const ConfigurationSettings &Settings);

  /// Runs profiling and exports memory usage metrics if tracing is enabled and
  /// profiling hasn't happened recently.
  void maybeExportMemoryProfile();
  PeriodicThrottler ShouldProfile;

  /// Run the MemoryCleanup callback if it's time.
  /// This method is thread safe.
  void maybeCleanupMemory();
  PeriodicThrottler ShouldCleanupMemory;

  /// Since initialization of CDBs and ClangdServer is done lazily, the
  /// following context captures the one used while creating ClangdLSPServer and
  /// passes it to above mentioned object instances to make sure they share the
  /// same state.
  Context BackgroundContext;

  /// Used to indicate that the 'shutdown' request was received from the
  /// Language Server client.
  bool ShutdownRequestReceived = false;

  /// Used to indicate the ClangdLSPServer is being destroyed.
  std::atomic<bool> IsBeingDestroyed = {false};

  // FIXME: The caching is a temporary solution to get corresponding clangd 
  // diagnostic from a LSP diagnostic.
  // Ideally, ClangdServer can generate an identifier for each diagnostic,
  // emit them via the LSP's data field (which was newly added in LSP 3.16).
  std::mutex DiagRefMutex;
  struct DiagKey {
    clangd::Range Rng;
    std::string Message;
    bool operator<(const DiagKey &Other) const {
      return std::tie(Rng, Message) < std::tie(Other.Rng, Other.Message);
    }
  };
  DiagKey toDiagKey(const clangd::Diagnostic &LSPDiag) {
    return {LSPDiag.range, LSPDiag.message};
  }
  /// A map from LSP diagnostic to clangd-naive diagnostic.
  typedef std::map<DiagKey, ClangdServer::DiagRef>
      DiagnosticToDiagRefMap;
  /// Caches the mapping LSP and clangd-naive diagnostics per file.
  llvm::StringMap<DiagnosticToDiagRefMap>
      DiagRefMap;

  // Last semantic-tokens response, for incremental requests.
  std::mutex SemanticTokensMutex;
  llvm::StringMap<SemanticTokens> LastSemanticTokens;

  // Most code should not deal with Transport, callMethod, notify directly.
  // Use LSPBinder to handle incoming and outgoing calls.
  clangd::Transport &Transp;
  class MessageHandler;
  std::unique_ptr<MessageHandler> MsgHandler;
  std::mutex TranspWriter;

  void callMethod(StringRef Method, llvm::json::Value Params,
                  Callback<llvm::json::Value> CB) override;
  void notify(StringRef Method, llvm::json::Value Params) override;

  LSPBinder::RawHandlers Handlers;

  const ThreadsafeFS &TFS;
  /// Options used for diagnostics.
  ClangdDiagnosticOptions DiagOpts;
  /// The supported kinds of the client.
  SymbolKindBitset SupportedSymbolKinds;
  /// The supported completion item kinds of the client.
  CompletionItemKindBitset SupportedCompletionItemKinds;
  // Whether the client supports CompletionItem.labelDetails.
  bool SupportsCompletionLabelDetails = false;
  /// Whether the client supports CodeAction response objects.
  bool SupportsCodeAction = false;
  /// From capabilities of textDocument/documentSymbol.
  bool SupportsHierarchicalDocumentSymbol = false;
  /// Whether the client supports showing file status.
  bool SupportFileStatus = false;
  /// Whether the client supports attaching a container string to references.
  bool SupportsReferenceContainer = false;
  /// Which kind of markup should we use in textDocument/hover responses.
  MarkupKind HoverContentFormat = MarkupKind::PlainText;
  /// Whether the client supports offsets for parameter info labels.
  bool SupportsOffsetsInSignatureHelp = false;
  /// Whether the client supports the versioned document changes.
  bool SupportsDocumentChanges = false;
  /// Whether the client supports change annotations on text edits.
  bool SupportsChangeAnnotation = false;

  std::mutex BackgroundIndexProgressMutex;
  enum class BackgroundIndexProgress {
    // Client doesn't support reporting progress. No transitions possible.
    Unsupported,
    // The queue is idle, and the client has no progress bar.
    // Can transition to Creating when we have some activity.
    Empty,
    // We've requested the client to create a progress bar.
    // Meanwhile, the state is buffered in PendingBackgroundIndexProgress.
    Creating,
    // The client has a progress bar, and we can send it updates immediately.
    Live,
  } BackgroundIndexProgressState = BackgroundIndexProgress::Unsupported;
  // The progress to send when the progress bar is created.
  // Only valid in state Creating.
  BackgroundQueue::Stats PendingBackgroundIndexProgress;
  /// LSP extension: skip WorkDoneProgressCreate, just send progress streams.
  bool BackgroundIndexSkipCreate = false;

  Options Opts;
  // The CDB is created by the "initialize" LSP method.
  std::unique_ptr<GlobalCompilationDatabase> BaseCDB;
  // CDB is BaseCDB plus any commands overridden via LSP extensions.
  std::optional<OverlayCDB> CDB;
  // The ClangdServer is created by the "initialize" LSP method.
  std::optional<ClangdServer> Server;
};
} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDLSPSERVER_H
