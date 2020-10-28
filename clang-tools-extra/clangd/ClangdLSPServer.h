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
#include "DraftStore.h"
#include "Features.inc"
#include "FindSymbols.h"
#include "GlobalCompilationDatabase.h"
#include "Protocol.h"
#include "Transport.h"
#include "support/Context.h"
#include "support/Path.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/JSON.h"
#include <memory>

namespace clang {
namespace clangd {

class SymbolIndex;

/// This class exposes ClangdServer's capabilities via Language Server Protocol.
///
/// MessageHandler binds the implemented LSP methods (e.g. onInitialize) to
/// corresponding JSON-RPC methods ("initialize").
/// The server also supports $/cancelRequest (MessageHandler provides this).
class ClangdLSPServer : private ClangdServer::Callbacks {
public:
  /// If \p CompileCommandsDir has a value, compile_commands.json will be
  /// loaded only from \p CompileCommandsDir. Otherwise, clangd will look
  /// for compile_commands.json in all parent directories of each file.
  /// If UseDirBasedCDB is false, compile commands are not read from disk.
  // FIXME: Clean up signature around CDBs.
  ClangdLSPServer(Transport &Transp, const ThreadsafeFS &TFS,
                  const clangd::CodeCompleteOptions &CCOpts,
                  const clangd::RenameOptions &RenameOpts,
                  llvm::Optional<Path> CompileCommandsDir, bool UseDirBasedCDB,
                  llvm::Optional<OffsetEncoding> ForcedOffsetEncoding,
                  const ClangdServer::Options &Opts);
  /// The destructor blocks on any outstanding background tasks.
  ~ClangdLSPServer();

  /// Run LSP server loop, communicating with the Transport provided in the
  /// constructor. This method must not be executed more than once.
  ///
  /// \return Whether we shut down cleanly with a 'shutdown' -> 'exit' sequence.
  bool run();

private:
  // Implement ClangdServer::Callbacks.
  void onDiagnosticsReady(PathRef File, llvm::StringRef Version,
                          std::vector<Diag> Diagnostics) override;
  void onFileUpdated(PathRef File, const TUStatus &Status) override;
  void
  onHighlightingsReady(PathRef File, llvm::StringRef Version,
                       std::vector<HighlightingToken> Highlightings) override;
  void onBackgroundIndexProgress(const BackgroundQueue::Stats &Stats) override;

  // LSP methods. Notifications have signature void(const Params&).
  // Calls have signature void(const Params&, Callback<Response>).
  void onInitialize(const InitializeParams &, Callback<llvm::json::Value>);
  void onInitialized(const InitializedParams &);
  void onShutdown(const ShutdownParams &, Callback<std::nullptr_t>);
  void onSync(const NoParams &, Callback<std::nullptr_t>);
  void onDocumentDidOpen(const DidOpenTextDocumentParams &);
  void onDocumentDidChange(const DidChangeTextDocumentParams &);
  void onDocumentDidClose(const DidCloseTextDocumentParams &);
  void onDocumentDidSave(const DidSaveTextDocumentParams &);
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
  void onReference(const ReferenceParams &, Callback<std::vector<Location>>);
  void onSwitchSourceHeader(const TextDocumentIdentifier &,
                            Callback<llvm::Optional<URIForFile>>);
  void onDocumentHighlight(const TextDocumentPositionParams &,
                           Callback<std::vector<DocumentHighlight>>);
  void onFileEvent(const DidChangeWatchedFilesParams &);
  void onCommand(const ExecuteCommandParams &, Callback<llvm::json::Value>);
  void onWorkspaceSymbol(const WorkspaceSymbolParams &,
                         Callback<std::vector<SymbolInformation>>);
  void onPrepareRename(const TextDocumentPositionParams &,
                       Callback<llvm::Optional<Range>>);
  void onRename(const RenameParams &, Callback<WorkspaceEdit>);
  void onHover(const TextDocumentPositionParams &,
               Callback<llvm::Optional<Hover>>);
  void onTypeHierarchy(const TypeHierarchyParams &,
                       Callback<llvm::Optional<TypeHierarchyItem>>);
  void onResolveTypeHierarchy(const ResolveTypeHierarchyItemParams &,
                              Callback<llvm::Optional<TypeHierarchyItem>>);
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

  std::vector<Fix> getFixes(StringRef File, const clangd::Diagnostic &D);

  /// Checks if completion request should be ignored. We need this due to the
  /// limitation of the LSP. Per LSP, a client sends requests for all "trigger
  /// character" we specify, but for '>' and ':' we need to check they actually
  /// produce '->' and '::', respectively.
  bool shouldRunCompletion(const CompletionParams &Params) const;

  /// Requests a reparse of currently opened files using their latest source.
  /// This will typically only rebuild if something other than the source has
  /// changed (e.g. the CDB yields different flags, or files included in the
  /// preamble have been modified).
  void reparseOpenFilesIfNeeded(
      llvm::function_ref<bool(llvm::StringRef File)> Filter);
  void applyConfiguration(const ConfigurationSettings &Settings);

  /// Sends a "publishSemanticHighlighting" notification to the LSP client.
  void
  publishTheiaSemanticHighlighting(const TheiaSemanticHighlightingParams &);

  /// Sends a "publishDiagnostics" notification to the LSP client.
  void publishDiagnostics(const PublishDiagnosticsParams &);

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

  std::mutex FixItsMutex;
  typedef std::map<clangd::Diagnostic, std::vector<Fix>, LSPDiagnosticCompare>
      DiagnosticToReplacementMap;
  /// Caches FixIts per file and diagnostics
  llvm::StringMap<DiagnosticToReplacementMap> FixItsMap;
  std::mutex HighlightingsMutex;
  llvm::StringMap<std::vector<HighlightingToken>> FileToHighlightings;
  // Last semantic-tokens response, for incremental requests.
  std::mutex SemanticTokensMutex;
  llvm::StringMap<SemanticTokens> LastSemanticTokens;

  // Most code should not deal with Transport directly.
  // MessageHandler deals with incoming messages, use call() etc for outgoing.
  clangd::Transport &Transp;
  class MessageHandler;
  std::unique_ptr<MessageHandler> MsgHandler;
  std::mutex TranspWriter;

  template <typename Response>
  void call(StringRef Method, llvm::json::Value Params, Callback<Response> CB) {
    // Wrap the callback with LSP conversion and error-handling.
    auto HandleReply =
        [CB = std::move(CB), Ctx = Context::current().clone()](
            llvm::Expected<llvm::json::Value> RawResponse) mutable {
          Response Rsp;
          if (!RawResponse) {
            CB(RawResponse.takeError());
          } else if (fromJSON(*RawResponse, Rsp)) {
            CB(std::move(Rsp));
          } else {
            elog("Failed to decode {0} response", *RawResponse);
            CB(llvm::make_error<LSPError>("failed to decode response",
                                          ErrorCode::InvalidParams));
          }
        };
    callRaw(Method, std::move(Params), std::move(HandleReply));
  }
  void callRaw(StringRef Method, llvm::json::Value Params,
               Callback<llvm::json::Value> CB);
  void notify(StringRef Method, llvm::json::Value Params);
  template <typename T> void progress(const llvm::json::Value &Token, T Value) {
    ProgressParams<T> Params;
    Params.token = Token;
    Params.value = std::move(Value);
    notify("$/progress", Params);
  }

  const ThreadsafeFS &TFS;
  /// Options used for code completion
  clangd::CodeCompleteOptions CCOpts;
  /// Options used for rename.
  clangd::RenameOptions RenameOpts;
  /// Options used for diagnostics.
  ClangdDiagnosticOptions DiagOpts;
  /// The supported kinds of the client.
  SymbolKindBitset SupportedSymbolKinds;
  /// The supported completion item kinds of the client.
  CompletionItemKindBitset SupportedCompletionItemKinds;
  /// Whether the client supports CodeAction response objects.
  bool SupportsCodeAction = false;
  /// From capabilities of textDocument/documentSymbol.
  bool SupportsHierarchicalDocumentSymbol = false;
  /// Whether the client supports showing file status.
  bool SupportFileStatus = false;
  /// Which kind of markup should we use in textDocument/hover responses.
  MarkupKind HoverContentFormat = MarkupKind::PlainText;
  /// Whether the client supports offsets for parameter info labels.
  bool SupportsOffsetsInSignatureHelp = false;
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
  // Store of the current versions of the open documents.
  DraftStore DraftMgr;

  // The CDB is created by the "initialize" LSP method.
  bool UseDirBasedCDB;                     // FIXME: make this a capability.
  llvm::Optional<Path> CompileCommandsDir; // FIXME: merge with capability?
  std::unique_ptr<GlobalCompilationDatabase> BaseCDB;
  // CDB is BaseCDB plus any commands overridden via LSP extensions.
  llvm::Optional<OverlayCDB> CDB;
  ClangdServer::Options ClangdServerOpts;
  llvm::Optional<OffsetEncoding> NegotiatedOffsetEncoding;
  // The ClangdServer is created by the "initialize" LSP method.
  llvm::Optional<ClangdServer> Server;
};
} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDLSPSERVER_H
