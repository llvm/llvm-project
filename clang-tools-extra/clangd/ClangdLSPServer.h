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
#include "Path.h"
#include "Protocol.h"
#include "Transport.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/Optional.h"
#include <memory>

namespace clang {
namespace clangd {

class SymbolIndex;

/// This class exposes ClangdServer's capabilities via Language Server Protocol.
///
/// MessageHandler binds the implemented LSP methods (e.g. onInitialize) to
/// corresponding JSON-RPC methods ("initialize").
/// The server also supports $/cancelRequest (MessageHandler provides this).
class ClangdLSPServer : private DiagnosticsConsumer {
public:
  /// If \p CompileCommandsDir has a value, compile_commands.json will be
  /// loaded only from \p CompileCommandsDir. Otherwise, clangd will look
  /// for compile_commands.json in all parent directories of each file.
  /// If UseDirBasedCDB is false, compile commands are not read from disk.
  // FIXME: Clean up signature around CDBs.
  ClangdLSPServer(Transport &Transp, const FileSystemProvider &FSProvider,
                  const clangd::CodeCompleteOptions &CCOpts,
                  llvm::Optional<Path> CompileCommandsDir, bool UseDirBasedCDB,
                  llvm::Optional<OffsetEncoding> ForcedOffsetEncoding,
                  const ClangdServer::Options &Opts);
  ~ClangdLSPServer();

  /// Run LSP server loop, communicating with the Transport provided in the
  /// constructor. This method must not be executed more than once.
  ///
  /// \return Whether we shut down cleanly with a 'shutdown' -> 'exit' sequence.
  bool run();

private:
  // Implement DiagnosticsConsumer.
  void onDiagnosticsReady(PathRef File, std::vector<Diag> Diagnostics) override;
  void onFileUpdated(PathRef File, const TUStatus &Status) override;

  // LSP methods. Notifications have signature void(const Params&).
  // Calls have signature void(const Params&, Callback<Response>).
  void onInitialize(const InitializeParams &, Callback<llvm::json::Value>);
  void onShutdown(const ShutdownParams &, Callback<std::nullptr_t>);
  void onSync(const NoParams &, Callback<std::nullptr_t>);
  void onDocumentDidOpen(const DidOpenTextDocumentParams &);
  void onDocumentDidChange(const DidChangeTextDocumentParams &);
  void onDocumentDidClose(const DidCloseTextDocumentParams &);
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
                            Callback<llvm::Optional<std::string>>);
  void onDocumentHighlight(const TextDocumentPositionParams &,
                           Callback<std::vector<DocumentHighlight>>);
  void onFileEvent(const DidChangeWatchedFilesParams &);
  void onCommand(const ExecuteCommandParams &, Callback<llvm::json::Value>);
  void onWorkspaceSymbol(const WorkspaceSymbolParams &,
                         Callback<std::vector<SymbolInformation>>);
  void onRename(const RenameParams &, Callback<WorkspaceEdit>);
  void onHover(const TextDocumentPositionParams &,
               Callback<llvm::Optional<Hover>>);
  void onTypeHierarchy(const TypeHierarchyParams &,
                       Callback<llvm::Optional<TypeHierarchyItem>>);
  void onChangeConfiguration(const DidChangeConfigurationParams &);
  void onSymbolInfo(const TextDocumentPositionParams &,
                    Callback<std::vector<SymbolDetails>>);

  std::vector<Fix> getFixes(StringRef File, const clangd::Diagnostic &D);

  /// Checks if completion request should be ignored. We need this due to the
  /// limitation of the LSP. Per LSP, a client sends requests for all "trigger
  /// character" we specify, but for '>' and ':' we need to check they actually
  /// produce '->' and '::', respectively.
  bool shouldRunCompletion(const CompletionParams &Params) const;

  /// Forces a reparse of all currently opened files.  As a result, this method
  /// may be very expensive.  This method is normally called when the
  /// compilation database is changed.
  void reparseOpenedFiles();
  void applyConfiguration(const ConfigurationSettings &Settings);

  /// Sends a "publishDiagnostics" notification to the LSP client.
  void publishDiagnostics(const URIForFile &File,
                          std::vector<clangd::Diagnostic> Diagnostics);

  /// Used to indicate that the 'shutdown' request was received from the
  /// Language Server client.
  bool ShutdownRequestReceived = false;

  std::mutex FixItsMutex;
  typedef std::map<clangd::Diagnostic, std::vector<Fix>, LSPDiagnosticCompare>
      DiagnosticToReplacementMap;
  /// Caches FixIts per file and diagnostics
  llvm::StringMap<DiagnosticToReplacementMap> FixItsMap;

  // Most code should not deal with Transport directly.
  // MessageHandler deals with incoming messages, use call() etc for outgoing.
  clangd::Transport &Transp;
  class MessageHandler;
  std::unique_ptr<MessageHandler> MsgHandler;
  std::atomic<int> NextCallID = {0};
  std::mutex TranspWriter;
  void call(StringRef Method, llvm::json::Value Params);
  void notify(StringRef Method, llvm::json::Value Params);

  const FileSystemProvider &FSProvider;
  /// Options used for code completion
  clangd::CodeCompleteOptions CCOpts;
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
  // Store of the current versions of the open documents.
  DraftStore DraftMgr;

  // The CDB is created by the "initialize" LSP method.
  bool UseDirBasedCDB;                     // FIXME: make this a capability.
  llvm::Optional<Path> CompileCommandsDir; // FIXME: merge with capability?
  std::unique_ptr<GlobalCompilationDatabase> BaseCDB;
  // CDB is BaseCDB plus any comands overridden via LSP extensions.
  llvm::Optional<OverlayCDB> CDB;
  // The ClangdServer is created by the "initialize" LSP method.
  // It is destroyed before run() returns, to ensure worker threads exit.
  ClangdServer::Options ClangdServerOpts;
  llvm::Optional<ClangdServer> Server;
  llvm::Optional<OffsetEncoding> NegotiatedOffsetEncoding;
};
} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDLSPSERVER_H
