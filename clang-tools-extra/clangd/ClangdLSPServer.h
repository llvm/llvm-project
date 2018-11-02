//===--- ClangdLSPServer.h - LSP server --------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDLSPSERVER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDLSPSERVER_H

#include "ClangdServer.h"
#include "DraftStore.h"
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
  ClangdLSPServer(Transport &Transp, const clangd::CodeCompleteOptions &CCOpts,
                  llvm::Optional<Path> CompileCommandsDir, bool UseDirBasedCDB,
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

  // LSP methods. Notifications have signature void(const Params&).
  // Calls have signature void(const Params&, Callback<Response>).
  void onInitialize(const InitializeParams &, Callback<llvm::json::Value>);
  void onShutdown(const ShutdownParams &, Callback<std::nullptr_t>);
  void onDocumentDidOpen(const DidOpenTextDocumentParams &);
  void onDocumentDidChange(const DidChangeTextDocumentParams &);
  void onDocumentDidClose(const DidCloseTextDocumentParams &);
  void onDocumentOnTypeFormatting(const DocumentOnTypeFormattingParams &,
                                  Callback<std::vector<TextEdit>>);
  void onDocumentRangeFormatting(const DocumentRangeFormattingParams &,
                                 Callback<std::vector<TextEdit>>);
  void onDocumentFormatting(const DocumentFormattingParams &,
                            Callback<std::vector<TextEdit>>);
  void onDocumentSymbol(const DocumentSymbolParams &,
                        Callback<std::vector<SymbolInformation>>);
  void onCodeAction(const CodeActionParams &, Callback<llvm::json::Value>);
  void onCompletion(const TextDocumentPositionParams &,
                    Callback<CompletionList>);
  void onSignatureHelp(const TextDocumentPositionParams &,
                       Callback<SignatureHelp>);
  void onGoToDefinition(const TextDocumentPositionParams &,
                        Callback<std::vector<Location>>);
  void onReference(const ReferenceParams &, Callback<std::vector<Location>>);
  void onSwitchSourceHeader(const TextDocumentIdentifier &,
                            Callback<std::string>);
  void onDocumentHighlight(const TextDocumentPositionParams &,
                           Callback<std::vector<DocumentHighlight>>);
  void onFileEvent(const DidChangeWatchedFilesParams &);
  void onCommand(const ExecuteCommandParams &, Callback<llvm::json::Value>);
  void onWorkspaceSymbol(const WorkspaceSymbolParams &,
                         Callback<std::vector<SymbolInformation>>);
  void onRename(const RenameParams &, Callback<WorkspaceEdit>);
  void onHover(const TextDocumentPositionParams &,
               Callback<llvm::Optional<Hover>>);
  void onChangeConfiguration(const DidChangeConfigurationParams &);

  std::vector<Fix> getFixes(StringRef File, const clangd::Diagnostic &D);

  /// Forces a reparse of all currently opened files.  As a result, this method
  /// may be very expensive.  This method is normally called when the
  /// compilation database is changed.
  void reparseOpenedFiles();
  void applyConfiguration(const ConfigurationSettings &Settings);

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

  RealFileSystemProvider FSProvider;
  /// Options used for code completion
  clangd::CodeCompleteOptions CCOpts;
  /// Options used for diagnostics.
  ClangdDiagnosticOptions DiagOpts;
  /// The supported kinds of the client.
  SymbolKindBitset SupportedSymbolKinds;
  /// The supported completion item kinds of the client.
  CompletionItemKindBitset SupportedCompletionItemKinds;
  // Whether the client supports CodeAction response objects.
  bool SupportsCodeAction = false;

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
};
} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDLSPSERVER_H
