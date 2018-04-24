//===--- ClangdLSPServer.h - LSP server --------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDLSPSERVER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDLSPSERVER_H

#include "ClangdServer.h"
#include "DraftStore.h"
#include "FindSymbols.h"
#include "GlobalCompilationDatabase.h"
#include "Path.h"
#include "Protocol.h"
#include "ProtocolHandlers.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/Optional.h"

namespace clang {
namespace clangd {

class JSONOutput;
class SymbolIndex;

/// This class provides implementation of an LSP server, glueing the JSON
/// dispatch and ClangdServer together.
class ClangdLSPServer : private DiagnosticsConsumer, private ProtocolCallbacks {
public:
  /// If \p CompileCommandsDir has a value, compile_commands.json will be
  /// loaded only from \p CompileCommandsDir. Otherwise, clangd will look
  /// for compile_commands.json in all parent directories of each file.
  ClangdLSPServer(JSONOutput &Out, const clangd::CodeCompleteOptions &CCOpts,
                  llvm::Optional<Path> CompileCommandsDir,
                  const ClangdServer::Options &Opts);

  /// Run LSP server loop, receiving input for it from \p In. \p In must be
  /// opened in binary mode. Output will be written using Out variable passed to
  /// class constructor. This method must not be executed more than once for
  /// each instance of ClangdLSPServer.
  ///
  /// \return Wether we received a 'shutdown' request before an 'exit' request
  bool run(std::istream &In,
           JSONStreamStyle InputStyle = JSONStreamStyle::Standard);

private:
  // Implement DiagnosticsConsumer.
  void onDiagnosticsReady(PathRef File, std::vector<Diag> Diagnostics) override;

  // Implement ProtocolCallbacks.
  void onInitialize(InitializeParams &Params) override;
  void onShutdown(ShutdownParams &Params) override;
  void onExit(ExitParams &Params) override;
  void onDocumentDidOpen(DidOpenTextDocumentParams &Params) override;
  void onDocumentDidChange(DidChangeTextDocumentParams &Params) override;
  void onDocumentDidClose(DidCloseTextDocumentParams &Params) override;
  void
  onDocumentOnTypeFormatting(DocumentOnTypeFormattingParams &Params) override;
  void
  onDocumentRangeFormatting(DocumentRangeFormattingParams &Params) override;
  void onDocumentFormatting(DocumentFormattingParams &Params) override;
  void onCodeAction(CodeActionParams &Params) override;
  void onCompletion(TextDocumentPositionParams &Params) override;
  void onSignatureHelp(TextDocumentPositionParams &Params) override;
  void onGoToDefinition(TextDocumentPositionParams &Params) override;
  void onSwitchSourceHeader(TextDocumentIdentifier &Params) override;
  void onDocumentHighlight(TextDocumentPositionParams &Params) override;
  void onFileEvent(DidChangeWatchedFilesParams &Params) override;
  void onCommand(ExecuteCommandParams &Params) override;
  void onWorkspaceSymbol(WorkspaceSymbolParams &Params) override;
  void onRename(RenameParams &Parames) override;
  void onHover(TextDocumentPositionParams &Params) override;
  void onChangeConfiguration(DidChangeConfigurationParams &Params) override;

  std::vector<Fix> getFixes(StringRef File, const clangd::Diagnostic &D);

  /// Forces a reparse of all currently opened files.  As a result, this method
  /// may be very expensive.  This method is normally called when the
  /// compilation database is changed.
  void reparseOpenedFiles();

  JSONOutput &Out;
  /// Used to indicate that the 'shutdown' request was received from the
  /// Language Server client.
  bool ShutdownRequestReceived = false;

  /// Used to indicate that the 'exit' notification was received from the
  /// Language Server client.
  /// It's used to break out of the LSP parsing loop.
  bool IsDone = false;

  std::mutex FixItsMutex;
  typedef std::map<clangd::Diagnostic, std::vector<Fix>, LSPDiagnosticCompare>
      DiagnosticToReplacementMap;
  /// Caches FixIts per file and diagnostics
  llvm::StringMap<DiagnosticToReplacementMap> FixItsMap;

  // Various ClangdServer parameters go here. It's important they're created
  // before ClangdServer.
  DirectoryBasedGlobalCompilationDatabase CDB;
  RealFileSystemProvider FSProvider;
  /// Options used for code completion
  clangd::CodeCompleteOptions CCOpts;
  /// The supported kinds of the client.
  SymbolKindBitset SupportedSymbolKinds;

  // Store of the current versions of the open documents.
  DraftStore DraftMgr;

  // Server must be the last member of the class to allow its destructor to exit
  // the worker thread that may otherwise run an async callback on partially
  // destructed instance of ClangdLSPServer.
  ClangdServer Server;
};

} // namespace clangd
} // namespace clang

#endif
