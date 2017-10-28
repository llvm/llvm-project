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
#include "GlobalCompilationDatabase.h"
#include "Path.h"
#include "Protocol.h"
#include "ProtocolHandlers.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/Optional.h"

namespace clang {
namespace clangd {

class JSONOutput;

/// This class provides implementation of an LSP server, glueing the JSON
/// dispatch and ClangdServer together.
class ClangdLSPServer : private DiagnosticsConsumer, private ProtocolCallbacks {
public:
  /// If \p CompileCommandsDir has a value, compile_commands.json will be
  /// loaded only from \p CompileCommandsDir. Otherwise, clangd will look
  /// for compile_commands.json in all parent directories of each file.
  ClangdLSPServer(JSONOutput &Out, unsigned AsyncThreadsCount,
                  bool SnippetCompletions,
                  llvm::Optional<StringRef> ResourceDir,
                  llvm::Optional<Path> CompileCommandsDir);

  /// Run LSP server loop, receiving input for it from \p In. \p In must be
  /// opened in binary mode. Output will be written using Out variable passed to
  /// class constructor. This method must not be executed more than once for
  /// each instance of ClangdLSPServer.
  ///
  /// \return Wether we received a 'shutdown' request before an 'exit' request
  bool run(std::istream &In);

private:
  // Implement DiagnosticsConsumer.
  virtual void
  onDiagnosticsReady(PathRef File,
                     Tagged<std::vector<DiagWithFixIts>> Diagnostics) override;

  // Implement ProtocolCallbacks.
  void onInitialize(Ctx C, InitializeParams &Params) override;
  void onShutdown(Ctx C, ShutdownParams &Params) override;
  void onExit(Ctx C, ExitParams &Params) override;
  void onDocumentDidOpen(Ctx C, DidOpenTextDocumentParams &Params) override;
  void onDocumentDidChange(Ctx C, DidChangeTextDocumentParams &Params) override;
  void onDocumentDidClose(Ctx C, DidCloseTextDocumentParams &Params) override;
  void
  onDocumentOnTypeFormatting(Ctx C,
                             DocumentOnTypeFormattingParams &Params) override;
  void
  onDocumentRangeFormatting(Ctx C,
                            DocumentRangeFormattingParams &Params) override;
  void onDocumentFormatting(Ctx C, DocumentFormattingParams &Params) override;
  void onCodeAction(Ctx C, CodeActionParams &Params) override;
  void onCompletion(Ctx C, TextDocumentPositionParams &Params) override;
  void onSignatureHelp(Ctx C, TextDocumentPositionParams &Params) override;
  void onGoToDefinition(Ctx C, TextDocumentPositionParams &Params) override;
  void onSwitchSourceHeader(Ctx C, TextDocumentIdentifier &Params) override;
  void onFileEvent(Ctx C, DidChangeWatchedFilesParams &Params) override;

  std::vector<clang::tooling::Replacement>
  getFixIts(StringRef File, const clangd::Diagnostic &D);

  JSONOutput &Out;
  /// Used to indicate that the 'shutdown' request was received from the
  /// Language Server client.
  bool ShutdownRequestReceived = false;

  /// Used to indicate that the 'exit' notification was received from the
  /// Language Server client.
  /// It's used to break out of the LSP parsing loop.
  bool IsDone = false;

  std::mutex FixItsMutex;
  typedef std::map<clangd::Diagnostic, std::vector<clang::tooling::Replacement>>
      DiagnosticToReplacementMap;
  /// Caches FixIts per file and diagnostics
  llvm::StringMap<DiagnosticToReplacementMap> FixItsMap;

  // Various ClangdServer parameters go here. It's important they're created
  // before ClangdServer.
  DirectoryBasedGlobalCompilationDatabase CDB;
  RealFileSystemProvider FSProvider;

  // Server must be the last member of the class to allow its destructor to exit
  // the worker thread that may otherwise run an async callback on partially
  // destructed instance of ClangdLSPServer.
  ClangdServer Server;
};

} // namespace clangd
} // namespace clang

#endif
