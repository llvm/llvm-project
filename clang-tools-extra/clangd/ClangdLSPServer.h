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
  void run(std::istream &In);

private:
  // Implement DiagnosticsConsumer.
  virtual void
  onDiagnosticsReady(PathRef File,
                     Tagged<std::vector<DiagWithFixIts>> Diagnostics) override;

  // Implement ProtocolCallbacks.
  void onInitialize(StringRef ID, InitializeParams IP,
                    JSONOutput &Out) override;
  void onShutdown(JSONOutput &Out) override;
  void onDocumentDidOpen(DidOpenTextDocumentParams Params,
                         JSONOutput &Out) override;
  void onDocumentDidChange(DidChangeTextDocumentParams Params,
                           JSONOutput &Out) override;
  void onDocumentDidClose(DidCloseTextDocumentParams Params,
                          JSONOutput &Out) override;
  void onDocumentOnTypeFormatting(DocumentOnTypeFormattingParams Params,
                                  StringRef ID, JSONOutput &Out) override;
  void onDocumentRangeFormatting(DocumentRangeFormattingParams Params,
                                 StringRef ID, JSONOutput &Out) override;
  void onDocumentFormatting(DocumentFormattingParams Params, StringRef ID,
                            JSONOutput &Out) override;
  void onCodeAction(CodeActionParams Params, StringRef ID,
                    JSONOutput &Out) override;
  void onCompletion(TextDocumentPositionParams Params, StringRef ID,
                    JSONOutput &Out) override;
  void onGoToDefinition(TextDocumentPositionParams Params, StringRef ID,
                        JSONOutput &Out) override;
  void onSwitchSourceHeader(TextDocumentIdentifier Params, StringRef ID,
                            JSONOutput &Out) override;
  void onFileEvent(const DidChangeWatchedFilesParams &Params) override;

  std::vector<clang::tooling::Replacement>
  getFixIts(StringRef File, const clangd::Diagnostic &D);

  /// Function that will be called on a separate thread when diagnostics are
  /// ready. Sends the Dianostics to LSP client via Out.writeMessage and caches
  /// corresponding fixits in the FixItsMap.
  void consumeDiagnostics(PathRef File,
                          std::vector<DiagWithFixIts> Diagnostics);

  JSONOutput &Out;
  /// Used to indicate that the 'shutdown' request was received from the
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
