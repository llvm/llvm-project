//===--- ProtocolHandlers.h - LSP callbacks ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// ProtocolHandlers translates incoming JSON requests from JSONRPCDispatcher
// into method calls on ClangLSPServer.
//
// Currently it parses requests into objects, but the ClangLSPServer is
// responsible for producing JSON responses. We should move that here, too.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_PROTOCOLHANDLERS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_PROTOCOLHANDLERS_H

#include "JSONRPCDispatcher.h"
#include "Protocol.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace clangd {

// The interface implemented by ClangLSPServer to handle incoming requests.
class ProtocolCallbacks {
public:
  virtual ~ProtocolCallbacks() = default;

  virtual void onInitialize(InitializeParams &Params) = 0;
  virtual void onShutdown(ShutdownParams &Params) = 0;
  virtual void onExit(ExitParams &Params) = 0;
  virtual void onDocumentDidOpen(DidOpenTextDocumentParams &Params) = 0;
  virtual void onDocumentDidChange(DidChangeTextDocumentParams &Params) = 0;
  virtual void onDocumentDidClose(DidCloseTextDocumentParams &Params) = 0;
  virtual void onDocumentFormatting(DocumentFormattingParams &Params) = 0;
  virtual void onDocumentSymbol(DocumentSymbolParams &Params) = 0;
  virtual void
  onDocumentOnTypeFormatting(DocumentOnTypeFormattingParams &Params) = 0;
  virtual void
  onDocumentRangeFormatting(DocumentRangeFormattingParams &Params) = 0;
  virtual void onCodeAction(CodeActionParams &Params) = 0;
  virtual void onCompletion(TextDocumentPositionParams &Params) = 0;
  virtual void onSignatureHelp(TextDocumentPositionParams &Params) = 0;
  virtual void onGoToDefinition(TextDocumentPositionParams &Params) = 0;
  virtual void onSwitchSourceHeader(TextDocumentIdentifier &Params) = 0;
  virtual void onFileEvent(DidChangeWatchedFilesParams &Params) = 0;
  virtual void onCommand(ExecuteCommandParams &Params) = 0;
  virtual void onWorkspaceSymbol(WorkspaceSymbolParams &Params) = 0;
  virtual void onRename(RenameParams &Parames) = 0;
  virtual void onDocumentHighlight(TextDocumentPositionParams &Params) = 0;
  virtual void onHover(TextDocumentPositionParams &Params) = 0;
  virtual void onChangeConfiguration(DidChangeConfigurationParams &Params) = 0;
};

void registerCallbackHandlers(JSONRPCDispatcher &Dispatcher,
                              ProtocolCallbacks &Callbacks);

} // namespace clangd
} // namespace clang

#endif
