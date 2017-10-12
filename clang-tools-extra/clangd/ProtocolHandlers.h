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
  using Ctx = RequestContext;
  virtual ~ProtocolCallbacks() = default;

  virtual void onInitialize(Ctx C, InitializeParams &Params) = 0;
  virtual void onShutdown(Ctx C, ShutdownParams &Params) = 0;
  virtual void onDocumentDidOpen(Ctx C, DidOpenTextDocumentParams &Params) = 0;
  virtual void onDocumentDidChange(Ctx C,
                                   DidChangeTextDocumentParams &Params) = 0;
  virtual void onDocumentDidClose(Ctx C,
                                  DidCloseTextDocumentParams &Params) = 0;
  virtual void onDocumentFormatting(Ctx C,
                                    DocumentFormattingParams &Params) = 0;
  virtual void
  onDocumentOnTypeFormatting(Ctx C, DocumentOnTypeFormattingParams &Params) = 0;
  virtual void
  onDocumentRangeFormatting(Ctx C, DocumentRangeFormattingParams &Params) = 0;
  virtual void onCodeAction(Ctx C, CodeActionParams &Params) = 0;
  virtual void onCompletion(Ctx C, TextDocumentPositionParams &Params) = 0;
  virtual void onSignatureHelp(Ctx C, TextDocumentPositionParams &Params) = 0;
  virtual void onGoToDefinition(Ctx C, TextDocumentPositionParams &Params) = 0;
  virtual void onSwitchSourceHeader(Ctx C, TextDocumentIdentifier &Params) = 0;
  virtual void onFileEvent(Ctx C, DidChangeWatchedFilesParams &Params) = 0;
};

void registerCallbackHandlers(JSONRPCDispatcher &Dispatcher, JSONOutput &Out,
                              ProtocolCallbacks &Callbacks);

} // namespace clangd
} // namespace clang

#endif
