//===--- ProtocolHandlers.h - LSP callbacks ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the actions performed when the server gets a specific
// request.
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

class ProtocolCallbacks {
public:
  virtual ~ProtocolCallbacks() = default;

  virtual void onInitialize(StringRef ID, InitializeParams IP,
                            JSONOutput &Out) = 0;
  virtual void onShutdown(JSONOutput &Out) = 0;
  virtual void onDocumentDidOpen(DidOpenTextDocumentParams Params,
                                 JSONOutput &Out) = 0;
  virtual void onDocumentDidChange(DidChangeTextDocumentParams Params,
                                   JSONOutput &Out) = 0;

  virtual void onDocumentDidClose(DidCloseTextDocumentParams Params,
                                  JSONOutput &Out) = 0;
  virtual void onDocumentFormatting(DocumentFormattingParams Params,
                                    StringRef ID, JSONOutput &Out) = 0;
  virtual void onDocumentOnTypeFormatting(DocumentOnTypeFormattingParams Params,
                                          StringRef ID, JSONOutput &Out) = 0;
  virtual void onDocumentRangeFormatting(DocumentRangeFormattingParams Params,
                                         StringRef ID, JSONOutput &Out) = 0;
  virtual void onCodeAction(CodeActionParams Params, StringRef ID,
                            JSONOutput &Out) = 0;
  virtual void onCompletion(TextDocumentPositionParams Params, StringRef ID,
                            JSONOutput &Out) = 0;
  virtual void onSignatureHelp(TextDocumentPositionParams Params, StringRef ID,
                               JSONOutput &Out) = 0;
  virtual void onGoToDefinition(TextDocumentPositionParams Params, StringRef ID,
                                JSONOutput &Out) = 0;
  virtual void onSwitchSourceHeader(TextDocumentIdentifier Params, StringRef ID,
                                    JSONOutput &Out) = 0;
  virtual void onFileEvent(const DidChangeWatchedFilesParams &Params) = 0;
};

void registerCallbackHandlers(JSONRPCDispatcher &Dispatcher, JSONOutput &Out,
                              ProtocolCallbacks &Callbacks);

} // namespace clangd
} // namespace clang

#endif
