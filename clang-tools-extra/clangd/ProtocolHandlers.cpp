//===--- ProtocolHandlers.cpp - LSP callbacks -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ProtocolHandlers.h"
#include "ClangdLSPServer.h"
#include "ClangdServer.h"
#include "DraftStore.h"
#include "Trace.h"

using namespace clang;
using namespace clang::clangd;
using namespace llvm;

namespace {

// Helper for attaching ProtocolCallbacks methods to a JSONRPCDispatcher.
// Invoke like: Registerer("foo", &ProtocolCallbacks::onFoo)
// onFoo should be: void onFoo(Ctx &C, FooParams &Params)
// FooParams should have a fromJSON function.
struct HandlerRegisterer {
  template <typename Param>
  void operator()(StringRef Method, void (ProtocolCallbacks::*Handler)(Param)) {
    // Capture pointers by value, as the lambda will outlive this object.
    auto *Callbacks = this->Callbacks;
    Dispatcher.registerHandler(Method, [=](const json::Value &RawParams) {
      typename std::remove_reference<Param>::type P;
      if (fromJSON(RawParams, P)) {
        (Callbacks->*Handler)(P);
      } else {
        elog("Failed to decode {0} request.", Method);
      }
    });
  }

  JSONRPCDispatcher &Dispatcher;
  ProtocolCallbacks *Callbacks;
};

} // namespace

void clangd::registerCallbackHandlers(JSONRPCDispatcher &Dispatcher,
                                      ProtocolCallbacks &Callbacks) {
  HandlerRegisterer Register{Dispatcher, &Callbacks};

  Register("initialize", &ProtocolCallbacks::onInitialize);
  Register("shutdown", &ProtocolCallbacks::onShutdown);
  Register("exit", &ProtocolCallbacks::onExit);
  Register("textDocument/didOpen", &ProtocolCallbacks::onDocumentDidOpen);
  Register("textDocument/didClose", &ProtocolCallbacks::onDocumentDidClose);
  Register("textDocument/didChange", &ProtocolCallbacks::onDocumentDidChange);
  Register("textDocument/rangeFormatting",
           &ProtocolCallbacks::onDocumentRangeFormatting);
  Register("textDocument/onTypeFormatting",
           &ProtocolCallbacks::onDocumentOnTypeFormatting);
  Register("textDocument/formatting", &ProtocolCallbacks::onDocumentFormatting);
  Register("textDocument/codeAction", &ProtocolCallbacks::onCodeAction);
  Register("textDocument/completion", &ProtocolCallbacks::onCompletion);
  Register("textDocument/signatureHelp", &ProtocolCallbacks::onSignatureHelp);
  Register("textDocument/definition", &ProtocolCallbacks::onGoToDefinition);
  Register("textDocument/switchSourceHeader",
           &ProtocolCallbacks::onSwitchSourceHeader);
  Register("textDocument/rename", &ProtocolCallbacks::onRename);
  Register("textDocument/hover", &ProtocolCallbacks::onHover);
  Register("textDocument/documentSymbol", &ProtocolCallbacks::onDocumentSymbol);
  Register("workspace/didChangeWatchedFiles", &ProtocolCallbacks::onFileEvent);
  Register("workspace/executeCommand", &ProtocolCallbacks::onCommand);
  Register("textDocument/documentHighlight",
           &ProtocolCallbacks::onDocumentHighlight);
  Register("workspace/didChangeConfiguration",
           &ProtocolCallbacks::onChangeConfiguration);
  Register("workspace/symbol", &ProtocolCallbacks::onWorkspaceSymbol);
}
