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

namespace {

// Helper for attaching ProtocolCallbacks methods to a JSONRPCDispatcher.
// Invoke like: Registerer("foo", &ProtocolCallbacks::onFoo)
// onFoo should be: void onFoo(Ctx &C, FooParams &Params)
// FooParams should have a static factory method: parse(yaml::MappingNode*).
struct HandlerRegisterer {
  template <typename Param>
  void operator()(StringRef Method,
                  void (ProtocolCallbacks::*Handler)(RequestContext, Param)) {
    // Capture pointers by value, as the lambda will outlive this object.
    auto *Out = this->Out;
    auto *Callbacks = this->Callbacks;
    Dispatcher.registerHandler(
        Method, [=](RequestContext C, llvm::yaml::MappingNode *RawParams) {
          if (auto P = [&] {
                trace::Span Tracer("Parse");
                return std::decay<Param>::type::parse(RawParams, *Out);
              }()) {
            (Callbacks->*Handler)(std::move(C), *P);
          } else {
            Out->log("Failed to decode " + Method + " request.\n");
          }
        });
  }

  JSONRPCDispatcher &Dispatcher;
  JSONOutput *Out;
  ProtocolCallbacks *Callbacks;
};

} // namespace

void clangd::registerCallbackHandlers(JSONRPCDispatcher &Dispatcher,
                                      JSONOutput &Out,
                                      ProtocolCallbacks &Callbacks) {
  HandlerRegisterer Register{Dispatcher, &Out, &Callbacks};

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
  Register("workspace/didChangeWatchedFiles", &ProtocolCallbacks::onFileEvent);
  Register("workspace/executeCommand", &ProtocolCallbacks::onCommand);
}
