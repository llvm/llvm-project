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
using namespace clang;
using namespace clangd;

namespace {

struct InitializeHandler : Handler {
  InitializeHandler(JSONOutput &Output, ProtocolCallbacks &Callbacks)
      : Handler(Output), Callbacks(Callbacks) {}

  void handleMethod(llvm::yaml::MappingNode *Params, StringRef ID) override {
    auto IP = InitializeParams::parse(Params, Output);
    if (!IP) {
      Output.log("Failed to decode InitializeParams!\n");
      IP = InitializeParams();
    }

    Callbacks.onInitialize(ID, *IP, Output);
  }

private:
  ProtocolCallbacks &Callbacks;
};

struct ShutdownHandler : Handler {
  ShutdownHandler(JSONOutput &Output, ProtocolCallbacks &Callbacks)
      : Handler(Output), Callbacks(Callbacks) {}

  void handleMethod(llvm::yaml::MappingNode *Params, StringRef ID) override {
    Callbacks.onShutdown(Output);
  }

private:
  ProtocolCallbacks &Callbacks;
};

struct TextDocumentDidOpenHandler : Handler {
  TextDocumentDidOpenHandler(JSONOutput &Output, ProtocolCallbacks &Callbacks)
      : Handler(Output), Callbacks(Callbacks) {}

  void handleNotification(llvm::yaml::MappingNode *Params) override {
    auto DOTDP = DidOpenTextDocumentParams::parse(Params, Output);
    if (!DOTDP) {
      Output.log("Failed to decode DidOpenTextDocumentParams!\n");
      return;
    }
    Callbacks.onDocumentDidOpen(*DOTDP, Output);
  }

private:
  ProtocolCallbacks &Callbacks;
};

struct TextDocumentDidChangeHandler : Handler {
  TextDocumentDidChangeHandler(JSONOutput &Output, ProtocolCallbacks &Callbacks)
      : Handler(Output), Callbacks(Callbacks) {}

  void handleNotification(llvm::yaml::MappingNode *Params) override {
    auto DCTDP = DidChangeTextDocumentParams::parse(Params, Output);
    if (!DCTDP || DCTDP->contentChanges.size() != 1) {
      Output.log("Failed to decode DidChangeTextDocumentParams!\n");
      return;
    }

    Callbacks.onDocumentDidChange(*DCTDP, Output);
  }

private:
  ProtocolCallbacks &Callbacks;
};

struct TextDocumentDidCloseHandler : Handler {
  TextDocumentDidCloseHandler(JSONOutput &Output, ProtocolCallbacks &Callbacks)
      : Handler(Output), Callbacks(Callbacks) {}

  void handleNotification(llvm::yaml::MappingNode *Params) override {
    auto DCTDP = DidCloseTextDocumentParams::parse(Params, Output);
    if (!DCTDP) {
      Output.log("Failed to decode DidCloseTextDocumentParams!\n");
      return;
    }

    Callbacks.onDocumentDidClose(*DCTDP, Output);
  }

private:
  ProtocolCallbacks &Callbacks;
};

struct TextDocumentOnTypeFormattingHandler : Handler {
  TextDocumentOnTypeFormattingHandler(JSONOutput &Output,
                                      ProtocolCallbacks &Callbacks)
      : Handler(Output), Callbacks(Callbacks) {}

  void handleMethod(llvm::yaml::MappingNode *Params, StringRef ID) override {
    auto DOTFP = DocumentOnTypeFormattingParams::parse(Params, Output);
    if (!DOTFP) {
      Output.log("Failed to decode DocumentOnTypeFormattingParams!\n");
      return;
    }

    Callbacks.onDocumentOnTypeFormatting(*DOTFP, ID, Output);
  }

private:
  ProtocolCallbacks &Callbacks;
};

struct TextDocumentRangeFormattingHandler : Handler {
  TextDocumentRangeFormattingHandler(JSONOutput &Output,
                                     ProtocolCallbacks &Callbacks)
      : Handler(Output), Callbacks(Callbacks) {}

  void handleMethod(llvm::yaml::MappingNode *Params, StringRef ID) override {
    auto DRFP = DocumentRangeFormattingParams::parse(Params, Output);
    if (!DRFP) {
      Output.log("Failed to decode DocumentRangeFormattingParams!\n");
      return;
    }

    Callbacks.onDocumentRangeFormatting(*DRFP, ID, Output);
  }

private:
  ProtocolCallbacks &Callbacks;
};

struct TextDocumentFormattingHandler : Handler {
  TextDocumentFormattingHandler(JSONOutput &Output,
                                ProtocolCallbacks &Callbacks)
      : Handler(Output), Callbacks(Callbacks) {}

  void handleMethod(llvm::yaml::MappingNode *Params, StringRef ID) override {
    auto DFP = DocumentFormattingParams::parse(Params, Output);
    if (!DFP) {
      Output.log("Failed to decode DocumentFormattingParams!\n");
      return;
    }

    Callbacks.onDocumentFormatting(*DFP, ID, Output);
  }

private:
  ProtocolCallbacks &Callbacks;
};

struct CodeActionHandler : Handler {
  CodeActionHandler(JSONOutput &Output, ProtocolCallbacks &Callbacks)
      : Handler(Output), Callbacks(Callbacks) {}

  void handleMethod(llvm::yaml::MappingNode *Params, StringRef ID) override {
    auto CAP = CodeActionParams::parse(Params, Output);
    if (!CAP) {
      Output.log("Failed to decode CodeActionParams!\n");
      return;
    }

    Callbacks.onCodeAction(*CAP, ID, Output);
  }

private:
  ProtocolCallbacks &Callbacks;
};

struct CompletionHandler : Handler {
  CompletionHandler(JSONOutput &Output, ProtocolCallbacks &Callbacks)
      : Handler(Output), Callbacks(Callbacks) {}

  void handleMethod(llvm::yaml::MappingNode *Params, StringRef ID) override {
    auto TDPP = TextDocumentPositionParams::parse(Params, Output);
    if (!TDPP) {
      Output.log("Failed to decode TextDocumentPositionParams!\n");
      return;
    }

    Callbacks.onCompletion(*TDPP, ID, Output);
  }

private:
  ProtocolCallbacks &Callbacks;
};

struct GotoDefinitionHandler : Handler {
  GotoDefinitionHandler(JSONOutput &Output, ProtocolCallbacks &Callbacks)
      : Handler(Output), Callbacks(Callbacks) {}

  void handleMethod(llvm::yaml::MappingNode *Params, StringRef ID) override {
    auto TDPP = TextDocumentPositionParams::parse(Params, Output);
    if (!TDPP) {
      Output.log("Failed to decode TextDocumentPositionParams!\n");
      return;
    }

    Callbacks.onGoToDefinition(*TDPP, ID, Output);
  }

private:
  ProtocolCallbacks &Callbacks;
};

struct SwitchSourceHeaderHandler : Handler {
  SwitchSourceHeaderHandler(JSONOutput &Output, ProtocolCallbacks &Callbacks)
      : Handler(Output), Callbacks(Callbacks) {}

  void handleMethod(llvm::yaml::MappingNode *Params, StringRef ID) override {
    auto TDPP = TextDocumentIdentifier::parse(Params, Output);
    if (!TDPP)
      return;

    Callbacks.onSwitchSourceHeader(*TDPP, ID, Output);
  }

private:
  ProtocolCallbacks &Callbacks;
};

struct WorkspaceDidChangeWatchedFilesHandler : Handler {
  WorkspaceDidChangeWatchedFilesHandler(JSONOutput &Output,
                                        ProtocolCallbacks &Callbacks)
      : Handler(Output), Callbacks(Callbacks) {}

  void handleNotification(llvm::yaml::MappingNode *Params) {
    auto DCWFP = DidChangeWatchedFilesParams::parse(Params, Output);
    if (!DCWFP) {
      Output.log("Failed to decode DidChangeWatchedFilesParams.\n");
      return;
    }

    Callbacks.onFileEvent(*DCWFP);
  }

private:
  ProtocolCallbacks &Callbacks;
};

} // namespace

void clangd::registerCallbackHandlers(JSONRPCDispatcher &Dispatcher,
                                     JSONOutput &Out,
                                     ProtocolCallbacks &Callbacks) {
  Dispatcher.registerHandler(
      "initialize", llvm::make_unique<InitializeHandler>(Out, Callbacks));
  Dispatcher.registerHandler(
      "shutdown", llvm::make_unique<ShutdownHandler>(Out, Callbacks));
  Dispatcher.registerHandler(
      "textDocument/didOpen",
      llvm::make_unique<TextDocumentDidOpenHandler>(Out, Callbacks));
  Dispatcher.registerHandler(
      "textDocument/didClose",
      llvm::make_unique<TextDocumentDidCloseHandler>(Out, Callbacks));
  Dispatcher.registerHandler(
      "textDocument/didChange",
      llvm::make_unique<TextDocumentDidChangeHandler>(Out, Callbacks));
  Dispatcher.registerHandler(
      "textDocument/rangeFormatting",
      llvm::make_unique<TextDocumentRangeFormattingHandler>(Out, Callbacks));
  Dispatcher.registerHandler(
      "textDocument/onTypeFormatting",
      llvm::make_unique<TextDocumentOnTypeFormattingHandler>(Out, Callbacks));
  Dispatcher.registerHandler(
      "textDocument/formatting",
      llvm::make_unique<TextDocumentFormattingHandler>(Out, Callbacks));
  Dispatcher.registerHandler(
      "textDocument/codeAction",
      llvm::make_unique<CodeActionHandler>(Out, Callbacks));
  Dispatcher.registerHandler(
      "textDocument/completion",
      llvm::make_unique<CompletionHandler>(Out, Callbacks));
  Dispatcher.registerHandler(
      "textDocument/definition",
      llvm::make_unique<GotoDefinitionHandler>(Out, Callbacks));
  Dispatcher.registerHandler(
      "textDocument/switchSourceHeader",
      llvm::make_unique<SwitchSourceHeaderHandler>(Out, Callbacks));
  Dispatcher.registerHandler(
      "workspace/didChangeWatchedFiles",
      llvm::make_unique<WorkspaceDidChangeWatchedFilesHandler>(Out, Callbacks));
}
