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
class ClangdLSPServer;
class ClangdLSPServer;

struct InitializeHandler : Handler {
  InitializeHandler(JSONOutput &Output) : Handler(Output) {}

  void handleMethod(llvm::yaml::MappingNode *Params, StringRef ID) override {
    writeMessage(
        R"({"jsonrpc":"2.0","id":)" + ID +
        R"(,"result":{"capabilities":{
          "textDocumentSync": 1,
          "documentFormattingProvider": true,
          "documentRangeFormattingProvider": true,
          "documentOnTypeFormattingProvider": {"firstTriggerCharacter":"}","moreTriggerCharacter":[]},
          "codeActionProvider": true,
          "completionProvider": {"resolveProvider": false, "triggerCharacters": [".",">"]}
        }}})");
  }
};

struct ShutdownHandler : Handler {
  ShutdownHandler(JSONOutput &Output) : Handler(Output) {}

  void handleMethod(llvm::yaml::MappingNode *Params, StringRef ID) override {
    IsDone = true;
  }

  bool isDone() const { return IsDone; }

private:
  bool IsDone = false;
};

struct TextDocumentDidOpenHandler : Handler {
  TextDocumentDidOpenHandler(JSONOutput &Output, ClangdLSPServer &AST)
      : Handler(Output), AST(AST) {}

  void handleNotification(llvm::yaml::MappingNode *Params) override;

private:
  ClangdLSPServer &AST;
};

struct TextDocumentDidChangeHandler : Handler {
  TextDocumentDidChangeHandler(JSONOutput &Output, ClangdLSPServer &AST)
      : Handler(Output), AST(AST) {}

  void handleNotification(llvm::yaml::MappingNode *Params) override;

private:
  ClangdLSPServer &AST;
};

struct TextDocumentDidCloseHandler : Handler {
  TextDocumentDidCloseHandler(JSONOutput &Output, ClangdLSPServer &AST)
      : Handler(Output), AST(AST) {}

  void handleNotification(llvm::yaml::MappingNode *Params) override;

private:
  ClangdLSPServer &AST;
};

struct TextDocumentOnTypeFormattingHandler : Handler {
  TextDocumentOnTypeFormattingHandler(JSONOutput &Output, ClangdLSPServer &AST)
      : Handler(Output), AST(AST) {}

  void handleMethod(llvm::yaml::MappingNode *Params, StringRef ID) override;

private:
  ClangdLSPServer &AST;
};

struct TextDocumentRangeFormattingHandler : Handler {
  TextDocumentRangeFormattingHandler(JSONOutput &Output, ClangdLSPServer &AST)
      : Handler(Output), AST(AST) {}

  void handleMethod(llvm::yaml::MappingNode *Params, StringRef ID) override;

private:
  ClangdLSPServer &AST;
};

struct TextDocumentFormattingHandler : Handler {
  TextDocumentFormattingHandler(JSONOutput &Output, ClangdLSPServer &AST)
      : Handler(Output), AST(AST) {}

  void handleMethod(llvm::yaml::MappingNode *Params, StringRef ID) override;

private:
  ClangdLSPServer &AST;
};

struct CodeActionHandler : Handler {
  CodeActionHandler(JSONOutput &Output, ClangdLSPServer &AST)
      : Handler(Output), AST(AST) {}

  void handleMethod(llvm::yaml::MappingNode *Params, StringRef ID) override;

private:
  ClangdLSPServer &AST;
};

struct CompletionHandler : Handler {
  CompletionHandler(JSONOutput &Output, ClangdLSPServer &AST)
      : Handler(Output), AST(AST) {}

  void handleMethod(llvm::yaml::MappingNode *Params, StringRef ID) override;

 private:
  ClangdLSPServer &AST;
};

} // namespace clangd
} // namespace clang

#endif
