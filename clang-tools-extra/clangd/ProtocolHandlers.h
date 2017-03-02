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
class ASTManager;
class DocumentStore;

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
          "codeActionProvider": true
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
  TextDocumentDidOpenHandler(JSONOutput &Output, DocumentStore &Store)
      : Handler(Output), Store(Store) {}

  void handleNotification(llvm::yaml::MappingNode *Params) override;

private:
  DocumentStore &Store;
};

struct TextDocumentDidChangeHandler : Handler {
  TextDocumentDidChangeHandler(JSONOutput &Output, DocumentStore &Store)
      : Handler(Output), Store(Store) {}

  void handleNotification(llvm::yaml::MappingNode *Params) override;

private:
  DocumentStore &Store;
};

struct TextDocumentOnTypeFormattingHandler : Handler {
  TextDocumentOnTypeFormattingHandler(JSONOutput &Output, DocumentStore &Store)
      : Handler(Output), Store(Store) {}

  void handleMethod(llvm::yaml::MappingNode *Params, StringRef ID) override;

private:
  DocumentStore &Store;
};

struct TextDocumentRangeFormattingHandler : Handler {
  TextDocumentRangeFormattingHandler(JSONOutput &Output, DocumentStore &Store)
      : Handler(Output), Store(Store) {}

  void handleMethod(llvm::yaml::MappingNode *Params, StringRef ID) override;

private:
  DocumentStore &Store;
};

struct TextDocumentFormattingHandler : Handler {
  TextDocumentFormattingHandler(JSONOutput &Output, DocumentStore &Store)
      : Handler(Output), Store(Store) {}

  void handleMethod(llvm::yaml::MappingNode *Params, StringRef ID) override;

private:
  DocumentStore &Store;
};

struct CodeActionHandler : Handler {
  CodeActionHandler(JSONOutput &Output, ASTManager &AST)
      : Handler(Output), AST(AST) {}

  void handleMethod(llvm::yaml::MappingNode *Params, StringRef ID) override;

private:
  ASTManager &AST;
};

} // namespace clangd
} // namespace clang

#endif
