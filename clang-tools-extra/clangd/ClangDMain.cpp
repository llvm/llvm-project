//===--- ClangDMain.cpp - clangd server loop ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ASTManager.h"
#include "DocumentStore.h"
#include "JSONRPCDispatcher.h"
#include "ProtocolHandlers.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include <iostream>
#include <string>
using namespace clang::clangd;

int main(int argc, char *argv[]) {
  llvm::raw_ostream &Outs = llvm::outs();
  llvm::raw_ostream &Logs = llvm::errs();
  JSONOutput Out(Outs, Logs);

  // Change stdin to binary to not lose \r\n on windows.
  llvm::sys::ChangeStdinToBinary();

  // Set up a document store and intialize all the method handlers for JSONRPC
  // dispatching.
  DocumentStore Store;
  ASTManager AST(Out, Store);
  Store.addListener(&AST);
  JSONRPCDispatcher Dispatcher(llvm::make_unique<Handler>(Out));
  Dispatcher.registerHandler("initialize",
                             llvm::make_unique<InitializeHandler>(Out));
  auto ShutdownPtr = llvm::make_unique<ShutdownHandler>(Out);
  auto *ShutdownHandler = ShutdownPtr.get();
  Dispatcher.registerHandler("shutdown",std::move(ShutdownPtr));
  Dispatcher.registerHandler(
      "textDocument/didOpen",
      llvm::make_unique<TextDocumentDidOpenHandler>(Out, Store));
  // FIXME: Implement textDocument/didClose.
  Dispatcher.registerHandler(
      "textDocument/didChange",
      llvm::make_unique<TextDocumentDidChangeHandler>(Out, Store));
  Dispatcher.registerHandler(
      "textDocument/rangeFormatting",
      llvm::make_unique<TextDocumentRangeFormattingHandler>(Out, Store));
  Dispatcher.registerHandler(
      "textDocument/formatting",
      llvm::make_unique<TextDocumentFormattingHandler>(Out, Store));

  while (std::cin.good()) {
    // A Language Server Protocol message starts with a HTTP header, delimited
    // by \r\n.
    std::string Line;
    std::getline(std::cin, Line);

    // Skip empty lines.
    llvm::StringRef LineRef(Line);
    if (LineRef.trim().empty())
      continue;

    // We allow YAML-style comments. Technically this isn't part of the
    // LSP specification, but makes writing tests easier.
    if (LineRef.startswith("#"))
      continue;

    unsigned long long Len = 0;
    // FIXME: Content-Type is a specified header, but does nothing.
    // Content-Length is a mandatory header. It specifies the length of the
    // following JSON.
    if (LineRef.consume_front("Content-Length: "))
      llvm::getAsUnsignedInteger(LineRef.trim(), 0, Len);

    // Check if the next line only contains \r\n. If not this is another header,
    // which we ignore.
    char NewlineBuf[2];
    std::cin.read(NewlineBuf, 2);
    if (std::memcmp(NewlineBuf, "\r\n", 2) != 0)
      continue;

    // Now read the JSON. Insert a trailing null byte as required by the YAML
    // parser.
    std::vector<char> JSON(Len + 1, '\0');
    std::cin.read(JSON.data(), Len);

    if (Len > 0) {
      llvm::StringRef JSONRef(JSON.data(), Len);
      // Log the message.
      Out.log("<-- " + JSONRef + "\n");

      // Finally, execute the action for this JSON message.
      if (!Dispatcher.call(JSONRef))
        Out.log("JSON dispatch failed!\n");

      // If we're done, exit the loop.
      if (ShutdownHandler->isDone())
        break;
    }
  }
}
