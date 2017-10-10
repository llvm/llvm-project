//===--- JSONRPCDispatcher.h - Main JSON parser entry point -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_JSONRPCDISPATCHER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_JSONRPCDISPATCHER_H

#include "Logger.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/YAMLParser.h"
#include <iosfwd>
#include <mutex>

namespace clang {
namespace clangd {

/// Encapsulates output and logs streams and provides thread-safe access to
/// them.
class JSONOutput : public Logger {
public:
  JSONOutput(llvm::raw_ostream &Outs, llvm::raw_ostream &Logs,
             llvm::raw_ostream *InputMirror = nullptr)
      : Outs(Outs), Logs(Logs), InputMirror(InputMirror) {}

  /// Emit a JSONRPC message.
  void writeMessage(const Twine &Message);

  /// Write to the logging stream.
  void log(const Twine &Message) override;

  /// Mirror \p Message into InputMirror stream. Does nothing if InputMirror is
  /// null.
  /// Unlike other methods of JSONOutput, mirrorInput is not thread-safe.
  void mirrorInput(const Twine &Message);

private:
  llvm::raw_ostream &Outs;
  llvm::raw_ostream &Logs;
  llvm::raw_ostream *InputMirror;

  std::mutex StreamMutex;
};

/// Callback for messages sent to the server, called by the JSONRPCDispatcher.
class Handler {
public:
  Handler(JSONOutput &Output) : Output(Output) {}
  virtual ~Handler() = default;

  /// Called when the server receives a method call. This is supposed to return
  /// a result on Outs. The default implementation returns an "unknown method"
  /// error to the client and logs a warning.
  virtual void handleMethod(llvm::yaml::MappingNode *Params, StringRef ID);
  /// Called when the server receives a notification. No result should be
  /// written to Outs. The default implemetation logs a warning.
  virtual void handleNotification(llvm::yaml::MappingNode *Params);

protected:
  JSONOutput &Output;

  /// Helper to write a JSONRPC result to Output.
  void writeMessage(const Twine &Message) { Output.writeMessage(Message); }
};

/// Main JSONRPC entry point. This parses the JSONRPC "header" and calls the
/// registered Handler for the method received.
class JSONRPCDispatcher {
public:
  /// Create a new JSONRPCDispatcher. UnknownHandler is called when an unknown
  /// method is received.
  JSONRPCDispatcher(std::unique_ptr<Handler> UnknownHandler)
      : UnknownHandler(std::move(UnknownHandler)) {}

  /// Registers a Handler for the specified Method.
  void registerHandler(StringRef Method, std::unique_ptr<Handler> H);

  /// Parses a JSONRPC message and calls the Handler for it.
  bool call(StringRef Content) const;

private:
  llvm::StringMap<std::unique_ptr<Handler>> Handlers;
  std::unique_ptr<Handler> UnknownHandler;
};

/// Parses input queries from LSP client (coming from \p In) and runs call
/// method of \p Dispatcher for each query.
/// After handling each query checks if \p IsDone is set true and exits the loop
/// if it is.
/// Input stream(\p In) must be opened in binary mode to avoid preliminary
/// replacements of \r\n with \n.
void runLanguageServerLoop(std::istream &In, JSONOutput &Out,
                           JSONRPCDispatcher &Dispatcher, bool &IsDone);

} // namespace clangd
} // namespace clang

#endif
