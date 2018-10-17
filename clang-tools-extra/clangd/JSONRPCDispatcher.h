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

#include "Cancellation.h"
#include "Logger.h"
#include "Protocol.h"
#include "Trace.h"
#include "Transport.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include <iosfwd>
#include <mutex>

namespace clang {
namespace clangd {

/// Sends a successful reply.
/// Current context must derive from JSONRPCDispatcher::Handler.
void reply(llvm::json::Value &&Result);
/// Sends an error response to the client, and logs it.
/// Current context must derive from JSONRPCDispatcher::Handler.
void replyError(ErrorCode Code, const llvm::StringRef &Message);
/// Implements ErrorCode and message extraction from a given llvm::Error. It
/// fetches the related message from error's message method. If error doesn't
/// match any known errors, uses ErrorCode::InvalidParams for the error.
void replyError(llvm::Error E);
/// Sends a request to the client.
/// Current context must derive from JSONRPCDispatcher::Handler.
void call(llvm::StringRef Method, llvm::json::Value &&Params);

/// Main JSONRPC entry point. This parses the JSONRPC "header" and calls the
/// registered Handler for the method received.
///
/// The `$/cancelRequest` notification is handled by the dispatcher itself.
/// It marks the matching request as cancelled, if it's still running.
class JSONRPCDispatcher : private Transport::MessageHandler {
public:
  /// A handler responds to requests for a particular method name.
  /// It returns false if the server should now shut down.
  ///
  /// JSONRPCDispatcher will mark the handler's context as cancelled if a
  /// matching cancellation request is received. Handlers are encouraged to
  /// check for cancellation and fail quickly in this case.
  using Handler = std::function<bool(const llvm::json::Value &)>;

  /// Create a new JSONRPCDispatcher. UnknownHandler is called when an unknown
  /// method is received.
  JSONRPCDispatcher(Handler UnknownHandler);

  /// Registers a Handler for the specified Method.
  void registerHandler(StringRef Method, Handler H);

  /// Parses input queries from LSP client (coming from \p In) and runs call
  /// method for each query.
  ///
  /// Input stream(\p In) must be opened in binary mode to avoid
  /// preliminary replacements of \r\n with \n. We use C-style FILE* for reading
  /// as std::istream has unclear interaction with signals, which are sent by
  /// debuggers on some OSs.
  llvm::Error runLanguageServerLoop(Transport &);

private:
  bool onReply(llvm::json::Value ID,
               llvm::Expected<llvm::json::Value> Result) override;
  bool onNotify(llvm::StringRef Method, llvm::json::Value Message) override;
  bool onCall(llvm::StringRef Method, llvm::json::Value Message,
              llvm::json::Value ID) override;

  // Tracking cancellations needs a mutex: handlers may finish on a different
  // thread, and that's when we clean up entries in the map.
  mutable std::mutex RequestCancelersMutex;
  llvm::StringMap<std::pair<Canceler, unsigned>> RequestCancelers;
  unsigned NextRequestCookie = 0;
  Context cancelableRequestContext(const llvm::json::Value &ID);
  void cancelRequest(const llvm::json::Value &ID);

  llvm::StringMap<Handler> Handlers;
  Handler UnknownHandler;
};

} // namespace clangd
} // namespace clang

#endif
