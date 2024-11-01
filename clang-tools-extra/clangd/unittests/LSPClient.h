//===-- LSPClient.h - Helper for ClangdLSPServer tests ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_UNITTESTS_LSPCLIENT_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_UNITTESTS_LSPCLIENT_H

#include "llvm/ADT/StringRef.h"
#include <condition_variable>
#include <llvm/Support/Error.h>
#include <llvm/Support/JSON.h>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace clang {
namespace clangd {
class Transport;

// A client library for talking to ClangdLSPServer in tests.
// Manages serialization of messages, pairing requests/repsonses, and implements
// the Transport abstraction.
class LSPClient {
  class TransportImpl;
  std::unique_ptr<TransportImpl> T;

public:
  // Represents the result of an LSP call: a promise for a result or error.
  class CallResult {
  public:
    ~CallResult();
    // Blocks up to 60 seconds for the result to be ready.
    // Records a test failure if there was no reply.
    llvm::Expected<llvm::json::Value> take();
    // Like take(), but records a test failure if the result was an error.
    llvm::json::Value takeValue();

  private:
    // Should be called once to provide the value.
    void set(llvm::Expected<llvm::json::Value> V);

    std::optional<llvm::Expected<llvm::json::Value>> Value;
    std::mutex Mu;
    std::condition_variable CV;

    friend TransportImpl; // Calls set().
  };

  LSPClient();
  ~LSPClient();
  LSPClient(LSPClient &&) = delete;
  LSPClient &operator=(LSPClient &&) = delete;

  // Enqueue an LSP method call, returns a promise for the reply. Threadsafe.
  CallResult &call(llvm::StringRef Method, llvm::json::Value Params);
  // Enqueue an LSP notification. Threadsafe.
  void notify(llvm::StringRef Method, llvm::json::Value Params);
  // Returns matching notifications since the last call to takeNotifications.
  std::vector<llvm::json::Value> takeNotifications(llvm::StringRef Method);
  // The transport is shut down after all pending messages are sent.
  void stop();

  // Shorthand for common LSP methods. Relative paths are passed to testPath().
  static llvm::json::Value uri(llvm::StringRef Path);
  static llvm::json::Value documentID(llvm::StringRef Path);
  void didOpen(llvm::StringRef Path, llvm::StringRef Content);
  void didChange(llvm::StringRef Path, llvm::StringRef Content);
  void didClose(llvm::StringRef Path);
  // Blocks until the server is idle (using the 'sync' protocol extension).
  void sync();
  // sync()s to ensure pending diagnostics arrive, and returns the newest set.
  std::optional<std::vector<llvm::json::Value>>
  diagnostics(llvm::StringRef Path);

  // Get the transport used to connect this client to a ClangdLSPServer.
  Transport &transport();

private:
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_UNITTESTS_LSPCLIENT_H
