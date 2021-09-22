//===-- JSONTransportTests.cpp  -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Protocol.h"
#include "Transport.h"
#include "support/Cancellation.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstdio>

namespace clang {
namespace clangd {
namespace {

// No fmemopen on windows or on versions of MacOS X earlier than 10.13, so we
// can't easily run this test.
#if !(defined(_WIN32) || (defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) && \
                          __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ < 101300))

// Fixture takes care of managing the input/output buffers for the transport.
class JSONTransportTest : public ::testing::Test {
  std::string InBuf, OutBuf, MirrorBuf;
  llvm::raw_string_ostream Out, Mirror;
  std::unique_ptr<FILE, int (*)(FILE *)> In;

protected:
  JSONTransportTest() : Out(OutBuf), Mirror(MirrorBuf), In(nullptr, nullptr) {}

  template <typename... Args>
  std::unique_ptr<Transport> transport(std::string InData, bool Pretty,
                                       JSONStreamStyle Style) {
    InBuf = std::move(InData);
    In = {fmemopen(&InBuf[0], InBuf.size(), "r"), &fclose};
    return newJSONTransport(In.get(), Out, &Mirror, Pretty, Style);
  }

  std::string input() const { return InBuf; }
  std::string output() { return Out.str(); }
  std::string input_mirror() { return Mirror.str(); }
};

// Echo is a simple server running on a transport:
//   - logs each message it gets.
//   - when it gets a call, replies to it
//   - when it gets a notification for method "call", makes a call on Target
// Hangs up when it gets an exit notification.
class Echo : public Transport::MessageHandler {
  Transport &Target;
  std::string LogBuf;
  llvm::raw_string_ostream Log;

public:
  Echo(Transport &Target) : Target(Target), Log(LogBuf) {}

  std::string log() { return Log.str(); }

  bool onNotify(llvm::StringRef Method, llvm::json::Value Params) override {
    Log << "Notification " << Method << ": " << Params << "\n";
    if (Method == "call")
      Target.call("echo call", std::move(Params), 42);
    return Method != "exit";
  }

  bool onCall(llvm::StringRef Method, llvm::json::Value Params,
              llvm::json::Value ID) override {
    Log << "Call " << Method << "(" << ID << "): " << Params << "\n";
    if (Method == "err")
      Target.reply(
          ID, llvm::make_error<LSPError>("trouble at mill", ErrorCode(88)));
    else if (Method == "invalidated") // gone out skew on treadle
      Target.reply(ID, llvm::make_error<CancelledError>(
                           static_cast<int>(ErrorCode::ContentModified)));
    else
      Target.reply(ID, std::move(Params));
    return true;
  }

  bool onReply(llvm::json::Value ID,
               llvm::Expected<llvm::json::Value> Params) override {
    if (Params)
      Log << "Reply(" << ID << "): " << *Params << "\n";
    else
      Log << "Reply(" << ID
          << "): error = " << llvm::toString(Params.takeError()) << "\n";
    return true;
  }
};

std::string trim(llvm::StringRef S) { return S.trim().str(); }

// Runs an Echo session using the standard JSON-RPC format we use in production.
TEST_F(JSONTransportTest, StandardDense) {
  auto T = transport(
      "Content-Length: 52\r\n\r\n"
      R"({"jsonrpc": "2.0", "method": "call", "params": 1234})"
      "Content-Length: 46\r\n\r\n"
      R"({"jsonrpc": "2.0", "id": 1234, "result": 5678})"
      "Content-Length: 67\r\n\r\n"
      R"({"jsonrpc": "2.0", "method": "foo", "id": "abcd", "params": "efgh"})"
      "Content-Length: 73\r\n\r\n"
      R"({"jsonrpc": "2.0", "id": "xyz", "error": {"code": 99, "message": "bad!"}})"
      "Content-Length: 68\r\n\r\n"
      R"({"jsonrpc": "2.0", "method": "err", "id": "wxyz", "params": "boom!"})"
      "Content-Length: 36\r\n\r\n"
      R"({"jsonrpc": "2.0", "method": "exit"})",
      /*Pretty=*/false, JSONStreamStyle::Standard);
  Echo E(*T);
  auto Err = T->loop(E);
  EXPECT_FALSE(bool(Err)) << toString(std::move(Err));

  const char *WantLog = R"(
Notification call: 1234
Reply(1234): 5678
Call foo("abcd"): "efgh"
Reply("xyz"): error = 99: bad!
Call err("wxyz"): "boom!"
Notification exit: null
  )";
  EXPECT_EQ(trim(E.log()), trim(WantLog));
  const char *WantOutput =
      "Content-Length: 60\r\n\r\n"
      R"({"id":42,"jsonrpc":"2.0","method":"echo call","params":1234})"
      "Content-Length: 45\r\n\r\n"
      R"({"id":"abcd","jsonrpc":"2.0","result":"efgh"})"
      "Content-Length: 77\r\n\r\n"
      R"({"error":{"code":88,"message":"trouble at mill"},"id":"wxyz","jsonrpc":"2.0"})";
  EXPECT_EQ(output(), WantOutput);
  EXPECT_EQ(trim(input_mirror()), trim(input()));
}

// Runs an Echo session using the "delimited" input and pretty-printed output
// that we use in lit tests.
TEST_F(JSONTransportTest, DelimitedPretty) {
  auto T = transport(R"jsonrpc(
{"jsonrpc": "2.0", "method": "call", "params": 1234}
---
{"jsonrpc": "2.0", "id": 1234, "result": 5678}
---
{"jsonrpc": "2.0", "method": "foo", "id": "abcd", "params": "efgh"}
---
{"jsonrpc": "2.0", "id": "xyz", "error": {"code": 99, "message": "bad!"}}
---
{"jsonrpc": "2.0", "method": "invalidated", "id": "wxyz", "params": "boom!"}
---
{"jsonrpc": "2.0", "method": "exit"}
  )jsonrpc",
                     /*Pretty=*/true, JSONStreamStyle::Delimited);
  Echo E(*T);
  auto Err = T->loop(E);
  EXPECT_FALSE(bool(Err)) << toString(std::move(Err));

  const char *WantLog = R"(
Notification call: 1234
Reply(1234): 5678
Call foo("abcd"): "efgh"
Reply("xyz"): error = 99: bad!
Call invalidated("wxyz"): "boom!"
Notification exit: null
  )";
  EXPECT_EQ(trim(E.log()), trim(WantLog));
  const char *WantOutput = "Content-Length: 77\r\n\r\n"
                           R"({
  "id": 42,
  "jsonrpc": "2.0",
  "method": "echo call",
  "params": 1234
})"
                           "Content-Length: 58\r\n\r\n"
                           R"({
  "id": "abcd",
  "jsonrpc": "2.0",
  "result": "efgh"
})"
                           "Content-Length: 145\r\n\r\n"
                           R"({
  "error": {
    "code": -32801,
    "message": "Request cancelled because the document was modified"
  },
  "id": "wxyz",
  "jsonrpc": "2.0"
})";
  EXPECT_EQ(output(), WantOutput);
  EXPECT_EQ(trim(input_mirror()), trim(input()));
}

// IO errors such as EOF ane reported.
// The only successful return from loop() is if a handler returned false.
TEST_F(JSONTransportTest, EndOfFile) {
  auto T = transport("Content-Length: 52\r\n\r\n"
                     R"({"jsonrpc": "2.0", "method": "call", "params": 1234})",
                     /*Pretty=*/false, JSONStreamStyle::Standard);
  Echo E(*T);
  auto Err = T->loop(E);
  EXPECT_EQ(trim(E.log()), "Notification call: 1234");
  EXPECT_TRUE(bool(Err)); // Ran into EOF with no handler signalling done.
  consumeError(std::move(Err));
  EXPECT_EQ(trim(input_mirror()), trim(input()));
}

#endif

} // namespace
} // namespace clangd
} // namespace clang
