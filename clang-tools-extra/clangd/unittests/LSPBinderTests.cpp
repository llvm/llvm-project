//===-- LSPBinderTests.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LSPBinder.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

using testing::ElementsAre;
using testing::HasSubstr;
using testing::IsEmpty;
using testing::UnorderedElementsAre;

// JSON-serializable type for testing.
struct Foo {
  int X;
  friend bool operator==(Foo A, Foo B) { return A.X == B.X; }
};
bool fromJSON(const llvm::json::Value &V, Foo &F, llvm::json::Path P) {
  return fromJSON(V, F.X, P.field("X"));
}
llvm::json::Value toJSON(const Foo &F) { return F.X; }

// Creates a Callback that writes its received value into an Optional<Expected>.
template <typename T>
llvm::unique_function<void(llvm::Expected<T>)>
capture(llvm::Optional<llvm::Expected<T>> &Out) {
  Out.reset();
  return [&Out](llvm::Expected<T> V) { Out.emplace(std::move(V)); };
}

struct OutgoingRecorder : public LSPBinder::RawOutgoing {
  llvm::StringMap<std::vector<llvm::json::Value>> Received;

  void callMethod(llvm::StringRef Method, llvm::json::Value Params,
                  Callback<llvm::json::Value> Reply) override {
    Received[Method].push_back(Params);
    if (Method == "fail")
      return Reply(error("Params={0}", Params));
    Reply(Params); // echo back the request
  }
  void notify(llvm::StringRef Method, llvm::json::Value Params) override {
    Received[Method].push_back(std::move(Params));
  }

  std::vector<llvm::json::Value> take(llvm::StringRef Method) {
    std::vector<llvm::json::Value> Result = Received.lookup(Method);
    Received.erase(Method);
    return Result;
  }
};

TEST(LSPBinderTest, IncomingCalls) {
  LSPBinder::RawHandlers RawHandlers;
  OutgoingRecorder RawOutgoing;
  LSPBinder Binder{RawHandlers, RawOutgoing};
  struct Handler {
    void plusOne(const Foo &Params, Callback<Foo> Reply) {
      Reply(Foo{Params.X + 1});
    }
    void fail(const Foo &Params, Callback<Foo> Reply) {
      Reply(error("X={0}", Params.X));
    }
    void notify(const Foo &Params) {
      LastNotify = Params.X;
      ++NotifyCount;
    }
    int LastNotify = -1;
    int NotifyCount = 0;
  };

  Handler H;
  Binder.method("plusOne", &H, &Handler::plusOne);
  Binder.method("fail", &H, &Handler::fail);
  Binder.notification("notify", &H, &Handler::notify);
  Binder.command("cmdPlusOne", &H, &Handler::plusOne);
  ASSERT_THAT(RawHandlers.MethodHandlers.keys(),
              UnorderedElementsAre("plusOne", "fail"));
  ASSERT_THAT(RawHandlers.NotificationHandlers.keys(),
              UnorderedElementsAre("notify"));
  ASSERT_THAT(RawHandlers.CommandHandlers.keys(),
              UnorderedElementsAre("cmdPlusOne"));
  llvm::Optional<llvm::Expected<llvm::json::Value>> Reply;

  auto &RawPlusOne = RawHandlers.MethodHandlers["plusOne"];
  RawPlusOne(1, capture(Reply));
  ASSERT_TRUE(Reply.hasValue());
  EXPECT_THAT_EXPECTED(Reply.getValue(), llvm::HasValue(2));
  RawPlusOne("foo", capture(Reply));
  ASSERT_TRUE(Reply.hasValue());
  EXPECT_THAT_EXPECTED(
      Reply.getValue(),
      llvm::FailedWithMessage(
          HasSubstr("failed to decode plusOne request: expected integer")));

  auto &RawFail = RawHandlers.MethodHandlers["fail"];
  RawFail(2, capture(Reply));
  ASSERT_TRUE(Reply.hasValue());
  EXPECT_THAT_EXPECTED(Reply.getValue(), llvm::FailedWithMessage("X=2"));

  auto &RawNotify = RawHandlers.NotificationHandlers["notify"];
  RawNotify(42);
  EXPECT_EQ(H.LastNotify, 42);
  EXPECT_EQ(H.NotifyCount, 1);
  RawNotify("hi"); // invalid, will be logged
  EXPECT_EQ(H.LastNotify, 42);
  EXPECT_EQ(H.NotifyCount, 1);

  auto &RawCmdPlusOne = RawHandlers.CommandHandlers["cmdPlusOne"];
  RawCmdPlusOne(1, capture(Reply));
  ASSERT_TRUE(Reply.hasValue());
  EXPECT_THAT_EXPECTED(Reply.getValue(), llvm::HasValue(2));

  // None of this generated any outgoing traffic.
  EXPECT_THAT(RawOutgoing.Received, IsEmpty());
}

TEST(LSPBinderTest, OutgoingCalls) {
  LSPBinder::RawHandlers RawHandlers;
  OutgoingRecorder RawOutgoing;
  LSPBinder Binder{RawHandlers, RawOutgoing};

  LSPBinder::OutgoingMethod<Foo, Foo> Echo;
  Echo = Binder.outgoingMethod("echo");
  LSPBinder::OutgoingMethod<Foo, std::string> WrongSignature;
  WrongSignature = Binder.outgoingMethod("wrongSignature");
  LSPBinder::OutgoingMethod<Foo, Foo> Fail;
  Fail = Binder.outgoingMethod("fail");

  llvm::Optional<llvm::Expected<Foo>> Reply;
  Echo(Foo{2}, capture(Reply));
  EXPECT_THAT(RawOutgoing.take("echo"), ElementsAre(llvm::json::Value(2)));
  ASSERT_TRUE(Reply.hasValue());
  EXPECT_THAT_EXPECTED(Reply.getValue(), llvm::HasValue(Foo{2}));

  // JSON response is integer, can't be parsed as string.
  llvm::Optional<llvm::Expected<std::string>> WrongTypeReply;
  WrongSignature(Foo{2}, capture(WrongTypeReply));
  EXPECT_THAT(RawOutgoing.take("wrongSignature"),
              ElementsAre(llvm::json::Value(2)));
  ASSERT_TRUE(Reply.hasValue());
  EXPECT_THAT_EXPECTED(WrongTypeReply.getValue(),
                       llvm::FailedWithMessage(
                           HasSubstr("failed to decode wrongSignature reply")));

  Fail(Foo{2}, capture(Reply));
  EXPECT_THAT(RawOutgoing.take("fail"), ElementsAre(llvm::json::Value(2)));
  ASSERT_TRUE(Reply.hasValue());
  EXPECT_THAT_EXPECTED(Reply.getValue(), llvm::FailedWithMessage("Params=2"));
}

} // namespace
} // namespace clangd
} // namespace clang
