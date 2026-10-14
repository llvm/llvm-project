//===-- sanitizer_new_handler_test.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for the operator new failure handling in sanitizer_new_handler.h,
// driven by fake allocation and exhaustion callbacks.
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_platform.h"

#if !SANITIZER_WINDOWS && defined(__cpp_exceptions)

#  include <stdio.h>
#  include <stdlib.h>

#  include <new>
#  include <stdexcept>

#  include "gtest/gtest.h"
#  include "sanitizer_common/sanitizer_new_handler.h"

namespace __sanitizer {
namespace {

int handler_calls;
// Number of calls after which CountingHandler uninstalls itself.
int handler_budget;

void CountingHandler() {
  if (++handler_calls >= handler_budget)
    std::set_new_handler(nullptr);
}

void ThrowingHandler() { throw std::runtime_error("handler"); }

// Ends the retry loop by throwing std::bad_alloc while staying installed.
void BadAllocHandler() {
  ++handler_calls;
  throw std::bad_alloc();
}

void ReportExhausted() {
  fprintf(stderr, "exhausted after %d handler call(s)\n", handler_calls);
  abort();
}

// Exhaustion callbacks must not return; throwing records the failure without
// killing the test binary.
struct UnexpectedExhaustion {};

void UnexpectedExhausted() {
  ADD_FAILURE() << "unexpected exhaustion";
  throw UnexpectedExhaustion();
}

class NewHandlerTest : public ::testing::Test {
 protected:
  // The new_handler and allocator_may_return_null are process-wide; restore
  // them so other tests in this binary are unaffected.
  void SetUp() override {
    saved_handler_ = std::get_new_handler();
    saved_may_return_null_ = AllocatorMayReturnNull();
    handler_calls = 0;
    handler_budget = 1;
    attempts_ = 0;
  }

  void TearDown() override {
    std::set_new_handler(saved_handler_);
    SetAllocatorMayReturnNull(saved_may_return_null_);
  }

  // Returns an allocation callback that fails `failures` times, then succeeds.
  auto Alloc(int failures) {
    return [this, failures]() -> void* {
      return ++attempts_ > failures ? &storage_ : nullptr;
    };
  }

  // "Always" means the first ~million attempts. Eventually succeeding makes an
  // infinite retry loop fail with a clear diagnostic instead of timing out.
  static constexpr int kAlwaysFail = 1 << 20;
  int attempts_;
  char storage_;

 private:
  std::new_handler saved_handler_;
  bool saved_may_return_null_;
};

TEST_F(NewHandlerTest, SuccessSkipsHandler) {
  std::set_new_handler(CountingHandler);
  EXPECT_EQ(&storage_, RunNewHandlerChain(Alloc(0)));
  EXPECT_EQ(1, attempts_);
  EXPECT_EQ(0, handler_calls);
}

TEST_F(NewHandlerTest, RetriesAfterEachHandlerCall) {
  handler_budget = 10;
  std::set_new_handler(CountingHandler);
  EXPECT_EQ(&storage_, RunNewHandlerChain(Alloc(3)));
  EXPECT_EQ(4, attempts_);
  EXPECT_EQ(3, handler_calls);
}

TEST_F(NewHandlerTest, NoHandlerReturnsNull) {
  std::set_new_handler(nullptr);
  EXPECT_EQ(nullptr, RunNewHandlerChain(Alloc(kAlwaysFail)));
  EXPECT_EQ(1, attempts_);
}

TEST_F(NewHandlerTest, ExhaustedChainReturnsNull) {
  handler_budget = 2;
  std::set_new_handler(CountingHandler);
  EXPECT_EQ(nullptr, RunNewHandlerChain(Alloc(kAlwaysFail)));
  EXPECT_EQ(3, attempts_);
  EXPECT_EQ(2, handler_calls);
}

TEST_F(NewHandlerTest, ThrowingThrowsBadAllocOnExhaustionWhenMayReturnNull) {
  SetAllocatorMayReturnNull(true);
  std::set_new_handler(CountingHandler);
  EXPECT_THROW((void)NewImplThrowing(Alloc(kAlwaysFail), UnexpectedExhausted),
               std::bad_alloc);
  EXPECT_EQ(1, handler_calls);
}

TEST_F(NewHandlerTest, ThrowingReportsExhaustion) {
  SetAllocatorMayReturnNull(false);
  std::set_new_handler(CountingHandler);
  EXPECT_DEATH((void)NewImplThrowing(Alloc(kAlwaysFail), ReportExhausted),
               "exhausted after 1 handler call");
}

TEST_F(NewHandlerTest, ThrowingPropagatesHandlerException) {
  std::set_new_handler(ThrowingHandler);
  for (bool may_return_null : {false, true}) {
    SetAllocatorMayReturnNull(may_return_null);
    EXPECT_THROW((void)NewImplThrowing(Alloc(kAlwaysFail), UnexpectedExhausted),
                 std::runtime_error);
  }
}

TEST_F(NewHandlerTest, ThrowingPropagatesHandlerBadAlloc) {
  std::set_new_handler(BadAllocHandler);
  for (bool may_return_null : {false, true}) {
    SetAllocatorMayReturnNull(may_return_null);
    handler_calls = attempts_ = 0;
    EXPECT_THROW((void)NewImplThrowing(Alloc(kAlwaysFail), UnexpectedExhausted),
                 std::bad_alloc);
    EXPECT_EQ(1, attempts_);
    EXPECT_EQ(1, handler_calls);
    EXPECT_EQ(&BadAllocHandler, std::get_new_handler());
  }
}

TEST_F(NewHandlerTest, NothrowReturnsNullOnExhaustionWhenMayReturnNull) {
  SetAllocatorMayReturnNull(true);
  std::set_new_handler(CountingHandler);
  EXPECT_EQ(nullptr, NewImplNothrow(Alloc(kAlwaysFail), UnexpectedExhausted));
  EXPECT_EQ(1, handler_calls);
}

TEST_F(NewHandlerTest, NothrowReportsExhaustion) {
  SetAllocatorMayReturnNull(false);
  std::set_new_handler(CountingHandler);
  EXPECT_DEATH((void)NewImplNothrow(Alloc(kAlwaysFail), ReportExhausted),
               "exhausted after 1 handler call");
}

TEST_F(NewHandlerTest, NothrowSwallowsHandlerException) {
  std::set_new_handler(ThrowingHandler);
  for (bool may_return_null : {false, true}) {
    SetAllocatorMayReturnNull(may_return_null);
    EXPECT_EQ(nullptr, NewImplNothrow(Alloc(kAlwaysFail), UnexpectedExhausted));
  }
}

TEST_F(NewHandlerTest, NothrowReturnsNullOnHandlerBadAlloc) {
  std::set_new_handler(BadAllocHandler);
  for (bool may_return_null : {false, true}) {
    SetAllocatorMayReturnNull(may_return_null);
    handler_calls = attempts_ = 0;
    EXPECT_EQ(nullptr, NewImplNothrow(Alloc(kAlwaysFail), UnexpectedExhausted));
    EXPECT_EQ(1, attempts_);
    EXPECT_EQ(1, handler_calls);
    EXPECT_EQ(&BadAllocHandler, std::get_new_handler());
  }
}

TEST_F(NewHandlerTest, ReturningExhaustionCallbackDies) {
  EXPECT_DEATH(InvokeOnExhausted([] {}), "OnExhausted callable returned");
}

}  // namespace
}  // namespace __sanitizer

#endif  // !SANITIZER_WINDOWS && defined(__cpp_exceptions)
