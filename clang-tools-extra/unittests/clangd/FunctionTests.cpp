//===-- FunctionsTests.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Function.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TEST(EventTest, Subscriptions) {
  Event<int> E;
  int N = 0;
  {
    Event<int>::Subscription SubA;
    // No subscriptions are active.
    E.broadcast(42);
    EXPECT_EQ(0, N);

    Event<int>::Subscription SubB = E.observe([&](int) { ++N; });
    // Now one is active.
    E.broadcast(42);
    EXPECT_EQ(1, N);

    SubA = E.observe([&](int) { ++N; });
    // Both are active.
    EXPECT_EQ(1, N);
    E.broadcast(42);
    EXPECT_EQ(3, N);

    SubA = std::move(SubB);
    // One is active.
    EXPECT_EQ(3, N);
    E.broadcast(42);
    EXPECT_EQ(4, N);
  }
  // None are active.
  EXPECT_EQ(4, N);
  E.broadcast(42);
  EXPECT_EQ(4, N);
}

} // namespace
} // namespace clangd
} // namespace clang
