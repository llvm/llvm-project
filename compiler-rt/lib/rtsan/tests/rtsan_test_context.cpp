//===--- rtsan_test_context.cpp - Realtime Sanitizer ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "rtsan_test_utilities.h"

#include "rtsan/rtsan.h"
#include "rtsan/rtsan_context.h"

#include <gtest/gtest.h>

using namespace ::testing;

class TestRtsanContext : public Test {
protected:
  void SetUp() override { __rtsan_ensure_initialized(); }
};

TEST_F(TestRtsanContext, IsNotRealtimeAfterDefaultConstruction) {
  __rtsan::Context context{};
  EXPECT_THAT(context.InRealtimeContext(), Eq(false));
}

TEST_F(TestRtsanContext, IsRealtimeAfterRealtimePush) {
  __rtsan::Context context{};
  context.RealtimePush();
  EXPECT_THAT(context.InRealtimeContext(), Eq(true));
}

TEST_F(TestRtsanContext, IsNotRealtimeAfterRealtimePushAndPop) {
  __rtsan::Context context{};
  context.RealtimePush();
  ASSERT_THAT(context.InRealtimeContext(), Eq(true));
  context.RealtimePop();
  EXPECT_THAT(context.InRealtimeContext(), Eq(false));
}

TEST_F(TestRtsanContext, RealtimeContextStateIsStatefullyTracked) {
  __rtsan::Context context{};
  auto const expect_rt = [&context](bool is_rt) {
    EXPECT_THAT(context.InRealtimeContext(), Eq(is_rt));
  };
  expect_rt(false);
  context.RealtimePush(); // depth 1
  expect_rt(true);
  context.RealtimePush(); // depth 2
  expect_rt(true);
  context.RealtimePop(); // depth 1
  expect_rt(true);
  context.RealtimePush(); // depth 2
  expect_rt(true);
  context.RealtimePop(); // depth 1
  expect_rt(true);
  context.RealtimePop(); // depth 0
  expect_rt(false);
  context.RealtimePush(); // depth 1
  expect_rt(true);
}

TEST_F(TestRtsanContext, IsNotBypassedAfterDefaultConstruction) {
  __rtsan::Context context{};
  EXPECT_THAT(context.IsBypassed(), Eq(false));
}

TEST_F(TestRtsanContext, IsBypassedAfterBypassPush) {
  __rtsan::Context context{};
  context.BypassPush();
  EXPECT_THAT(context.IsBypassed(), Eq(true));
}

TEST_F(TestRtsanContext, BypassedStateIsStatefullyTracked) {
  __rtsan::Context context{};
  auto const expect_bypassed = [&context](bool is_bypassed) {
    EXPECT_THAT(context.IsBypassed(), Eq(is_bypassed));
  };
  expect_bypassed(false);
  context.BypassPush(); // depth 1
  expect_bypassed(true);
  context.BypassPush(); // depth 2
  expect_bypassed(true);
  context.BypassPop(); // depth 1
  expect_bypassed(true);
  context.BypassPush(); // depth 2
  expect_bypassed(true);
  context.BypassPop(); // depth 1
  expect_bypassed(true);
  context.BypassPop(); // depth 0
  expect_bypassed(false);
  context.BypassPush(); // depth 1
  expect_bypassed(true);
}
