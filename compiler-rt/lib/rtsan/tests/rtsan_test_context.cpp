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

class TestRtsanContext : public ::testing::Test {
protected:
  void SetUp() override { __rtsan_ensure_initialized(); }
};

TEST_F(TestRtsanContext, ExpectNotRealtimeDoesNotDieBeforeRealtimePush) {
  __rtsan::Context context{};
  ExpectNotRealtime(context, "do_some_stuff");
}

TEST_F(TestRtsanContext, ExpectNotRealtimeDoesNotDieAfterPushAndPop) {
  __rtsan::Context context{};
  context.RealtimePush();
  context.RealtimePop();
  ExpectNotRealtime(context, "do_some_stuff");
}

TEST_F(TestRtsanContext, ExpectNotRealtimeDiesAfterRealtimePush) {
  __rtsan::Context context{};

  context.RealtimePush();
  EXPECT_DEATH(ExpectNotRealtime(context, "do_some_stuff"), "");
}

TEST_F(TestRtsanContext,
       ExpectNotRealtimeDiesAfterRealtimeAfterMorePushesThanPops) {
  __rtsan::Context context{};

  context.RealtimePush();
  context.RealtimePush();
  context.RealtimePush();
  context.RealtimePop();
  context.RealtimePop();
  EXPECT_DEATH(ExpectNotRealtime(context, "do_some_stuff"), "");
}

TEST_F(TestRtsanContext, ExpectNotRealtimeDoesNotDieAfterBypassPush) {
  __rtsan::Context context{};

  context.RealtimePush();
  context.BypassPush();
  ExpectNotRealtime(context, "do_some_stuff");
}

TEST_F(TestRtsanContext,
       ExpectNotRealtimeDoesNotDieIfBypassDepthIsGreaterThanZero) {
  __rtsan::Context context{};

  context.RealtimePush();
  context.BypassPush();
  context.BypassPush();
  context.BypassPush();
  context.BypassPop();
  context.BypassPop();
  ExpectNotRealtime(context, "do_some_stuff");
  context.BypassPop();
  EXPECT_DEATH(ExpectNotRealtime(context, "do_some_stuff"), "");
}
