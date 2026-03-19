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

using namespace __rtsan;
using namespace ::testing;

class TestRtsanContext : public Test {
protected:
  void SetUp() override { __rtsan_ensure_initialized(); }
};

TEST_F(TestRtsanContext, IsNotRealtimeAfterDefaultConstruction) {
  Context context{};
  EXPECT_THAT(context.InRealtimeContext(), Eq(false));
}

TEST_F(TestRtsanContext, IsRealtimeAfterRealtimePush) {
  Context context{};
  context.RealtimePush();
  EXPECT_THAT(context.InRealtimeContext(), Eq(true));
}

TEST_F(TestRtsanContext, IsNotRealtimeAfterRealtimePushAndPop) {
  Context context{};
  context.RealtimePush();
  ASSERT_THAT(context.InRealtimeContext(), Eq(true));
  context.RealtimePop();
  EXPECT_THAT(context.InRealtimeContext(), Eq(false));
}

TEST_F(TestRtsanContext, RealtimeContextStateIsStatefullyTracked) {
  Context context{};
  auto const ExpectRealtime = [&context](bool is_rt) {
    EXPECT_THAT(context.InRealtimeContext(), Eq(is_rt));
  };
  ExpectRealtime(false);
  context.RealtimePush(); // depth 1
  ExpectRealtime(true);
  context.RealtimePush(); // depth 2
  ExpectRealtime(true);
  context.RealtimePop(); // depth 1
  ExpectRealtime(true);
  context.RealtimePush(); // depth 2
  ExpectRealtime(true);
  context.RealtimePop(); // depth 1
  ExpectRealtime(true);
  context.RealtimePop(); // depth 0
  ExpectRealtime(false);
  context.RealtimePush(); // depth 1
  ExpectRealtime(true);
}

TEST_F(TestRtsanContext, IsNotBypassedAfterDefaultConstruction) {
  Context context{};
  EXPECT_THAT(context.IsBypassed(), Eq(false));
}

TEST_F(TestRtsanContext, IsBypassedAfterBypassPush) {
  Context context{};
  context.BypassPush();
  EXPECT_THAT(context.IsBypassed(), Eq(true));
}

TEST_F(TestRtsanContext, BypassedStateIsStatefullyTracked) {
  Context context{};
  auto const ExpectBypassed = [&context](bool is_bypassed) {
    EXPECT_THAT(context.IsBypassed(), Eq(is_bypassed));
  };
  ExpectBypassed(false);
  context.BypassPush(); // depth 1
  ExpectBypassed(true);
  context.BypassPush(); // depth 2
  ExpectBypassed(true);
  context.BypassPop(); // depth 1
  ExpectBypassed(true);
  context.BypassPush(); // depth 2
  ExpectBypassed(true);
  context.BypassPop(); // depth 1
  ExpectBypassed(true);
  context.BypassPop(); // depth 0
  ExpectBypassed(false);
  context.BypassPush(); // depth 1
  ExpectBypassed(true);
}
