//===--- rtsan_test_assertions.cpp - Realtime Sanitizer ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of the RealtimeSanitizer runtime library test suite
//
//===----------------------------------------------------------------------===//

#include "rtsan_test_utilities.h"

#include "rtsan/rtsan_assertions.h"

#include <gtest/gtest.h>

class TestRtsanAssertions : public ::testing::Test {
protected:
  void SetUp() override { __rtsan_ensure_initialized(); }
};

TEST_F(TestRtsanAssertions, ExpectNotRealtimeDoesNotDieIfNotInRealtimeContext) {
  __rtsan::Context context{};
  ASSERT_FALSE(context.InRealtimeContext());
  ExpectNotRealtime(context, "fake_function_name");
}

TEST_F(TestRtsanAssertions, ExpectNotRealtimeDiesIfInRealtimeContext) {
  __rtsan::Context context{};
  context.RealtimePush();
  ASSERT_TRUE(context.InRealtimeContext());
  EXPECT_DEATH(ExpectNotRealtime(context, "fake_function_name"), "");
}

TEST_F(TestRtsanAssertions, ExpectNotRealtimeDoesNotDieIfRealtimeButBypassed) {
  __rtsan::Context context{};
  context.RealtimePush();
  context.BypassPush();
  ExpectNotRealtime(context, "fake_function_name");
}
