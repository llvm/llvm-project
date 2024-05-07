//===--- radsan_test_context.cpp - Realtime Sanitizer --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "radsan_test_utilities.h"

#include "radsan_context.h"

TEST(TestRadsanContext, canCreateContext) { auto context = radsan::Context{}; }

TEST(TestRadsanContext, expectNotRealtimeDoesNotDieBeforeRealtimePush) {
  auto context = radsan::Context{};
  context.expectNotRealtime("do_some_stuff");
}

TEST(TestRadsanContext, expectNotRealtimeDoesNotDieAfterPushAndPop) {
  auto context = radsan::Context{};
  context.realtimePush();
  context.realtimePop();
  context.expectNotRealtime("do_some_stuff");
}

TEST(TestRadsanContext, expectNotRealtimeDiesAfterRealtimePush) {
  auto context = radsan::Context{};

  context.realtimePush();
  EXPECT_DEATH(context.expectNotRealtime("do_some_stuff"), "");
}

TEST(TestRadsanContext,
     expectNotRealtimeDiesAfterRealtimeAfterMorePushesThanPops) {
  auto context = radsan::Context{};

  context.realtimePush();
  context.realtimePush();
  context.realtimePush();
  context.realtimePop();
  context.realtimePop();
  EXPECT_DEATH(context.expectNotRealtime("do_some_stuff"), "");
}

TEST(TestRadsanContext, expectNotRealtimeDoesNotDieAfterBypassPush) {
  auto context = radsan::Context{};

  context.realtimePush();
  context.bypassPush();
  context.expectNotRealtime("do_some_stuff");
}

TEST(TestRadsanContext,
     expectNotRealtimeDoesNotDieIfBypassDepthIsGreaterThanZero) {
  auto context = radsan::Context{};

  context.realtimePush();
  context.bypassPush();
  context.bypassPush();
  context.bypassPush();
  context.bypassPop();
  context.bypassPop();
  context.expectNotRealtime("do_some_stuff");
  context.bypassPop();
  EXPECT_DEATH(context.expectNotRealtime("do_some_stuff"), "");
}
