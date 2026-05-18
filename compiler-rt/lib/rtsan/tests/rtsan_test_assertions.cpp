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

#include "sanitizer_common/sanitizer_stacktrace.h"

#include <gmock/gmock.h>

using namespace __sanitizer;
using namespace __rtsan;

class TestRtsanAssertions : public ::testing::Test {
protected:
  void SetUp() override { __rtsan_ensure_initialized(); }
};

static void ExpectViolationAction(Context &context,
                                  bool expect_violation_callback) {
  ::testing::MockFunction<void(const BufferedStackTrace &stack,
                               const DiagnosticsInfo &info)>
      mock_on_violation;
  EXPECT_CALL(mock_on_violation, Call).Times(expect_violation_callback ? 1 : 0);
  DiagnosticsInfo info{};
  ExpectNotRealtime(context, info, mock_on_violation.AsStdFunction());
}

TEST_F(TestRtsanAssertions,
       ExpectNotRealtimeDoesNotCallViolationActionIfNotInRealtimeContext) {
  Context context{};
  ASSERT_FALSE(context.InRealtimeContext());
  ExpectViolationAction(context, false);
}

TEST_F(TestRtsanAssertions,
       ExpectNotRealtimeCallsViolationActionIfInRealtimeContext) {
  Context context{};
  context.RealtimePush();
  ASSERT_TRUE(context.InRealtimeContext());
  ExpectViolationAction(context, true);
}

TEST_F(TestRtsanAssertions,
       ExpectNotRealtimeDoesNotCallViolationActionIfRealtimeButBypassed) {
  Context context{};
  context.RealtimePush();
  context.BypassPush();
  ASSERT_TRUE(context.IsBypassed());
  ExpectViolationAction(context, false);
}
