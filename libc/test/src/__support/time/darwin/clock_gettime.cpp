//===-- unit tests for darwin's time utilities --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/time/clock_gettime.h"
#include "src/__support/CPP/expected.h"
#include "test/UnitTest/Test.h"

template <class T, class E>
using expected = LIBC_NAMESPACE::cpp::expected<T, E>;

TEST(LlvmLibcSupportDarwinClockGetTime, BasicGetTime) {
  struct timespec ts;
  auto result = LIBC_NAMESPACE::internal::clock_gettime(CLOCK_REALTIME, &ts);
  ASSERT_TRUE(result.has_value());
}
