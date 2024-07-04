//===-- Unittests for vsscanf ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/__support/arg_list.h"

#include "src/stdio/vsscanf.h"

#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcVSScanfTest, IntConvSimple) {
  int result = 0;

  auto fn = [](int fmt, ...) {
    va_list vlist;
    va_start(vlist, fmt);
    return LIBC_NAMESPACE::vsscanf("123", "%d", vlist);
  };
  EXPECT_EQ(fn(0, &result), 1);
  EXPECT_EQ(result, 123);

  auto fn = [](int fmt, ...) {
    va_list vlist;
    va_start(vlist, fmt);
    return LIBC_NAMESPACE::vsscanf("456", "%i", vlist);
  };
  EXPECT_EQ(fn(0, &result), 1);
  EXPECT_EQ(result, 456);

  auto fn = [](int fmt, ...) {
    va_list vlist;
    va_start(vlist, fmt);
    return LIBC_NAMESPACE::vsscanf("789", "%x", vlist);
  };
  EXPECT_EQ(fn(0, &result), 1);
  EXPECT_EQ(result, 0x789);

  auto fn = [](int fmt, ...) {
    va_list vlist;
    va_start(vlist, fmt);
    return LIBC_NAMESPACE::vsscanf("012", "%o", vlist);
  };
  EXPECT_EQ(fn(0, &result), 1);
  EXPECT_EQ(result, 012);

  auto fn = [](int fmt, ...) {
    va_list vlist;
    va_start(vlist, fmt);
    return LIBC_NAMESPACE::vsscanf("345", "%u", vlist);
  };
  EXPECT_EQ(fn(0, &result), 1);
  EXPECT_EQ(result, 345);

  auto fn = [](int fmt, ...) {
    va_list vlist;
    va_start(vlist, fmt);
    return LIBC_NAMESPACE::vsscanf("10000000000000000000000000000000"
                                   "00000000000000000000000000000000"
                                   "00000000000000000000000000000000"
                                   "00000000000000000000000000000000"
                                   "00000000000000000000000000000000"
                                   "00000000000000000000000000000000"
                                   "00000000000000000000000000000000"
                                   "00000000000000000000000000000000"
                                   "00000000000000000000000000000000",
                                   "%d", vlist);
  };
  EXPECT_EQ(fn(0, &result), 1);
  EXPECT_EQ(result, int(LIBC_NAMESPACE::cpp::numeric_limits<intmax_t>::max()));

  auto fn = [](int fmt, ...) {
    va_list vlist;
    va_start(vlist, fmt);
    return LIBC_NAMESPACE::vsscanf("Not an integer", "%d", vlist);
  };
  EXPECT_EQ(fn(0, &result), 0);
}
