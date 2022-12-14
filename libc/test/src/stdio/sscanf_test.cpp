//===-- Unittests for sscanf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdio/sscanf.h"

#include <stdio.h> // For EOF

#include "utils/UnitTest/Test.h"

TEST(LlvmLibcSScanfTest, SimpleStringConv) {
  int ret_val;
  char buffer[10];
  char buffer2[10];
  ret_val = __llvm_libc::sscanf("abc123", "abc %s", buffer);
  ASSERT_EQ(ret_val, 1);
  ASSERT_STREQ(buffer, "123");

  ret_val = __llvm_libc::sscanf("abc123", "%3s %3s", buffer, buffer2);
  ASSERT_EQ(ret_val, 2);
  ASSERT_STREQ(buffer, "abc");
  ASSERT_STREQ(buffer2, "123");

  ret_val = __llvm_libc::sscanf("abc 123", "%3s%3s", buffer, buffer2);
  ASSERT_EQ(ret_val, 2);
  ASSERT_STREQ(buffer, "abc");
  ASSERT_STREQ(buffer2, "123");
}

TEST(LlvmLibcSScanfTest, IntConvSimple) {
  int ret_val;
  int result = 0;
  ret_val = __llvm_libc::sscanf("123", "%d", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 123);

  ret_val = __llvm_libc::sscanf("456", "%i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 456);

  ret_val = __llvm_libc::sscanf("789", "%x", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 0x789);

  ret_val = __llvm_libc::sscanf("012", "%o", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 012);

  ret_val = __llvm_libc::sscanf("345", "%u", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 345);

  ret_val = __llvm_libc::sscanf("Not an integer", "%d", &result);
  EXPECT_EQ(ret_val, 0);
}

TEST(LlvmLibcSScanfTest, IntConvLengthModifier) {
  int ret_val;
  uintmax_t max_result = 0;
  int int_result = 0;
  char char_result = 0;

  ret_val = __llvm_libc::sscanf("123", "%ju", &max_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(max_result, uintmax_t(123));

  // Check overflow handling
  ret_val = __llvm_libc::sscanf("999999999999999999999999999999999999", "%ju",
                                &max_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(max_result, __llvm_libc::cpp::numeric_limits<uintmax_t>::max());

  // Because this is unsigned, any out of range value should return the maximum,
  // even with a negative sign.
  ret_val = __llvm_libc::sscanf("-999999999999999999999999999999999999", "%ju",
                                &max_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(max_result, __llvm_libc::cpp::numeric_limits<uintmax_t>::max());

  ret_val = __llvm_libc::sscanf("-18446744073709551616", "%ju", &max_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(max_result, __llvm_libc::cpp::numeric_limits<uintmax_t>::max());

  // But any number below the maximum should have the - sign applied.
  ret_val = __llvm_libc::sscanf("-1", "%ju", &max_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(max_result, uintmax_t(-1));

  ret_val = __llvm_libc::sscanf("-1", "%u", &int_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(int_result, -1);

  max_result = 0xff00ff00ff00ff00;
  char_result = 0x6f;

  // Overflows for sizes larger than the maximum are handled by casting.
  ret_val = __llvm_libc::sscanf("8589967360", "%d", &int_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(int_result, int(8589967360)); // 2^33 + 2^15

  // Check that the adjacent values weren't touched by the overflow.
  ASSERT_EQ(max_result, uintmax_t(0xff00ff00ff00ff00));
  ASSERT_EQ(char_result, char(0x6f));

  ret_val = __llvm_libc::sscanf("-8589967360", "%d", &int_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(int_result, int(-8589967360));
  ASSERT_EQ(max_result, uintmax_t(0xff00ff00ff00ff00));
  ASSERT_EQ(char_result, char(0x6f));

  ret_val = __llvm_libc::sscanf("25", "%hhd", &char_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(char_result, char(25));
}

TEST(LlvmLibcSScanfTest, IntConvBaseSelection) {
  int ret_val;
  int result = 0;
  ret_val = __llvm_libc::sscanf("0xabc123", "%i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 0xabc123);

  ret_val = __llvm_libc::sscanf("0456", "%i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 0456);

  ret_val = __llvm_libc::sscanf("0999", "%i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 0);

  ret_val = __llvm_libc::sscanf("123abc456", "%i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 123);
}

TEST(LlvmLibcSScanfTest, IntConvMaxLengthTests) {
  int ret_val;
  int result = 0;

  ret_val = __llvm_libc::sscanf("12", "%1d", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 1);

  ret_val = __llvm_libc::sscanf("-1", "%1d", &result);
  EXPECT_EQ(ret_val, 0);
  EXPECT_EQ(result, 0);

  ret_val = __llvm_libc::sscanf("+1", "%1d", &result);
  EXPECT_EQ(ret_val, 0);
  EXPECT_EQ(result, 0);

  ret_val = __llvm_libc::sscanf("01", "%1d", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 0);

  ret_val = __llvm_libc::sscanf("01", "%1i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 0);

  ret_val = __llvm_libc::sscanf("0x1", "%2i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 0);

  ret_val = __llvm_libc::sscanf("-0x1", "%3i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 0);

  ret_val = __llvm_libc::sscanf("-0x123", "%4i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, -1);

  ret_val = __llvm_libc::sscanf("123456789", "%5i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 12345);

  ret_val = __llvm_libc::sscanf("123456789", "%10i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 123456789);
}

TEST(LlvmLibcSScanfTest, IntConvNoWriteTests) {
  int ret_val;
  // Result shouldn't be used by these tests, but it's safer to have it and
  // check it.
  int result = 0;
  ret_val = __llvm_libc::sscanf("-1", "%*1d", &result);
  EXPECT_EQ(ret_val, 0);
  EXPECT_EQ(result, 0);

  ret_val = __llvm_libc::sscanf("01", "%*1i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 0);

  ret_val = __llvm_libc::sscanf("0x1", "%*2i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 0);

  ret_val = __llvm_libc::sscanf("a", "%*i", &result);
  EXPECT_EQ(ret_val, 0);
  EXPECT_EQ(result, 0);

  ret_val = __llvm_libc::sscanf("123", "%*i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 0);
}

TEST(LlvmLibcSScanfTest, CombinedConv) {
  int ret_val;
  int result = 0;
  char buffer[10];
  ret_val = __llvm_libc::sscanf("123abc", "%i%s", &result, buffer);
  EXPECT_EQ(ret_val, 2);
  EXPECT_EQ(result, 123);
  ASSERT_STREQ(buffer, "abc");

  ret_val = __llvm_libc::sscanf("0xZZZ", "%i%s", &result, buffer);
  EXPECT_EQ(ret_val, 2);
  EXPECT_EQ(result, 0);
  ASSERT_STREQ(buffer, "ZZZ");

  ret_val = __llvm_libc::sscanf("0xZZZ", "%X%s", &result, buffer);
  EXPECT_EQ(ret_val, 2);
  EXPECT_EQ(result, 0);
  ASSERT_STREQ(buffer, "ZZZ");
}
