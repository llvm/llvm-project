//===-- Unittests for sscanf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/vsscanf.h"

#include "test/UnitTest/Test.h"

int call_vsscanf(const char *__restrict buffer, const char *__restrict format,
                 ...) {
  va_list vlist;
  va_start(vlist, format);
  int ret = LIBC_NAMESPACE::vsscanf(buffer, format, vlist);
  va_end(vlist);
  return ret;
}

TEST(LlvmLibcVSScanfTest, SimpleStringConv) {
  int ret_val;
  char buffer[10];
  char buffer2[10];
  ret_val = call_vsscanf("abc123", "abc %s", buffer);
  ASSERT_EQ(ret_val, 1);
  ASSERT_STREQ(buffer, "123");

  ret_val = call_vsscanf("abc123", "%3s %3s", buffer, buffer2);
  ASSERT_EQ(ret_val, 2);
  ASSERT_STREQ(buffer, "abc");
  ASSERT_STREQ(buffer2, "123");

  ret_val = call_vsscanf("abc 123", "%3s%3s", buffer, buffer2);
  ASSERT_EQ(ret_val, 2);
  ASSERT_STREQ(buffer, "abc");
  ASSERT_STREQ(buffer2, "123");
}

TEST(LlvmLibcVSScanfTest, IntConvSimple) {
  int ret_val;
  int result = 0;
  ret_val = call_vsscanf("123", "%d", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 123);

  ret_val = call_vsscanf("456", "%i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 456);

  ret_val = call_vsscanf("789", "%x", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 0x789);

  ret_val = call_vsscanf("012", "%o", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 012);

  ret_val = call_vsscanf("345", "%u", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 345);

  // 288 characters
  ret_val = call_vsscanf("10000000000000000000000000000000"
                         "00000000000000000000000000000000"
                         "00000000000000000000000000000000"
                         "00000000000000000000000000000000"
                         "00000000000000000000000000000000"
                         "00000000000000000000000000000000"
                         "00000000000000000000000000000000"
                         "00000000000000000000000000000000"
                         "00000000000000000000000000000000",
                         "%d", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, int(LIBC_NAMESPACE::cpp::numeric_limits<intmax_t>::max()));

  ret_val = call_vsscanf("Not an integer", "%d", &result);
  EXPECT_EQ(ret_val, 0);
}

TEST(LlvmLibcVSScanfTest, IntConvLengthModifier) {
  int ret_val;
  uintmax_t max_result = 0;
  int int_result = 0;
  char char_result = 0;

  ret_val = call_vsscanf("123", "%ju", &max_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(max_result, uintmax_t(123));

  // Check overflow handling
  ret_val =
      call_vsscanf("999999999999999999999999999999999999", "%ju", &max_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(max_result, LIBC_NAMESPACE::cpp::numeric_limits<uintmax_t>::max());

  // Because this is unsigned, any out of range value should return the maximum,
  // even with a negative sign.
  ret_val =
      call_vsscanf("-999999999999999999999999999999999999", "%ju", &max_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(max_result, LIBC_NAMESPACE::cpp::numeric_limits<uintmax_t>::max());

  ret_val = call_vsscanf("-18446744073709551616", "%ju", &max_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(max_result, LIBC_NAMESPACE::cpp::numeric_limits<uintmax_t>::max());

  // But any number below the maximum should have the - sign applied.
  ret_val = call_vsscanf("-1", "%ju", &max_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(max_result, uintmax_t(-1));

  ret_val = call_vsscanf("-1", "%u", &int_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(int_result, -1);

  max_result = 0xff00ff00ff00ff00;
  char_result = 0x6f;

  // Overflows for sizes larger than the maximum are handled by casting.
  ret_val = call_vsscanf("8589967360", "%d", &int_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(int_result, int(8589967360)); // 2^33 + 2^15

  // Check that the adjacent values weren't touched by the overflow.
  ASSERT_EQ(max_result, uintmax_t(0xff00ff00ff00ff00));
  ASSERT_EQ(char_result, char(0x6f));

  ret_val = call_vsscanf("-8589967360", "%d", &int_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(int_result, int(-8589967360));
  ASSERT_EQ(max_result, uintmax_t(0xff00ff00ff00ff00));
  ASSERT_EQ(char_result, char(0x6f));

  ret_val = call_vsscanf("25", "%hhd", &char_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(char_result, char(25));
}

TEST(LlvmLibcVSScanfTest, IntConvBaseSelection) {
  int ret_val;
  int result = 0;
  ret_val = call_vsscanf("0xabc123", "%i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 0xabc123);

  ret_val = call_vsscanf("0456", "%i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 0456);

  ret_val = call_vsscanf("0999", "%i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 0);

  ret_val = call_vsscanf("123abc456", "%i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 123);
}
