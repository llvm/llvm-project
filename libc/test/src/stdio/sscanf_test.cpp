//===-- Unittests for sscanf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/sscanf.h"

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
