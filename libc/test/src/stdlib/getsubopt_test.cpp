//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for getsubopt.
///
//===----------------------------------------------------------------------===//

#include "src/stdlib/getsubopt.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcGetsuboptTest, BasicMatching) {
  char *const tokens[] = {const_cast<char *>("ro"), const_cast<char *>("rw"),
                          const_cast<char *>("name"), nullptr};
  char options[] = "ro,name=myhost,rw";
  char *opt_ptr = options;
  char *val = nullptr;

  int idx = LIBC_NAMESPACE::getsubopt(&opt_ptr, tokens, &val);
  EXPECT_EQ(idx, 0);
  EXPECT_EQ(val, static_cast<char *>(nullptr));

  idx = LIBC_NAMESPACE::getsubopt(&opt_ptr, tokens, &val);
  EXPECT_EQ(idx, 2);
  ASSERT_NE(val, static_cast<char *>(nullptr));
  EXPECT_STREQ(val, "myhost");

  idx = LIBC_NAMESPACE::getsubopt(&opt_ptr, tokens, &val);
  EXPECT_EQ(idx, 1);
  EXPECT_EQ(val, static_cast<char *>(nullptr));

  idx = LIBC_NAMESPACE::getsubopt(&opt_ptr, tokens, &val);
  EXPECT_EQ(idx, -1);
}

TEST(LlvmLibcGetsuboptTest, UnmatchedSuboption) {
  char *const tokens[] = {const_cast<char *>("opt1"), nullptr};
  char options[] = "unknown=val,opt1";
  char *opt_ptr = options;
  char *val = nullptr;

  int idx = LIBC_NAMESPACE::getsubopt(&opt_ptr, tokens, &val);
  EXPECT_EQ(idx, -1);
  ASSERT_NE(val, static_cast<char *>(nullptr));
  EXPECT_STREQ(val, "unknown=val");

  idx = LIBC_NAMESPACE::getsubopt(&opt_ptr, tokens, &val);
  EXPECT_EQ(idx, 0);
  EXPECT_EQ(val, static_cast<char *>(nullptr));
}

TEST(LlvmLibcGetsuboptTest, TrailingEquals) {
  char *const tokens[] = {const_cast<char *>("key"),
                          const_cast<char *>("other"), nullptr};
  char options[] = "key=,other";
  char *opt_ptr = options;
  char *val = nullptr;

  int idx = LIBC_NAMESPACE::getsubopt(&opt_ptr, tokens, &val);
  EXPECT_EQ(idx, 0);
  ASSERT_NE(val, static_cast<char *>(nullptr));
  EXPECT_STREQ(val, "");

  idx = LIBC_NAMESPACE::getsubopt(&opt_ptr, tokens, &val);
  EXPECT_EQ(idx, 1);
  EXPECT_EQ(val, static_cast<char *>(nullptr));

  // Trailing equals at the end of the option string without subsequent tokens.
  char options2[] = "key=";
  opt_ptr = options2;
  val = nullptr;
  idx = LIBC_NAMESPACE::getsubopt(&opt_ptr, tokens, &val);
  EXPECT_EQ(idx, 0);
  ASSERT_NE(val, static_cast<char *>(nullptr));
  EXPECT_STREQ(val, "");
}

TEST(LlvmLibcGetsuboptTest, MultipleEquals) {
  char *const tokens[] = {const_cast<char *>("param"), nullptr};
  char options[] = "param=a=b=c";
  char *opt_ptr = options;
  char *val = nullptr;

  int idx = LIBC_NAMESPACE::getsubopt(&opt_ptr, tokens, &val);
  EXPECT_EQ(idx, 0);
  ASSERT_NE(val, static_cast<char *>(nullptr));
  EXPECT_STREQ(val, "a=b=c");
}

TEST(LlvmLibcGetsuboptTest, EmptyOptions) {
  char *const tokens[] = {const_cast<char *>("opt"), nullptr};
  char empty[] = "";
  char *opt_ptr = empty;
  char *val = nullptr;

  EXPECT_EQ(LIBC_NAMESPACE::getsubopt(&opt_ptr, tokens, &val), -1);
}
