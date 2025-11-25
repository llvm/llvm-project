//===-- Unittests for strpbrk ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strpbrk.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibcStrPBrkTest, EmptyStringShouldReturnNullptr) {
  // The search should not include the null terminator.
  EXPECT_STREQ(LIBC_NAMESPACE::strpbrk("", ""), nullptr);
  EXPECT_STREQ(LIBC_NAMESPACE::strpbrk("_", ""), nullptr);
  EXPECT_STREQ(LIBC_NAMESPACE::strpbrk("", "_"), nullptr);
}

TEST(LlvmLibcStrPBrkTest, ShouldNotFindAnythingAfterNullTerminator) {
  const char src[4] = {'a', 'b', '\0', 'c'};
  EXPECT_STREQ(LIBC_NAMESPACE::strpbrk(src, "c"), nullptr);
}

TEST(LlvmLibcStrPBrkTest, ShouldReturnNullptrIfNoCharactersFound) {
  EXPECT_STREQ(LIBC_NAMESPACE::strpbrk("12345", "abcdef"), nullptr);
}

TEST(LlvmLibcStrPBrkTest, FindsFirstCharacter) {
  const char *src = "12345";
  EXPECT_STREQ(LIBC_NAMESPACE::strpbrk(src, "1"), "12345");
  EXPECT_STREQ(LIBC_NAMESPACE::strpbrk(src, "-1"), "12345");
  EXPECT_STREQ(LIBC_NAMESPACE::strpbrk(src, "1_"), "12345");
  EXPECT_STREQ(LIBC_NAMESPACE::strpbrk(src, "f1_"), "12345");
  ASSERT_STREQ(src, "12345");
}

TEST(LlvmLibcStrPBrkTest, FindsMiddleCharacter) {
  const char *src = "12345";
  EXPECT_STREQ(LIBC_NAMESPACE::strpbrk(src, "3"), "345");
  EXPECT_STREQ(LIBC_NAMESPACE::strpbrk(src, "?3"), "345");
  EXPECT_STREQ(LIBC_NAMESPACE::strpbrk(src, "3F"), "345");
  EXPECT_STREQ(LIBC_NAMESPACE::strpbrk(src, "z3_"), "345");
  ASSERT_STREQ(src, "12345");
}

TEST(LlvmLibcStrPBrkTest, FindsLastCharacter) {
  const char *src = "12345";
  EXPECT_STREQ(LIBC_NAMESPACE::strpbrk(src, "5"), "5");
  EXPECT_STREQ(LIBC_NAMESPACE::strpbrk(src, "r5"), "5");
  EXPECT_STREQ(LIBC_NAMESPACE::strpbrk(src, "59"), "5");
  EXPECT_STREQ(LIBC_NAMESPACE::strpbrk(src, "n5_"), "5");
  ASSERT_STREQ(src, "12345");
}

TEST(LlvmLibcStrPBrkTest, FindsFirstOfRepeated) {
  EXPECT_STREQ(LIBC_NAMESPACE::strpbrk("A,B,C,D", ","), ",B,C,D");
}

TEST(LlvmLibcStrPBrkTest, FindsFirstInBreakset) {
  EXPECT_STREQ(LIBC_NAMESPACE::strpbrk("12345", "34"), "345");
}

TEST(LlvmLibcStrPBrkTest, TopBitSet) {
  EXPECT_STREQ(LIBC_NAMESPACE::strpbrk("hello\x80world", "\x80 "), "\x80world");
}
