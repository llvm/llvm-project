//===-- Unittests for wcspbrk ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/wchar_t.h"
#include "src/wchar/wcspbrk.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWCSPBrkTest, EmptyStringShouldReturnNullptr) {
  // The search should not include the null terminator.
  EXPECT_EQ(LIBC_NAMESPACE::wcspbrk(L"", L""), nullptr);
  EXPECT_EQ(LIBC_NAMESPACE::wcspbrk(L"_", L""), nullptr);
  EXPECT_EQ(LIBC_NAMESPACE::wcspbrk(L"", L"_"), nullptr);
}

TEST(LlvmLibcWCSPBrkTest, ShouldNotFindAnythingAfterNullTerminator) {
  const wchar_t src[4] = {'a', 'b', '\0', 'c'};
  EXPECT_EQ(LIBC_NAMESPACE::wcspbrk(src, L"c"), nullptr);
}

TEST(LlvmLibcWCSPBrkTest, ShouldReturnNullptrIfNoCharactersFound) {
  EXPECT_EQ(LIBC_NAMESPACE::wcspbrk(L"12345", L"abcdef"), nullptr);
}

TEST(LlvmLibcWCSPBrkTest, FindsFirstCharacter) {
  const wchar_t *src = L"12345";
  EXPECT_EQ(LIBC_NAMESPACE::wcspbrk(src, L"1"), src);
  EXPECT_EQ(LIBC_NAMESPACE::wcspbrk(src, L"-1"), src);
  EXPECT_EQ(LIBC_NAMESPACE::wcspbrk(src, L"1_"), src);
  EXPECT_EQ(LIBC_NAMESPACE::wcspbrk(src, L"f1_"), src);
}

TEST(LlvmLibcWCSPBrkTest, FindsMiddleCharacter) {
  const wchar_t *src = L"12345";
  EXPECT_EQ(LIBC_NAMESPACE::wcspbrk(src, L"3"), src + 2);
  EXPECT_EQ(LIBC_NAMESPACE::wcspbrk(src, L"?3"), src + 2);
  EXPECT_EQ(LIBC_NAMESPACE::wcspbrk(src, L"3F"), src + 2);
  EXPECT_EQ(LIBC_NAMESPACE::wcspbrk(src, L"z3_"), src + 2);
}

TEST(LlvmLibcWCSPBrkTest, FindsLastCharacter) {
  const wchar_t *src = L"12345";
  EXPECT_EQ(LIBC_NAMESPACE::wcspbrk(src, L"5"), src + 4);
  EXPECT_EQ(LIBC_NAMESPACE::wcspbrk(src, L"r5"), src + 4);
  EXPECT_EQ(LIBC_NAMESPACE::wcspbrk(src, L"59"), src + 4);
  EXPECT_EQ(LIBC_NAMESPACE::wcspbrk(src, L"n5_"), src + 4);
}

TEST(LlvmLibcWCSPBrkTest, FindsFirstOfRepeated) {
  const wchar_t *src = L"A,B,C,D";
  EXPECT_EQ(LIBC_NAMESPACE::wcspbrk(src, L","), src + 1);
}

TEST(LlvmLibcWCSPBrkTest, FindsFirstInBreakset) {
  const wchar_t *src = L"12345";
  EXPECT_EQ(LIBC_NAMESPACE::wcspbrk(src, L"34"), src + 2);
  EXPECT_EQ(LIBC_NAMESPACE::wcspbrk(src, L"43"), src + 2);
}

#if defined(LIBC_ADD_NULL_CHECKS)
TEST(LlvmLibcWCSPBrkTest, NullptrCrash) {
  // Passing in a nullptr should crash the program.
  EXPECT_DEATH([] { LIBC_NAMESPACE::wcspbrk(L"aaaaaaaaaaaaaa", nullptr); },
               WITH_SIGNAL(-1));
  EXPECT_DEATH([] { LIBC_NAMESPACE::wcspbrk(nullptr, L"aaaaaaaaaaaaaa"); },
               WITH_SIGNAL(-1));
}
#endif // LIBC_ADD_NULL_CHECKS
