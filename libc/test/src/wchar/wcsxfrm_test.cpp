//===-- Unittests for wcsxfrm --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcsxfrm.h"
#include "test/UnitTest/Test.h"

// TODO: Remove this once the test framework supports direct wchar_t
// comparisons in EXPECT_EQ.
#define EXPECT_WCHAR_EQ(ACTUAL, EXPECTED)                                      \
  EXPECT_EQ(static_cast<int>(ACTUAL), static_cast<int>(EXPECTED))

TEST(LlvmLibcWCSXfrmTest, EmptyString) {
  wchar_t dest[8];
  size_t result = LIBC_NAMESPACE::wcsxfrm(dest, L"", 8);

  EXPECT_EQ(result, size_t(0));
  EXPECT_WCHAR_EQ(dest[0], L'\0');
}

TEST(LlvmLibcWCSXfrmTest, NullDestinationWhenCountIsZero) {
  size_t result = LIBC_NAMESPACE::wcsxfrm(nullptr, L"abc", 0);
  EXPECT_EQ(result, size_t(3));
}

TEST(LlvmLibcWCSXfrmTest, CopiesWholeStringWhenBufferIsLargeEnough) {
  wchar_t dest[16];
  size_t result = LIBC_NAMESPACE::wcsxfrm(dest, L"hello", 16);

  EXPECT_EQ(result, size_t(5));
  EXPECT_WCHAR_EQ(dest[0], L'h');
  EXPECT_WCHAR_EQ(dest[1], L'e');
  EXPECT_WCHAR_EQ(dest[2], L'l');
  EXPECT_WCHAR_EQ(dest[3], L'l');
  EXPECT_WCHAR_EQ(dest[4], L'o');
  EXPECT_WCHAR_EQ(dest[5], L'\0');
}

TEST(LlvmLibcWCSXfrmTest, ExactFitIncludingNullTerminator) {
  wchar_t dest[6];
  size_t result = LIBC_NAMESPACE::wcsxfrm(dest, L"hello", 6);

  EXPECT_EQ(result, size_t(5));
  EXPECT_WCHAR_EQ(dest[0], L'h');
  EXPECT_WCHAR_EQ(dest[1], L'e');
  EXPECT_WCHAR_EQ(dest[2], L'l');
  EXPECT_WCHAR_EQ(dest[3], L'l');
  EXPECT_WCHAR_EQ(dest[4], L'o');
  EXPECT_WCHAR_EQ(dest[5], L'\0');
}

TEST(LlvmLibcWCSXfrmTest, TruncatesAndNullTerminates) {
  wchar_t dest[4];
  size_t result = LIBC_NAMESPACE::wcsxfrm(dest, L"hello", 4);

  EXPECT_EQ(result, size_t(5));
  EXPECT_WCHAR_EQ(dest[0], L'h');
  EXPECT_WCHAR_EQ(dest[1], L'e');
  EXPECT_WCHAR_EQ(dest[2], L'l');
  EXPECT_WCHAR_EQ(dest[3], L'\0');
}

TEST(LlvmLibcWCSXfrmTest, BufferSizeOneWritesOnlyNullTerminator) {
  wchar_t dest[1];
  dest[0] = L'x';

  size_t result = LIBC_NAMESPACE::wcsxfrm(dest, L"hello", 1);

  EXPECT_EQ(result, size_t(5));
  EXPECT_WCHAR_EQ(dest[0], L'\0');
}

TEST(LlvmLibcWCSXfrmTest, DoesNotWriteWhenCountIsZero) {
  wchar_t dest[4] = {L'x', L'y', L'z', L'\0'};

  size_t result = LIBC_NAMESPACE::wcsxfrm(dest, L"hello", 0);

  EXPECT_EQ(result, size_t(5));
  EXPECT_WCHAR_EQ(dest[0], L'x');
  EXPECT_WCHAR_EQ(dest[1], L'y');
  EXPECT_WCHAR_EQ(dest[2], L'z');
  EXPECT_WCHAR_EQ(dest[3], L'\0');
}

TEST(LlvmLibcWCSXfrmTest, WideCharactersAreHandledCorrectly) {
  wchar_t dest[8];
  const wchar_t src[] = {L'A', L'\u03A9', L'\u2603', L'\0'};

  size_t result = LIBC_NAMESPACE::wcsxfrm(dest, src, 8);

  EXPECT_EQ(result, size_t(3));
  EXPECT_WCHAR_EQ(dest[0], L'A');
  EXPECT_WCHAR_EQ(dest[1], L'\u03A9');
  EXPECT_WCHAR_EQ(dest[2], L'\u2603');
  EXPECT_WCHAR_EQ(dest[3], L'\0');
}
