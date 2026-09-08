//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for wcscasecmp.
///
//===----------------------------------------------------------------------===//

#include "src/wchar/wcscasecmp.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWcscasecmpTest, EmptyStrings) {
  EXPECT_EQ(LIBC_NAMESPACE::wcscasecmp(L"", L""), 0);
}

TEST(LlvmLibcWcscasecmpTest, EmptyVsNonEmptyStrings) {
  EXPECT_LT(LIBC_NAMESPACE::wcscasecmp(L"", L"a"), 0);
  EXPECT_GT(LIBC_NAMESPACE::wcscasecmp(L"a", L""), 0);
}

TEST(LlvmLibcWcscasecmpTest, Substrings) {
  EXPECT_GT(LIBC_NAMESPACE::wcscasecmp(L"abc", L"ab"), 0);
  EXPECT_GT(LIBC_NAMESPACE::wcscasecmp(L"abc", L"a"), 0);
  EXPECT_GT(LIBC_NAMESPACE::wcscasecmp(L"abc", L""), 0);

  EXPECT_LT(LIBC_NAMESPACE::wcscasecmp(L"ab", L"abc"), 0);
  EXPECT_LT(LIBC_NAMESPACE::wcscasecmp(L"a", L"abc"), 0);
  EXPECT_LT(LIBC_NAMESPACE::wcscasecmp(L"", L"abc"), 0);
}

TEST(LlvmLibcWcscasecmpTest, MatchingStringsIgnoreCase) {
  EXPECT_EQ(LIBC_NAMESPACE::wcscasecmp(L"abc", L"abc"), 0);

  EXPECT_EQ(LIBC_NAMESPACE::wcscasecmp(L"Abc", L"abc"), 0);
  EXPECT_EQ(LIBC_NAMESPACE::wcscasecmp(L"aBc", L"abc"), 0);
  EXPECT_EQ(LIBC_NAMESPACE::wcscasecmp(L"abC", L"abc"), 0);
  EXPECT_EQ(LIBC_NAMESPACE::wcscasecmp(L"ABC", L"abc"), 0);

  EXPECT_EQ(LIBC_NAMESPACE::wcscasecmp(L"abc", L"Abc"), 0);
  EXPECT_EQ(LIBC_NAMESPACE::wcscasecmp(L"abc", L"aBc"), 0);
  EXPECT_EQ(LIBC_NAMESPACE::wcscasecmp(L"abc", L"abC"), 0);
  EXPECT_EQ(LIBC_NAMESPACE::wcscasecmp(L"abc", L"ABC"), 0);
}

TEST(LlvmLibcWcscasecmpTest, NonMatchingStrings) {
  EXPECT_GT(LIBC_NAMESPACE::wcscasecmp(L"Xbc", L"abc"), 0);
  EXPECT_GT(LIBC_NAMESPACE::wcscasecmp(L"aXc", L"abc"), 0);
  EXPECT_GT(LIBC_NAMESPACE::wcscasecmp(L"abX", L"abc"), 0);

  EXPECT_LT(LIBC_NAMESPACE::wcscasecmp(L"abc", L"Xbc"), 0);
  EXPECT_LT(LIBC_NAMESPACE::wcscasecmp(L"abc", L"aXc"), 0);
  EXPECT_LT(LIBC_NAMESPACE::wcscasecmp(L"abc", L"abX"), 0);
}

#if defined(LIBC_ADD_NULL_CHECKS)

TEST(LlvmLibcWcscasecmpTest, CrashOnNullPtr) {
  EXPECT_DEATH([] { LIBC_NAMESPACE::wcscasecmp(L"hello", nullptr); },
               WITH_SIGNAL(-1));
  EXPECT_DEATH([] { LIBC_NAMESPACE::wcscasecmp(nullptr, L"hello"); },
               WITH_SIGNAL(-1));
}

#endif // LIBC_ADD_NULL_CHECKS
