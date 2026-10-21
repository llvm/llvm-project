//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for wcsncasecmp.
///
//===----------------------------------------------------------------------===//

#include "hdr/limits_macros.h"
#include "src/__support/wctype_utils.h"
#include "src/wchar/wcsncasecmp.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWcsncasecmpTest, EmptyStrings) {
  EXPECT_EQ(LIBC_NAMESPACE::wcsncasecmp(L"", L"", 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::wcsncasecmp(L"", L"", INT_MAX), 0);
}

TEST(LlvmLibcWcsncasecmpTest, EmptyVsNonEmptyStringsUnlimited) {
  EXPECT_LT(LIBC_NAMESPACE::wcsncasecmp(L"", L"a", INT_MAX), 0);
  EXPECT_GT(LIBC_NAMESPACE::wcsncasecmp(L"a", L"", INT_MAX), 0);
}

TEST(LlvmLibcWcsncasecmpTest, SubstringsUnlimited) {
  EXPECT_GT(LIBC_NAMESPACE::wcsncasecmp(L"aβc", L"aβ", INT_MAX), 0);
  EXPECT_GT(LIBC_NAMESPACE::wcsncasecmp(L"aβc", L"a", INT_MAX), 0);
  EXPECT_GT(LIBC_NAMESPACE::wcsncasecmp(L"aβc", L"", INT_MAX), 0);

  EXPECT_LT(LIBC_NAMESPACE::wcsncasecmp(L"aβ", L"aβc", INT_MAX), 0);
  EXPECT_LT(LIBC_NAMESPACE::wcsncasecmp(L"a", L"aβc", INT_MAX), 0);
  EXPECT_LT(LIBC_NAMESPACE::wcsncasecmp(L"", L"aβc", INT_MAX), 0);
}

TEST(LlvmLibcWcsncasecmpTest, MatchingStringsIgnoreCaseUnlimited) {
  EXPECT_EQ(LIBC_NAMESPACE::wcsncasecmp(L"abc", L"abc", INT_MAX), 0);

  EXPECT_EQ(LIBC_NAMESPACE::wcsncasecmp(L"Abc", L"abc", INT_MAX), 0);
  EXPECT_EQ(LIBC_NAMESPACE::wcsncasecmp(L"aBc", L"abc", INT_MAX), 0);
  EXPECT_EQ(LIBC_NAMESPACE::wcsncasecmp(L"abC", L"abc", INT_MAX), 0);
  EXPECT_EQ(LIBC_NAMESPACE::wcsncasecmp(L"ABC", L"abc", INT_MAX), 0);

  EXPECT_EQ(LIBC_NAMESPACE::wcsncasecmp(L"abc", L"Abc", INT_MAX), 0);
  EXPECT_EQ(LIBC_NAMESPACE::wcsncasecmp(L"abc", L"aBc", INT_MAX), 0);
  EXPECT_EQ(LIBC_NAMESPACE::wcsncasecmp(L"abc", L"abC", INT_MAX), 0);
  EXPECT_EQ(LIBC_NAMESPACE::wcsncasecmp(L"abc", L"ABC", INT_MAX), 0);

#if LIBC_CONF_WCTYPE_MODE == LIBC_WCTYPE_MODE_UTF8
  EXPECT_EQ(LIBC_NAMESPACE::wcsncasecmp(L"Βγδ", L"βγδ", INT_MAX), 0);
  EXPECT_EQ(LIBC_NAMESPACE::wcsncasecmp(L"βΓδ", L"βγδ", INT_MAX), 0);
  EXPECT_EQ(LIBC_NAMESPACE::wcsncasecmp(L"βγΔ", L"βγδ", INT_MAX), 0);
  EXPECT_EQ(LIBC_NAMESPACE::wcsncasecmp(L"ΒΓΔ", L"βγδ", INT_MAX), 0);

  EXPECT_EQ(LIBC_NAMESPACE::wcsncasecmp(L"βγδ", L"Βγδ", INT_MAX), 0);
  EXPECT_EQ(LIBC_NAMESPACE::wcsncasecmp(L"βγδ", L"βΓδ", INT_MAX), 0);
  EXPECT_EQ(LIBC_NAMESPACE::wcsncasecmp(L"βγδ", L"βγΔ", INT_MAX), 0);
  EXPECT_EQ(LIBC_NAMESPACE::wcsncasecmp(L"βγδ", L"ΒΓΔ", INT_MAX), 0);
#endif // LIBC_CONF_WCTYPE_MODE == LIBC_WCTYPE_MODE_UTF8
}

TEST(LlvmLibcWcsncasecmpTest, NonMatchingStringsUnlimited) {
  EXPECT_GT(LIBC_NAMESPACE::wcsncasecmp(L"Δbc", L"abc", INT_MAX), 0);
  EXPECT_GT(LIBC_NAMESPACE::wcsncasecmp(L"aΔc", L"abc", INT_MAX), 0);
  EXPECT_GT(LIBC_NAMESPACE::wcsncasecmp(L"abΔ", L"abc", INT_MAX), 0);

  EXPECT_LT(LIBC_NAMESPACE::wcsncasecmp(L"abc", L"Δbc", INT_MAX), 0);
  EXPECT_LT(LIBC_NAMESPACE::wcsncasecmp(L"abc", L"aΔc", INT_MAX), 0);
  EXPECT_LT(LIBC_NAMESPACE::wcsncasecmp(L"abc", L"abΔ", INT_MAX), 0);
}

TEST(LlvmLibcWcsncasecmpTest, CompareAtMostStartingMatch) {
  EXPECT_EQ(LIBC_NAMESPACE::wcsncasecmp(L"βγδ", L"", 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::wcsncasecmp(L"", L"def", 0), 0);

  EXPECT_EQ(LIBC_NAMESPACE::wcsncasecmp(L"βγδ", L"def", 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::wcsncasecmp(L"βγδ", L"βde", 1), 0);
  EXPECT_EQ(LIBC_NAMESPACE::wcsncasecmp(L"βγδ", L"βγd", 2), 0);
}

#if defined(LIBC_ADD_NULL_CHECKS)

TEST(LlvmLibcWcsncasecmpTest, CrashOnNullPtr) {
  EXPECT_DEATH([] { LIBC_NAMESPACE::wcsncasecmp(L"hello", nullptr, 1); },
               WITH_SIGNAL(-1));
  EXPECT_DEATH([] { LIBC_NAMESPACE::wcsncasecmp(nullptr, L"hello", 1); },
               WITH_SIGNAL(-1));
}

#endif // LIBC_ADD_NULL_CHECKS
