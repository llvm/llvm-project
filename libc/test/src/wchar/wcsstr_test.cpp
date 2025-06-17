//===-- Unittests for wcsstr ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/wchar_t.h"
#include "src/wchar/wcsstr.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWCSStrTest, NeedleNotInHaystack) {
  // Should return nullptr if string is not found.
  const wchar_t *haystack = L"12345";
  const wchar_t *needle = L"a";
  ASSERT_EQ(LIBC_NAMESPACE::wcsstr(haystack, needle), nullptr);
}

TEST(LlvmLibcWCSStrTest, NeedleIsEmptyString) {
  // Should return pointer to first character if needle is empty.
  const wchar_t *haystack = L"12345";
  const wchar_t *needle = L"";
  ASSERT_EQ(LIBC_NAMESPACE::wcsstr(haystack, needle), haystack);
}

TEST(LlvmLibcWCSStrTest, HaystackIsEmptyString) {
  // Should return nullptr since haystack is empty.
  const wchar_t *needle = L"12345";
  const wchar_t *haystack = L"";
  ASSERT_EQ(LIBC_NAMESPACE::wcsstr(haystack, needle), nullptr);
}

TEST(LlvmLibcWCSStrTest, HaystackAndNeedleAreEmptyStrings) {
  // Should point to haystack since needle is empty.
  const wchar_t *needle = L"";
  const wchar_t *haystack = L"";
  ASSERT_EQ(LIBC_NAMESPACE::wcsstr(haystack, needle), haystack);
}

TEST(LlvmLibcWCSStrTest, HaystackAndNeedleAreSingleCharacters) {
  const wchar_t *haystack = L"a";
  // Should point to haystack.
  ASSERT_EQ(LIBC_NAMESPACE::wcsstr(haystack, L"a"), haystack);
  // Should return nullptr.
  ASSERT_EQ(LIBC_NAMESPACE::wcsstr(haystack, L"b"), nullptr);
}

TEST(LlvmLibcWCSStrTest, NeedleEqualToHaystack) {
  const wchar_t *haystack = L"12345";
  // Should point to haystack.
  ASSERT_EQ(LIBC_NAMESPACE::wcsstr(haystack, L"12345"), haystack);
}

TEST(LlvmLibcWCSStrTest, NeedleLargerThanHaystack) {
  const wchar_t *haystack = L"123";
  // Should return nullptr.
  ASSERT_EQ(LIBC_NAMESPACE::wcsstr(haystack, L"12345"), nullptr);
}

TEST(LlvmLibcWCSStrTest, NeedleAtBeginning) {
  const wchar_t *haystack = L"12345";
  const wchar_t *needle = L"12";
  ASSERT_EQ(LIBC_NAMESPACE::wcsstr(haystack, needle), haystack);
}

TEST(LlvmLibcWCSStrTest, NeedleInMiddle) {
  const wchar_t *haystack = L"abcdefghi";
  const wchar_t *needle = L"def";
  ASSERT_EQ(LIBC_NAMESPACE::wcsstr(haystack, needle), haystack + 3);
}

TEST(LlvmLibcWCSStrTest, NeedleDirectlyBeforeNullTerminator) {
  const wchar_t *haystack = L"abcdefghi";
  const wchar_t *needle = L"ghi";
  ASSERT_EQ(LIBC_NAMESPACE::wcsstr(haystack, needle), haystack + 6);
}

TEST(LlvmLibcWCSStrTest, NeedlePastNullTerminator) {
  const wchar_t haystack[5] = {L'1', L'2', L'\0', L'3', L'4'};
  // Shouldn't find anything after the null terminator.
  ASSERT_EQ(LIBC_NAMESPACE::wcsstr(haystack, /*needle=*/L"3"), nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::wcsstr(haystack, /*needle=*/L"4"), nullptr);
}

TEST(LlvmLibcWCSStrTest, PartialNeedle) {
  const wchar_t *haystack = L"la_ap_lap";
  const wchar_t *needle = L"lap";
  // Shouldn't find la or ap.
  ASSERT_EQ(LIBC_NAMESPACE::wcsstr(haystack, needle), haystack + 6);
}

TEST(LlvmLibcWCSStrTest, MisspelledNeedle) {
  const wchar_t *haystack = L"atalloftwocities...wait, tale";
  const wchar_t *needle = L"tale";
  ASSERT_EQ(LIBC_NAMESPACE::wcsstr(haystack, needle), haystack + 25);
}

TEST(LlvmLibcWCSStrTest, AnagramNeedle) {
  const wchar_t *haystack = L"dgo_ogd_god_odg_gdo_dog";
  const wchar_t *needle = L"dog";
  ASSERT_EQ(LIBC_NAMESPACE::wcsstr(haystack, needle), haystack + 20);
}

TEST(LlvmLibcWCSStrTest, MorphedNeedle) {
  // Changes a single letter in the needle to mismatch with the haystack.
  const wchar_t *haystack = L"once upon a time";
  ASSERT_EQ(LIBC_NAMESPACE::wcsstr(haystack, L"time"), haystack + 12);
  ASSERT_EQ(LIBC_NAMESPACE::wcsstr(haystack, L"lime"), nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::wcsstr(haystack, L"tome"), nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::wcsstr(haystack, L"tire"), nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::wcsstr(haystack, L"timo"), nullptr);
}
