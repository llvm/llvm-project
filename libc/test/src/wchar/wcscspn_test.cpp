//===-- Unittests for wcscspn ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/wchar/wcscspn.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWCSCSpnTest, EmptyStringShouldReturnZeroLengthSpan) {
  // The search should not include the null terminator.
  EXPECT_EQ(LIBC_NAMESPACE::wcscspn(L"", L""), size_t(0));
  EXPECT_EQ(LIBC_NAMESPACE::wcscspn(L"_", L""), size_t(1));
  EXPECT_EQ(LIBC_NAMESPACE::wcscspn(L"", L"_"), size_t(0));
}

TEST(LlvmLibcWCSCSpnTest, ShouldNotSpanAnythingAfterNullTerminator) {
  const wchar_t src[4] = {L'a', L'b', L'\0', L'c'};
  EXPECT_EQ(LIBC_NAMESPACE::wcscspn(src, L"de"), size_t(2));
  EXPECT_EQ(LIBC_NAMESPACE::wcscspn(src, L"c"), size_t(2));

  // Same goes for the segment to be searched for.
  const wchar_t segment[4] = {L'1', L'2', L'\0', L'3'};
  EXPECT_EQ(LIBC_NAMESPACE::wcscspn(L"3", segment), size_t(1));
}

TEST(LlvmLibcWCSCSpnTest, SpanEachIndividualCharacter) {
  const wchar_t *src = L"12345";
  // These are all in the segment.
  EXPECT_EQ(LIBC_NAMESPACE::wcscspn(src, L"1"), size_t(0));
  EXPECT_EQ(LIBC_NAMESPACE::wcscspn(src, L"2"), size_t(1));
  EXPECT_EQ(LIBC_NAMESPACE::wcscspn(src, L"3"), size_t(2));
  EXPECT_EQ(LIBC_NAMESPACE::wcscspn(src, L"4"), size_t(3));
  EXPECT_EQ(LIBC_NAMESPACE::wcscspn(src, L"5"), size_t(4));
}

TEST(LlvmLibcWCSCSpnTest, UnmatchedCharacterShouldReturnLength) {
  EXPECT_EQ(LIBC_NAMESPACE::wcscspn(L"a", L"b"), size_t(1));
  EXPECT_EQ(LIBC_NAMESPACE::wcscspn(L"abcdef", L"1"), size_t(6));
  EXPECT_EQ(LIBC_NAMESPACE::wcscspn(L"123", L"4"), size_t(3));
}

TEST(LlvmLibcWCSCSpnTest, NonSequentialCharactersShouldNotSpan) {
  const wchar_t *src = L"abc456789";
  EXPECT_EQ(LIBC_NAMESPACE::wcscspn(src, L"_1_abc_2_def_3_"), size_t(0));
  EXPECT_EQ(LIBC_NAMESPACE::wcscspn(src, L"67__34xyz12"), size_t(3));
}

TEST(LlvmLibcWCSCSpnTest, ReverseCharacters) {
  // These are all in the string.
  EXPECT_EQ(LIBC_NAMESPACE::wcscspn(L"12345", L"54321"), size_t(0));
  // 1 is not in the span.
  EXPECT_EQ(LIBC_NAMESPACE::wcscspn(L"12345", L"432"), size_t(1));
  // 1 is in the span.
  EXPECT_EQ(LIBC_NAMESPACE::wcscspn(L"12345", L"51"), size_t(0));
}

TEST(LlvmLibcWCSCSpnTest, DuplicatedCharactersToBeSearchedForShouldStillMatch) {
  EXPECT_EQ(LIBC_NAMESPACE::wcscspn(L"a", L"aa"), size_t(0));
  EXPECT_EQ(LIBC_NAMESPACE::wcscspn(L"aa", L"aa"), size_t(0));
  EXPECT_EQ(LIBC_NAMESPACE::wcscspn(L"aaa", L"bb"), size_t(3));
  EXPECT_EQ(LIBC_NAMESPACE::wcscspn(L"aaaa", L"bb"), size_t(4));
}
