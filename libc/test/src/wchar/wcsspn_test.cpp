//===-- Unittests for wcsspn ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/wchar/wcsspn.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWCSSpnTest, EmptyStringShouldReturnZeroLengthSpan) {
  // The search should not include the null terminator.
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(L"", L""), size_t{0});
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(L"_", L""), size_t{0});
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(L"", L"_"), size_t{0});
}

TEST(LlvmLibcWCSSpnTest, ShouldNotSpanAnythingAfterNullTerminator) {
  const wchar_t src[4] = {'a', 'b', '\0', 'c'};
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(src, L"ab"), size_t{2});
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(src, L"c"), size_t{0});

  // Same goes for the segment to be searched for.
  const wchar_t segment[4] = {'1', '2', '\0', '3'};
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(L"123", segment), size_t{2});
}

TEST(LlvmLibcWCSSpnTest, SpanEachIndividualCharacter) {
  const wchar_t *src = L"12345";
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(src, L"1"), size_t{1});
  // Since '1' is not within the segment, the span
  // size should remain zero.
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(src, L"2"), size_t{0});
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(src, L"3"), size_t{0});
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(src, L"4"), size_t{0});
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(src, L"5"), size_t{0});
}

TEST(LlvmLibcWCSSpnTest, UnmatchedCharacterShouldNotBeCountedInSpan) {
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(L"a", L"b"), size_t{0});
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(L"abcdef", L"1"), size_t{0});
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(L"123", L"4"), size_t{0});
}

TEST(LlvmLibcWCSSpnTest, SequentialCharactersShouldSpan) {
  const wchar_t *src = L"abcde";
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(src, L"a"), size_t{1});
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(src, L"ab"), size_t{2});
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(src, L"abc"), size_t{3});
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(src, L"abcd"), size_t{4});
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(src, L"abcde"), size_t{5});
  // Same thing for when the roles are reversed.
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(L"abcde", src), size_t{5});
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(L"abcd", src), size_t{4});
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(L"abc", src), size_t{3});
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(L"ab", src), size_t{2});
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(L"a", src), size_t{1});
}

TEST(LlvmLibcWCSSpnTest, NonSequentialCharactersShouldNotSpan) {
  const wchar_t *src = L"123456789";
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(src, L"_1_abc_2_def_3_"), size_t{3});
  // Only spans 4 since '5' is not within the span.
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(src, L"67__34abc12"), size_t{4});
}

TEST(LlvmLibcWCSSpnTest, ReverseCharacters) {
  // Since these are still sequential, this should span.
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(L"12345", L"54321"), size_t{5});
  // Does not span any since '1' is not within the span.
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(L"12345", L"432"), size_t{0});
  // Only spans 1 since '2' is not within the span.
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(L"12345", L"51"), size_t{1});
}

TEST(LlvmLibcWCSSpnTest, DuplicatedCharactersToBeSearchedForShouldStillMatch) {
  // Only a single character, so only spans 1.
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(L"a", L"aa"), size_t{1});
  // This should count once for each 'a' in the source string.
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(L"aa", L"aa"), size_t{2});
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(L"aaa", L"aa"), size_t{3});
  EXPECT_EQ(LIBC_NAMESPACE::wcsspn(L"aaaa", L"aa"), size_t{4});
}
