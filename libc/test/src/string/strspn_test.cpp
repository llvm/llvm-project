//===-- Unittests for strspn ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strspn.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibcStrSpnTest, EmptyStringShouldReturnZeroLengthSpan) {
  // The search should not include the null terminator.
  EXPECT_EQ(LIBC_NAMESPACE::strspn("", ""), size_t{0});
  EXPECT_EQ(LIBC_NAMESPACE::strspn("_", ""), size_t{0});
  EXPECT_EQ(LIBC_NAMESPACE::strspn("", "_"), size_t{0});
}

TEST(LlvmLibcStrSpnTest, ShouldNotSpanAnythingAfterNullTerminator) {
  const char src[4] = {'a', 'b', '\0', 'c'};
  EXPECT_EQ(LIBC_NAMESPACE::strspn(src, "ab"), size_t{2});
  EXPECT_EQ(LIBC_NAMESPACE::strspn(src, "c"), size_t{0});

  // Same goes for the segment to be searched for.
  const char segment[4] = {'1', '2', '\0', '3'};
  EXPECT_EQ(LIBC_NAMESPACE::strspn("123", segment), size_t{2});
}

TEST(LlvmLibcStrSpnTest, SpanEachIndividualCharacter) {
  const char *src = "12345";
  EXPECT_EQ(LIBC_NAMESPACE::strspn(src, "1"), size_t{1});
  // Since '1' is not within the segment, the span
  // size should remain zero.
  EXPECT_EQ(LIBC_NAMESPACE::strspn(src, "2"), size_t{0});
  EXPECT_EQ(LIBC_NAMESPACE::strspn(src, "3"), size_t{0});
  EXPECT_EQ(LIBC_NAMESPACE::strspn(src, "4"), size_t{0});
  EXPECT_EQ(LIBC_NAMESPACE::strspn(src, "5"), size_t{0});
}

TEST(LlvmLibcStrSpnTest, UnmatchedCharacterShouldNotBeCountedInSpan) {
  EXPECT_EQ(LIBC_NAMESPACE::strspn("a", "b"), size_t{0});
  EXPECT_EQ(LIBC_NAMESPACE::strspn("abcdef", "1"), size_t{0});
  EXPECT_EQ(LIBC_NAMESPACE::strspn("123", "4"), size_t{0});
}

TEST(LlvmLibcStrSpnTest, SequentialCharactersShouldSpan) {
  const char *src = "abcde";
  EXPECT_EQ(LIBC_NAMESPACE::strspn(src, "a"), size_t{1});
  EXPECT_EQ(LIBC_NAMESPACE::strspn(src, "ab"), size_t{2});
  EXPECT_EQ(LIBC_NAMESPACE::strspn(src, "abc"), size_t{3});
  EXPECT_EQ(LIBC_NAMESPACE::strspn(src, "abcd"), size_t{4});
  EXPECT_EQ(LIBC_NAMESPACE::strspn(src, "abcde"), size_t{5});
  // Same thing for when the roles are reversed.
  EXPECT_EQ(LIBC_NAMESPACE::strspn("abcde", src), size_t{5});
  EXPECT_EQ(LIBC_NAMESPACE::strspn("abcd", src), size_t{4});
  EXPECT_EQ(LIBC_NAMESPACE::strspn("abc", src), size_t{3});
  EXPECT_EQ(LIBC_NAMESPACE::strspn("ab", src), size_t{2});
  EXPECT_EQ(LIBC_NAMESPACE::strspn("a", src), size_t{1});
}

TEST(LlvmLibcStrSpnTest, NonSequentialCharactersShouldNotSpan) {
  const char *src = "123456789";
  EXPECT_EQ(LIBC_NAMESPACE::strspn(src, "_1_abc_2_def_3_"), size_t{3});
  // Only spans 4 since '5' is not within the span.
  EXPECT_EQ(LIBC_NAMESPACE::strspn(src, "67__34abc12"), size_t{4});
}

TEST(LlvmLibcStrSpnTest, ReverseCharacters) {
  // Since these are still sequential, this should span.
  EXPECT_EQ(LIBC_NAMESPACE::strspn("12345", "54321"), size_t{5});
  // Does not span any since '1' is not within the span.
  EXPECT_EQ(LIBC_NAMESPACE::strspn("12345", "432"), size_t{0});
  // Only spans 1 since '2' is not within the span.
  EXPECT_EQ(LIBC_NAMESPACE::strspn("12345", "51"), size_t{1});
}

TEST(LlvmLibcStrSpnTest, DuplicatedCharactersToBeSearchedForShouldStillMatch) {
  // Only a single character, so only spans 1.
  EXPECT_EQ(LIBC_NAMESPACE::strspn("a", "aa"), size_t{1});
  // This should count once for each 'a' in the source string.
  EXPECT_EQ(LIBC_NAMESPACE::strspn("aa", "aa"), size_t{2});
  EXPECT_EQ(LIBC_NAMESPACE::strspn("aaa", "aa"), size_t{3});
  EXPECT_EQ(LIBC_NAMESPACE::strspn("aaaa", "aa"), size_t{4});
}
