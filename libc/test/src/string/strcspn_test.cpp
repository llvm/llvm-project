//===-- Unittests for strcspn ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strcspn.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibcStrCSpnTest, ComplementarySpanShouldNotGoPastNullTerminator) {
  const char src[5] = {'a', 'b', '\0', 'c', 'd'};
  EXPECT_EQ(__llvm_libc::strcspn(src, "b"), size_t{1});
  EXPECT_EQ(__llvm_libc::strcspn(src, "d"), size_t{2});

  // Same goes for the segment to be searched for.
  const char segment[5] = {'1', '2', '\0', '3', '4'};
  EXPECT_EQ(__llvm_libc::strcspn("123", segment), size_t{0});
}

TEST(LlvmLibcStrCSpnTest, ComplementarySpanForEachIndividualCharacter) {
  const char *src = "12345";
  // The complementary span size should increment accordingly.
  EXPECT_EQ(__llvm_libc::strcspn(src, "1"), size_t{0});
  EXPECT_EQ(__llvm_libc::strcspn(src, "2"), size_t{1});
  EXPECT_EQ(__llvm_libc::strcspn(src, "3"), size_t{2});
  EXPECT_EQ(__llvm_libc::strcspn(src, "4"), size_t{3});
  EXPECT_EQ(__llvm_libc::strcspn(src, "5"), size_t{4});
}

TEST(LlvmLibcStrCSpnTest, ComplementarySpanIsStringLengthIfNoCharacterFound) {
  // Null terminator.
  EXPECT_EQ(__llvm_libc::strcspn("", ""), size_t{0});
  EXPECT_EQ(__llvm_libc::strcspn("", "_"), size_t{0});
  // Single character.
  EXPECT_EQ(__llvm_libc::strcspn("a", "b"), size_t{1});
  // Multiple characters.
  EXPECT_EQ(__llvm_libc::strcspn("abc", "1"), size_t{3});
}

TEST(LlvmLibcStrCSpnTest, DuplicatedCharactersNotPartOfComplementarySpan) {
  // Complementary span should be zero in all these cases.
  EXPECT_EQ(__llvm_libc::strcspn("a", "aa"), size_t{0});
  EXPECT_EQ(__llvm_libc::strcspn("aa", "a"), size_t{0});
  EXPECT_EQ(__llvm_libc::strcspn("aaa", "aa"), size_t{0});
  EXPECT_EQ(__llvm_libc::strcspn("aaaa", "aa"), size_t{0});
  EXPECT_EQ(__llvm_libc::strcspn("aaaa", "baa"), size_t{0});
}
