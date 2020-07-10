//===-- Unittests for memchr ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memchr.h"
#include "utils/UnitTest/Test.h"
#include <stddef.h>

// A helper function that calls memchr and abstracts away the explicit cast for
// readability purposes.
const char *call_memchr(const void *src, int c, size_t size) {
  return reinterpret_cast<const char *>(__llvm_libc::memchr(src, c, size));
}

TEST(MemChrTest, FindsCharacterAfterNullTerminator) {
  // memchr should continue searching after a null terminator.
  const size_t size = 5;
  const unsigned char src[size] = {'a', '\0', 'b', 'c', '\0'};
  // Should return 'b', 'c', '\0' even when after null terminator.
  ASSERT_STREQ(call_memchr(src, 'b', size), "bc");
}

TEST(MemChrTest, FindsCharacterInNonNullTerminatedCollection) {
  const size_t size = 3;
  const unsigned char src[size] = {'a', 'b', 'c'};
  // Should return 'b', 'c'.
  const char *ret = call_memchr(src, 'b', size);
  ASSERT_EQ(ret[0], 'b');
  ASSERT_EQ(ret[1], 'c');
}

TEST(MemChrTest, FindsFirstCharacter) {
  const size_t size = 6;
  const unsigned char src[size] = {'a', 'b', 'c', 'd', 'e', '\0'};
  // Should return original array since 'a' is the first character.
  ASSERT_STREQ(call_memchr(src, 'a', size), "abcde");
}

TEST(MemChrTest, FindsMiddleCharacter) {
  const size_t size = 6;
  const unsigned char src[size] = {'a', 'b', 'c', 'd', 'e', '\0'};
  // Should return characters after (and including) 'c'.
  ASSERT_STREQ(call_memchr(src, 'c', size), "cde");
}

TEST(MemChrTest, FindsLastCharacterThatIsNotNullTerminator) {
  const size_t size = 6;
  const unsigned char src[size] = {'a', 'b', 'c', 'd', 'e', '\0'};
  // Should return 'e' and null-terminator.
  ASSERT_STREQ(call_memchr(src, 'e', size), "e");
}

TEST(MemChrTest, FindsNullTerminator) {
  const size_t size = 6;
  const unsigned char src[size] = {'a', 'b', 'c', 'd', 'e', '\0'};
  // Should return null terminator.
  ASSERT_STREQ(call_memchr(src, '\0', size), "");
}

TEST(MemChrTest, CharacterNotWithinStringShouldReturnNullptr) {
  const size_t size = 4;
  const unsigned char src[size] = {'1', '2', '3', '?'};
  // Since 'z' is not within 'characters', should return nullptr.
  ASSERT_STREQ(call_memchr(src, 'z', size), nullptr);
}

TEST(MemChrTest, CharacterNotWithinSizeShouldReturnNullptr) {
  const unsigned char src[5] = {'1', '2', '3', '4', '\0'};
  // Since '4' is not the first or second character, this should return nullptr.
  const size_t size = 2;
  ASSERT_STREQ(call_memchr(src, '4', size), nullptr);
}

TEST(MemChrTest, TheSourceShouldNotChange) {
  const size_t size = 6;
  const unsigned char src[size] = {'a', 'b', 'c', 'd', 'e', '\0'};
  const char *src_copy = reinterpret_cast<const char *>(src);
  // When the character is found, the source string should not change.
  __llvm_libc::memchr(src, 'd', size);
  ASSERT_STREQ(reinterpret_cast<const char *>(src), src_copy);
  // Same case for when the character is not found.
  __llvm_libc::memchr(src, 'z', size);
  ASSERT_STREQ(reinterpret_cast<const char *>(src), src_copy);
}

TEST(MemChrTest, ShouldFindFirstOfDuplicates) {
  const size_t size = 12; // 11 characters + null terminator.
  const char *dups = "abc1def1ghi";
  // 1 is duplicated in 'dups', but it should find the first copy.
  ASSERT_STREQ(call_memchr(dups, '1', size), "1def1ghi");
}

TEST(MemChrTest, EmptyStringShouldOnlyMatchNullTerminator) {
  const size_t size = 1; // Null terminator.
  const char *empty_string = "";
  // Null terminator should match.
  ASSERT_STREQ(call_memchr(empty_string, '\0', size), "");
  // All other characters should not match.
  ASSERT_STREQ(call_memchr(empty_string, 'A', size), nullptr);
  ASSERT_STREQ(call_memchr(empty_string, '9', size), nullptr);
  ASSERT_STREQ(call_memchr(empty_string, '?', size), nullptr);
}

TEST(MemChrTest, SingleRepeatedCharacterShouldReturnFirst) {
  const char *dups = "XXXXX";
  const size_t size = 6; // 5 characters + null terminator.
  // Should return original string since X is first character.
  ASSERT_STREQ(call_memchr(dups, 'X', size), dups);
}

TEST(MemChrTest, SignedCharacterFound) {
  char c = -1;
  const size_t size = 1;
  char src[size] = {c};
  const char *actual = call_memchr(src, c, size);
  // Should find the first character 'c'.
  ASSERT_EQ(actual[0], c);
}
