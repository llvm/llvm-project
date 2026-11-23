//===-- Unittests for memrchr ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memrchr.h"

#include <stddef.h>

#include "hdr/signal_macros.h"
#include "test/UnitTest/Test.h"

namespace {

// A helper function that calls memrchr and abstracts away the explicit cast for
// readability purposes.
const char *call_memrchr(const void *src, int c, size_t size) {
  return reinterpret_cast<const char *>(LIBC_NAMESPACE::memrchr(src, c, size));
}

TEST(LlvmLibcMemRChrTest, FindsCharacterAfterNullTerminator) {
  // memrchr should continue searching after a null terminator.
  const size_t size = 6;
  const unsigned char src[size] = {'a', '\0', 'b', 'c', 'd', '\0'};
  // Should return 'b', 'c', 'd', '\0' even when after null terminator.
  ASSERT_STREQ(call_memrchr(src, 'b', size), "bcd");
}

TEST(LlvmLibcMemRChrTest, FindsCharacterInNonNullTerminatedCollection) {
  const size_t size = 3;
  const unsigned char src[size] = {'a', 'b', 'c'};
  // Should return 'b', 'c'.
  const char *ret = call_memrchr(src, 'b', size);
  ASSERT_EQ(ret[0], 'b');
  ASSERT_EQ(ret[1], 'c');
}

TEST(LlvmLibcMemRChrTest, FindsFirstCharacter) {
  const size_t size = 6;
  const unsigned char src[size] = {'a', 'b', 'c', 'd', 'e', '\0'};
  // Should return original array since 'a' is the first character.
  ASSERT_STREQ(call_memrchr(src, 'a', size), "abcde");
}

TEST(LlvmLibcMemRChrTest, FindsMiddleCharacter) {
  const size_t size = 6;
  const unsigned char src[size] = {'a', 'b', 'c', 'd', 'e', '\0'};
  // Should return characters after (and including) 'c'.
  ASSERT_STREQ(call_memrchr(src, 'c', size), "cde");
}

TEST(LlvmLibcMemRChrTest, FindsLastCharacterThatIsNotNullTerminator) {
  const size_t size = 6;
  const unsigned char src[size] = {'a', 'b', 'c', 'd', 'e', '\0'};
  // Should return 'e' and null-terminator.
  ASSERT_STREQ(call_memrchr(src, 'e', size), "e");
}

TEST(LlvmLibcMemRChrTest, FindsNullTerminator) {
  const size_t size = 6;
  const unsigned char src[size] = {'a', 'b', 'c', 'd', 'e', '\0'};
  // Should return null terminator.
  ASSERT_STREQ(call_memrchr(src, '\0', size), "");
}

TEST(LlvmLibcMemRChrTest, CharacterNotWithinStringShouldReturnNullptr) {
  const size_t size = 4;
  const unsigned char src[size] = {'1', '2', '3', '?'};
  // Since 'z' is not within 'characters', should return nullptr.
  ASSERT_STREQ(call_memrchr(src, 'z', size), nullptr);
}

TEST(LlvmLibcMemRChrTest, CharacterNotWithinSizeShouldReturnNullptr) {
  const unsigned char src[5] = {'1', '2', '3', '4', '\0'};
  // Since '4' is not within the first 2 characters, this should return nullptr.
  const size_t size = 2;
  ASSERT_STREQ(call_memrchr(src, '4', size), nullptr);
}

TEST(LlvmLibcMemRChrTest, ShouldFindLastOfDuplicates) {
  size_t size = 12; // 11 characters + null terminator.
  const char *dups = "abc1def1ghi";
  // 1 is duplicated in 'dups', but it should find the last copy.
  ASSERT_STREQ(call_memrchr(dups, '1', size), "1ghi");

  const char *repeated = "XXXXX";
  size = 6; // 5 characters + null terminator.
  // Should return the last X with the null terminator.
  ASSERT_STREQ(call_memrchr(repeated, 'X', size), "X");
}

TEST(LlvmLibcMemRChrTest, EmptyStringShouldOnlyMatchNullTerminator) {
  const size_t size = 1; // Null terminator.
  const char *empty_string = "";
  // Null terminator should match.
  ASSERT_STREQ(call_memrchr(empty_string, '\0', size), "");
  // All other characters should not match.
  ASSERT_STREQ(call_memrchr(empty_string, 'A', size), nullptr);
  ASSERT_STREQ(call_memrchr(empty_string, '9', size), nullptr);
  ASSERT_STREQ(call_memrchr(empty_string, '?', size), nullptr);
}

TEST(LlvmLibcMemRChrTest, SignedCharacterFound) {
  char c = -1;
  const size_t size = 1;
  char src[size] = {c};
  const char *actual = call_memrchr(src, c, size);
  // Should find the last character 'c'.
  ASSERT_EQ(actual[0], c);
}

TEST(LlvmLibcMemRChrTest, ZeroLengthShouldReturnNullptr) {
  const unsigned char src[4] = {'a', 'b', 'c', '\0'};
  // This will iterate over exactly zero characters, so should return nullptr.
  ASSERT_STREQ(call_memrchr(src, 'd', 0), nullptr);
}

#if defined(LIBC_ADD_NULL_CHECKS)

TEST(LlvmLibcMemRChrTest, CrashOnNullPtr) {
  ASSERT_DEATH([]() { LIBC_NAMESPACE::memrchr(nullptr, 'd', 1); },
               WITH_SIGNAL(-1));
}

#endif // defined(LIBC_ADD_NULL_CHECKS)

} // namespace
