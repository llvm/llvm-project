//===-- Unittests for l64a ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdlib/l64a.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcL64aTest, Zero) {
  ASSERT_STREQ(LIBC_NAMESPACE::l64a(0), "......");
}
TEST(LlvmLibcL64aTest, Max) {
  ASSERT_STREQ(LIBC_NAMESPACE::l64a(
                   LIBC_NAMESPACE::cpp::numeric_limits<uint32_t>::max()),
               "zzzzz1");
}

constexpr char B64_CHARS[64] = {
    '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A',
    'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
    'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a',
    'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
};

TEST(LlvmLibcL64aTest, OneCharacter) {
  // The trailing null is technically unnecessary, but it means it won't look
  // bad when we print it.
  char expected_str[7] = {'\0', '.', '.', '.', '.', '.', '\0'};

  for (size_t i = 0; i < 64; ++i) {
    expected_str[0] = B64_CHARS[i];
    ASSERT_STREQ(LIBC_NAMESPACE::l64a(i), expected_str);
  }
}

TEST(LlvmLibcL64aTest, TwoCharacters) {
  char expected_str[7] = {'\0', '\0', '.', '.', '.', '.', '\0'};

  for (size_t first = 0; first < 64; ++first) {
    expected_str[0] = B64_CHARS[first];
    for (size_t second = 0; second < 64; ++second) {
      expected_str[1] = B64_CHARS[second];

      ASSERT_STREQ(LIBC_NAMESPACE::l64a(first + (second * 64)), expected_str);
    }
  }
}

TEST(LlvmLibcL64aTest, FiveSameCharacters) {
  // Only using 5 because those are the only digits that can be any character.
  char expected_str[7] = {'\0', '\0', '\0', '\0', '\0', '.', '\0'};

  // set every 6th bit
  const long BASE_NUM = 0b1000001000001000001000001;

  for (size_t char_val = 0; char_val < 64; ++char_val) {
    for (size_t i = 0; i < 5; ++i)
      expected_str[i] = B64_CHARS[char_val];

    const long input_num = BASE_NUM * char_val;

    ASSERT_STREQ(LIBC_NAMESPACE::l64a(input_num), expected_str);
  }
}

TEST(LlvmLibcL64aTest, OneOfSixCharacters) {
  char expected_str[7] = {'\0', '\0', '\0', '\0', '\0', '\0', '\0'};

  for (size_t cur_char = 0; cur_char < 6; ++cur_char) {
    // clear the string, set all the chars to b64(0)
    for (size_t i = 0; i < 6; ++i)
      expected_str[i] = B64_CHARS[0];

    for (size_t char_val = 0; char_val < 64; ++char_val) {
      // Since each base64 character holds 6 bits and we're only using 32 bits
      // of input, the 6th character only gets 2 bits, so it can never be
      // greater than 3.
      if (char_val > 3 && cur_char == 5)
        break;
      expected_str[cur_char] = B64_CHARS[char_val];

      // Need to limit to 32 bits, since that's what the standard says the
      // function does.
      const long input_num = static_cast<int32_t>(char_val << (6 * cur_char));

      ASSERT_STREQ(LIBC_NAMESPACE::l64a(input_num), expected_str);
    }
  }
}
