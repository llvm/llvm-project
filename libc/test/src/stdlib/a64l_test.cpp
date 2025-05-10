//===-- Unittests for a64l ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/a64l.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcA64lTest, EmptyString) { ASSERT_EQ(LIBC_NAMESPACE::a64l(""), 0l); }
TEST(LlvmLibcA64lTest, FullString) {
  ASSERT_EQ(LIBC_NAMESPACE::a64l("AbC12/"), 1141696972l);
}

constexpr char B64_CHARS[64] = {
    '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A',
    'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
    'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a',
    'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
};

TEST(LlvmLibcA64lTest, OneCharacter) {
  char example_str[2] = {'\0', '\0'};

  for (size_t i = 0; i < 64; ++i) {
    example_str[0] = B64_CHARS[i];
    ASSERT_EQ(LIBC_NAMESPACE::a64l(example_str), static_cast<long>(i));
  }
}

TEST(LlvmLibcA64lTest, TwoCharacters) {
  char example_str[3] = {'\0', '\0', '\0'};

  for (size_t first = 0; first < 64; ++first) {
    example_str[0] = B64_CHARS[first];
    for (size_t second = 0; second < 64; ++second) {
      example_str[1] = B64_CHARS[second];

      ASSERT_EQ(LIBC_NAMESPACE::a64l(example_str),
                static_cast<long>(first + (second * 64)));
    }
  }
}

TEST(LlvmLibcA64lTest, FiveSameCharacters) {
  // Technically the last digit can be parsed to give the last two bits. Not
  // handling that here.
  char example_str[6] = {
      '\0', '\0', '\0', '\0', '\0', '\0',
  };

  // set every 6th bit
  const long BASE_NUM = 0b1000001000001000001000001;

  for (size_t char_val = 0; char_val < 64; ++char_val) {
    for (size_t i = 0; i < 5; ++i)
      example_str[i] = B64_CHARS[char_val];

    const long expected_result = BASE_NUM * char_val;

    ASSERT_EQ(LIBC_NAMESPACE::a64l(example_str), expected_result);
  }
}

TEST(LlvmLibcA64lTest, OneOfSixCharacters) {
  char example_str[7] = {'\0', '\0', '\0', '\0', '\0', '\0', '\0'};

  for (size_t cur_char = 0; cur_char < 6; ++cur_char) {
    // clear the string, set all the chars to b64(0)
    for (size_t i = 0; i < 6; ++i)
      example_str[i] = B64_CHARS[0];

    for (size_t char_val = 0; char_val < 64; ++char_val) {
      example_str[cur_char] = B64_CHARS[char_val];

      // Need to limit to 32 bits, since that's what the standard says the
      // function does.
      const long expected_result =
          static_cast<int32_t>(char_val << (6 * cur_char));

      ASSERT_EQ(LIBC_NAMESPACE::a64l(example_str), expected_result);
    }
  }
}
