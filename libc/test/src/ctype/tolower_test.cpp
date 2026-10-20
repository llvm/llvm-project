//===-- Unittests for tolower----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/span.h"
#include "src/ctype/tolower.h"

#include "test/UnitTest/Test.h"

namespace {

// TODO: Merge the ctype tests using this framework.
// Invariant: UPPER_ARR and LOWER_ARR are both the complete alphabet in the same
// order.
constexpr char UPPER_ARR[] = {
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
};
constexpr char LOWER_ARR[] = {
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
};

static_assert(
    sizeof(UPPER_ARR) == sizeof(LOWER_ARR),
    "There must be the same number of uppercase and lowercase letters.");

int span_index(int ch, LIBC_NAMESPACE::cpp::span<const char> arr) {
  for (size_t i = 0; i < arr.size(); ++i)
    if (static_cast<int>(arr[i]) == ch)
      return static_cast<int>(i);
  return -1;
}

} // namespace

TEST(LlvmLibcToLower, SimpleTest) {
  EXPECT_EQ(LIBC_NAMESPACE::tolower('a'), int('a'));
  EXPECT_EQ(LIBC_NAMESPACE::tolower('B'), int('b'));
  EXPECT_EQ(LIBC_NAMESPACE::tolower('3'), int('3'));

  EXPECT_EQ(LIBC_NAMESPACE::tolower(' '), int(' '));
  EXPECT_EQ(LIBC_NAMESPACE::tolower('?'), int('?'));
  EXPECT_EQ(LIBC_NAMESPACE::tolower('\0'), int('\0'));
  EXPECT_EQ(LIBC_NAMESPACE::tolower(-1), int(-1));
}

TEST(LlvmLibcToLower, DefaultLocale) {
  for (int ch = -255; ch < 255; ++ch) {
    int char_index = span_index(ch, UPPER_ARR);
    if (char_index != -1)
      EXPECT_EQ(LIBC_NAMESPACE::tolower(ch),
                static_cast<int>(LOWER_ARR[char_index]));
    else
      EXPECT_EQ(LIBC_NAMESPACE::tolower(ch), ch);
  }
}
