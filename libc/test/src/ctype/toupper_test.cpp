//===-- Unittests for toupper----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/span.h"
#include "src/ctype/toupper.h"

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

TEST(LlvmLibcToUpper, SimpleTest) {
  EXPECT_EQ(LIBC_NAMESPACE::toupper('a'), int('A'));
  EXPECT_EQ(LIBC_NAMESPACE::toupper('B'), int('B'));
  EXPECT_EQ(LIBC_NAMESPACE::toupper('3'), int('3'));

  EXPECT_EQ(LIBC_NAMESPACE::toupper(' '), int(' '));
  EXPECT_EQ(LIBC_NAMESPACE::toupper('?'), int('?'));
  EXPECT_EQ(LIBC_NAMESPACE::toupper('\0'), int('\0'));
  EXPECT_EQ(LIBC_NAMESPACE::toupper(-1), int(-1));
}

TEST(LlvmLibcToUpper, DefaultLocale) {
  for (int ch = -255; ch < 255; ++ch) {
    int char_index = span_index(ch, LOWER_ARR);
    if (char_index != -1)
      EXPECT_EQ(LIBC_NAMESPACE::toupper(ch),
                static_cast<int>(UPPER_ARR[char_index]));
    else
      EXPECT_EQ(LIBC_NAMESPACE::toupper(ch), ch);
  }
}
