//===-- Unittests for isupper----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/span.h"
#include "src/ctype/isupper.h"

#include "test/UnitTest/Test.h"

namespace {

// TODO: Merge the ctype tests using this framework.
constexpr char UPPER_ARRAY[] = {
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
};

bool in_span(int ch, LIBC_NAMESPACE::cpp::span<const char> arr) {
  for (size_t i = 0; i < arr.size(); ++i)
    if (static_cast<int>(arr[i]) == ch)
      return true;
  return false;
}

} // namespace

TEST(LlvmLibcIsUpper, SimpleTest) {
  EXPECT_NE(LIBC_NAMESPACE::isupper('B'), 0);

  EXPECT_EQ(LIBC_NAMESPACE::isupper('a'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::isupper('3'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::isupper(' '), 0);
  EXPECT_EQ(LIBC_NAMESPACE::isupper('?'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::isupper('\0'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::isupper(-1), 0);
}

TEST(LlvmLibcIsUpper, DefaultLocale) {
  // Loops through all characters, verifying that numbers and letters
  // return non-zero integer and everything else returns a zero.
  for (int ch = -255; ch < 255; ++ch) {
    if (in_span(ch, UPPER_ARRAY))
      EXPECT_NE(LIBC_NAMESPACE::isupper(ch), 0);
    else
      EXPECT_EQ(LIBC_NAMESPACE::isupper(ch), 0);
  }
}
