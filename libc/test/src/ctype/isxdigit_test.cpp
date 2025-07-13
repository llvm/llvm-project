//===-- Unittests for isxdigit---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/span.h"
#include "src/ctype/isxdigit.h"

#include "test/UnitTest/Test.h"

namespace {

// TODO: Merge the ctype tests using this framework.
constexpr char XDIGIT_ARRAY[] = {
    'a', 'b', 'c', 'd', 'e', 'f', 'A', 'B', 'C', 'D', 'E',
    'F', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
};

bool in_span(int ch, LIBC_NAMESPACE::cpp::span<const char> arr) {
  for (size_t i = 0; i < arr.size(); ++i)
    if (static_cast<int>(arr[i]) == ch)
      return true;
  return false;
}

} // namespace

TEST(LlvmLibcIsXdigit, SimpleTest) {
  EXPECT_NE(LIBC_NAMESPACE::isxdigit('a'), 0);
  EXPECT_NE(LIBC_NAMESPACE::isxdigit('B'), 0);
  EXPECT_NE(LIBC_NAMESPACE::isxdigit('3'), 0);

  EXPECT_EQ(LIBC_NAMESPACE::isxdigit('z'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::isxdigit(' '), 0);
  EXPECT_EQ(LIBC_NAMESPACE::isxdigit('?'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::isxdigit('\0'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::isxdigit(-1), 0);
}

TEST(LlvmLibcIsXdigit, DefaultLocale) {
  // Loops through all characters, verifying that numbers and letters
  // return non-zero integer and everything else returns a zero.
  for (int ch = -255; ch < 255; ++ch) {
    if (in_span(ch, XDIGIT_ARRAY))
      EXPECT_NE(LIBC_NAMESPACE::isxdigit(ch), 0);
    else
      EXPECT_EQ(LIBC_NAMESPACE::isxdigit(ch), 0);
  }
}
