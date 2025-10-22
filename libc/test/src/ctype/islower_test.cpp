//===-- Unittests for islower----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/span.h"
#include "src/ctype/islower.h"

#include "test/UnitTest/Test.h"

namespace {

// TODO: Merge the ctype tests using this framework.
constexpr char LOWER_ARRAY[] = {
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
};

bool in_span(int ch, LIBC_NAMESPACE::cpp::span<const char> arr) {
  for (size_t i = 0; i < arr.size(); ++i)
    if (static_cast<int>(arr[i]) == ch)
      return true;
  return false;
}

} // namespace

TEST(LlvmLibcIsLower, SimpleTest) {
  EXPECT_NE(LIBC_NAMESPACE::islower('a'), 0);

  EXPECT_EQ(LIBC_NAMESPACE::islower('B'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::islower('3'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::islower(' '), 0);
  EXPECT_EQ(LIBC_NAMESPACE::islower('?'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::islower('\0'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::islower(-1), 0);
}

TEST(LlvmLibcIsLower, DefaultLocale) {
  // Loops through all characters, verifying that numbers and letters
  // return non-zero integer and everything else returns a zero.
  for (int ch = -255; ch < 255; ++ch) {
    if (in_span(ch, LOWER_ARRAY))
      EXPECT_NE(LIBC_NAMESPACE::islower(ch), 0);
    else
      EXPECT_EQ(LIBC_NAMESPACE::islower(ch), 0);
  }
}
