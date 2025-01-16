//===-- Unittests for isalpha----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/span.h"
#include "src/ctype/isalpha.h"

#include "test/UnitTest/Test.h"

namespace {

// TODO: Merge the ctype tests using this framework.
constexpr char ALPHA_ARRAY[] = {
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
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

TEST(LlvmLibcIsAlpha, SimpleTest) {
  EXPECT_NE(LIBC_NAMESPACE::isalpha('a'), 0);
  EXPECT_NE(LIBC_NAMESPACE::isalpha('B'), 0);

  EXPECT_EQ(LIBC_NAMESPACE::isalpha('3'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::isalpha(' '), 0);
  EXPECT_EQ(LIBC_NAMESPACE::isalpha('?'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::isalpha('\0'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::isalpha(-1), 0);
}

TEST(LlvmLibcIsAlpha, DefaultLocale) {
  // Loops through all characters, verifying that letters return a
  // non-zero integer and everything else returns zero.
  // TODO: encoding indep
  for (int ch = -255; ch < 255; ++ch) {
    if (in_span(ch, ALPHA_ARRAY))
      EXPECT_NE(LIBC_NAMESPACE::isalpha(ch), 0);
    else
      EXPECT_EQ(LIBC_NAMESPACE::isalpha(ch), 0);
  }
}
