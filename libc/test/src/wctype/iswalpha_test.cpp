//===-- Unittests for iswalpha --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/span.h"
#include "src/wctype/iswalpha.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibciswalpha, SimpleTest) {
  EXPECT_NE(LIBC_NAMESPACE::iswalpha('a'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswalpha('B'), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswalpha('3'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswalpha(' '), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswalpha('?'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswalpha('\0'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswalpha(-1), 0);
}

// TODO: once iswalpha supports more than just ascii-range characters add a
// proper test.

// namespace {

// // TODO: Merge the wctype tests using this framework.
// constexpr char WALPHA_ARRAY[] = {
//     'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
//     'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
//     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
//     'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
// };

// bool in_span(int ch, LIBC_NAMESPACE::cpp::span<const char> arr) {
//   for (size_t i = 0; i < arr.size(); ++i)
//     if (static_cast<int>(arr[i]) == ch)
//       return true;
//   return false;
// }

// } // namespace

// TEST(LlvmLibciswalpha, DefaultLocale) {
//   // Loops through all characters, verifying that letters return
//   // true and everything else returns false.
//   for (int ch = -255; ch < 255; ++ch) {
//     if (in_span(ch, WALPHA_ARRAY))
//       EXPECT_TRUE(LIBC_NAMESPACE::iswalpha(ch));
//     else
//       EXPECT_FALSE(LIBC_NAMESPACE::iswalpha(ch));
//   }
// }
