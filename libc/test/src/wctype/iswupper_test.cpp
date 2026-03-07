//===-- Unittests for iswupper --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/span.h"
#include "src/wctype/iswupper.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibciswupper, SimpleTest) {
  EXPECT_NE(LIBC_NAMESPACE::iswupper('B'), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswupper('a'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswupper('3'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswupper(' '), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswupper('?'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswupper('\0'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswupper(-1), 0);
}

// TODO: once iswupper supports more than just ascii-range characters add a
// proper test.

// namespace {

// // TODO: Merge the wctype tests using this framework.
// constexpr char WUPPER_ARRAY[] = {
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

// TEST(LlvmLibciswupper, DefaultLocale) {
//   // Loops through all characters, verifying that letters return
//   // true and everything else returns false.
//   for (int ch = -255; ch < 255; ++ch) {
//     if (in_span(ch, WUPPER_ARRAY))
//       EXPECT_TRUE(LIBC_NAMESPACE::iswupper(ch));
//     else
//       EXPECT_FALSE(LIBC_NAMESPACE::iswupper(ch));
//   }
// }
