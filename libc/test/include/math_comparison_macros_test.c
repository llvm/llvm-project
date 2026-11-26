//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for math comparison macros in C.
///
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/math-function-macros.h"
#include "test/UnitTest/LibcCTest.h"

#if !defined(isgreater) || !defined(isgreaterequal) || !defined(isless) ||     \
    !defined(islessequal) || !defined(islessgreater) || !defined(isunordered)
#error "One or more math comparison macros are not defined"
#else
TEST(math_comparison_macros) {
  EXPECT_TRUE(isgreater(2.0f, 1.0f) == 1);
  EXPECT_TRUE(isgreater(1.0, 2.0) == 0);
  EXPECT_TRUE(isgreater(__builtin_nan(""), 1.0) == 0);

  EXPECT_TRUE(isgreaterequal(1.0, 1.0) == 1);
  EXPECT_TRUE(isgreaterequal(0.0, -0.0) == 1);
  EXPECT_TRUE(isgreaterequal(__builtin_nanf(""), 0.0f) == 0);

  EXPECT_TRUE(isless(-1.0L, 0.0L) == 1);
  EXPECT_TRUE(isless(0.0f, -0.0f) == 0);
  EXPECT_TRUE(isless(1.0, __builtin_nan("")) == 0);

  EXPECT_TRUE(islessequal(-0.0, 0.0) == 1);
  EXPECT_TRUE(islessequal(2.0f, 1.0) == 0);
  EXPECT_TRUE(islessequal(__builtin_nanl(""), 1.0L) == 0);

  EXPECT_TRUE(islessgreater(1.0f, -1.0f) == 1);
  EXPECT_TRUE(islessgreater(0.0, -0.0) == 0);
  EXPECT_TRUE(islessgreater(__builtin_nan(""), 0.0) == 0);

  EXPECT_TRUE(isunordered(__builtin_nanf(""), 1.0f) == 1);
  EXPECT_TRUE(isunordered(1.0, __builtin_nan("")) == 1);
  EXPECT_TRUE(isunordered(1.0L, -1.0L) == 0);
}
#endif // defined(isgreater) && ... && defined(isunordered)
