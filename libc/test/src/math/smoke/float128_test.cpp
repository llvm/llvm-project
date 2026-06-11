//===-- Unittests for Float128 emulated type -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/float128.h"
#include "src/__support/uint128.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::fputil::Float128;

TEST(LlvmLibcFloat128Test, DefaultConstructor) {
  Float128 x;
  (void)x;
}

TEST(LlvmLibcFloat128Test, UnaryOperators) {
  Float128 a(1.0f), b(1.0f), c(2.0f);

  // comparison operators
  ASSERT_TRUE(a == b);
  // ASSERT_TRUE(a == Float128(1.0));
  ASSERT_TRUE(a != c);
  ASSERT_TRUE(b != c);
  ASSERT_TRUE(c > b);
  ASSERT_TRUE(a >= b);
  ASSERT_TRUE(b <= c);
  ASSERT_TRUE(a < c);

  // //negation 
  // Float128 pa(1.0),na(-1.0f);
  // ASSERT_TRUE(-pa == na);
  // ASSERT_TRUE(-(-pa)== -na);
}

TEST(LlvmLibcFloat128Test, BinaryOperators) {
  Float128 a(1.0f), b(1.0f), c(2.0f), d(3.0f);
  ASSERT_TRUE((a + b) == c);
  ASSERT_TRUE((a - b) == Float128(0.0f));
  ASSERT_TRUE((c * d) == Float128(6.0f));
  ASSERT_TRUE((Float128(6.0f) / d) == Float128(2.0f));
}
