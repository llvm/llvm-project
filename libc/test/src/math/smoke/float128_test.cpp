//===-- Unittests for Float128 emulated type ------------------------------===//
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
  (void)x; // TEST 1>
}

TEST(LlvmLibcFloat128Test, doubleToFloat128) {
  // Double -> Float128 -> double should give back the same value
  constexpr double vals[] = {0.0, 1.0, -1.0, 2.0, 0.5, 3.14};
  for (double v : vals) {
    Float128 x(v);
    ASSERT_EQ(static_cast<double>(x), v);
  }
}

TEST(LlvmLibcFloat128Test, ZeroBits) {
  Float128 x(0.0);
  ASSERT_EQ(x.bits, static_cast<LIBC_NAMESPACE::UInt128>(0));
}

TEST(LlvmLibcFloat128Test, Equality) {
  Float128 a(1.0), b(1.0), c(2.0);
  ASSERT_TRUE(a == b);
  ASSERT_TRUE(a != c);
}
