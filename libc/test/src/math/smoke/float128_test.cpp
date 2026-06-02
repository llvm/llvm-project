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

TEST(LlvmLibcFloat128Test, Equality) {
  Float128 a(1.0), b(1.0), c(2.0);
  ASSERT_TRUE(a == b);
  ASSERT_TRUE(a != c);
  ASSERT_TRUE(b != c);
}
