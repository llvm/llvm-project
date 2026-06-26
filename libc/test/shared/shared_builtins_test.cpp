//===-- Unittests for shared builtins -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "shared/builtins.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcSharedBuiltinsTest, AllFloat) {
  // TODO: assertions for shared::*sf3 builtins.
}

TEST(LlvmLibcSharedBuiltinsTest, AllDouble) {
  // TODO: assertions for shared::*df3 builtins.
}

#ifdef LIBC_TYPES_HAS_FLOAT128

TEST(LlvmLibcSharedBuiltinsTest, AllFloat128) {
  namespace shared = LIBC_NAMESPACE::shared;

  EXPECT_FP_EQ(float128(3.0), shared::addtf3(float128(1.0), float128(2.0)));
}

#endif // LIBC_TYPES_HAS_FLOAT128
