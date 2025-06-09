//===-- Unittests for fbfloat16 function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fbfloat16.h"

#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

// TODO: change this to bfloat16
using LlvmLibcFBfloat16Test = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcFBfloat16Test, SpecialNumbers) {

  LIBC_NAMESPACE::bfloat16 x{0.0f};
  ASSERT_EQ(0, static_cast<int>(x.bits));


  LIBC_NAMESPACE::bfloat16 y{1.0f};
  ASSERT_EQ(1, static_cast<int>(y.bits));

  // TODO: implement this!
  // x = some bfloat number
  // float y = x as float (using our ctor?)
  // float z = mfpr(x) as float
  // check y == z
}
