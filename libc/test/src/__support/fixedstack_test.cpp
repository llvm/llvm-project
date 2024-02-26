//===-- Unittests for FixedStack ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/fixedstack.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcFixedVectorTest, PushAndPop) {
  static LIBC_NAMESPACE::FixedStack<int, 20> fixed_stack;
  ASSERT_TRUE(fixed_stack.empty());
  for (int i = 0; i < 20; i++)
    ASSERT_TRUE(fixed_stack.push(i));
  ASSERT_FALSE(fixed_stack.empty());
  ASSERT_FALSE(fixed_stack.push(123));
  int val;
  for (int i = 20; i > 0; --i) {
    ASSERT_TRUE(fixed_stack.pop(val));
    ASSERT_EQ(val, i - 1);
  }
  ASSERT_FALSE(fixed_stack.pop(val));
  ASSERT_TRUE(fixed_stack.empty());
}
