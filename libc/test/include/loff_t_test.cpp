//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for loff_t.
///
//===----------------------------------------------------------------------===//

#include "test/UnitTest/Test.h"

#include "include/llvm-libc-types/loff_t.h"
#include "include/llvm-libc-types/off_t.h"

// A call which takes an off_t * has to take a loff_t * too.
TEST(LlvmLibcLoffTTest, IsOffT) {
  ASSERT_TRUE(__is_same(loff_t, off_t));
  ASSERT_EQ(sizeof(loff_t), size_t(8));
}
