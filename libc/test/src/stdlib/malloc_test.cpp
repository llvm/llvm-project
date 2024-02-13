//===-- Unittests for malloc ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/free.h"
#include "src/stdlib/malloc.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcMallocTest, Allocate) {
  int *ptr = reinterpret_cast<int *>(LIBC_NAMESPACE::malloc(sizeof(int)));
  EXPECT_NE(reinterpret_cast<void *>(ptr), static_cast<void *>(nullptr));
  *ptr = 1;
  EXPECT_EQ(*ptr, 1);
  LIBC_NAMESPACE::free(ptr);
}
