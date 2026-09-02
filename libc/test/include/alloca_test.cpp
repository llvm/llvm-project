//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for alloca.
///
//===----------------------------------------------------------------------===//

#include "test/UnitTest/Test.h"

#include "include/llvm-libc-macros/alloca-macros.h"

TEST(LlvmLibcAllocaTest, Basic) {
  // Ensure the macro is defined.
#ifndef alloca
  FAIL() << "alloca macro is not defined";
#endif

  // Allocate some memory on the stack.
  void *ptr = alloca(10);
  ASSERT_NE(ptr, static_cast<void *>(nullptr));

  // Write to it to make sure it's valid memory.
  char *char_ptr = static_cast<char *>(ptr);
  for (int i = 0; i < 10; ++i) {
    char_ptr[i] = static_cast<char>(i);
  }

  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(char_ptr[i], static_cast<char>(i));
  }
}
