//===-- Unittests for strlen ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strlen.h"
#include "utils/UnitTest/Test.h"

TEST(LlvmLibcStrLenTest, EmptyString) {
  const char *empty = "";

  size_t result = __llvm_libc::strlen(empty);
  ASSERT_EQ((size_t)0, result);
}

TEST(LlvmLibcStrLenTest, AnyString) {
  const char *any = "Hello World!";

  size_t result = __llvm_libc::strlen(any);
  ASSERT_EQ((size_t)12, result);
}
