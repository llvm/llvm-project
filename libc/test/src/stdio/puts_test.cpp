//===-- Unittests for puts ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/puts.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibcPutsTest, PrintOut) {
  int result;

  constexpr char simple[] = "A simple string";
  result = __llvm_libc::puts(simple);
  EXPECT_GE(result, 0);

  // check that it appends a second newline at the end.
  constexpr char numbers[] = "1234567890\n";
  result = __llvm_libc::puts(numbers);
  EXPECT_GE(result, 0);

  constexpr char more[] = "1234 and more\n6789 and rhyme";
  result = __llvm_libc::puts(more);
  EXPECT_GE(result, 0);
}
