//===-- Tests for Test Filter functionality -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "utils/UnitTest/LibcTest.h"

TEST(LlvmLibcTestFilterTest, CorrectFilter) {}

TEST(LlvmLibcTestFilterTest, CorrectFilter2) {}

TEST(LlvmLibcTestFilterTest, IncorrectFilter) {}

TEST(LlvmLibcTestFilterTest, NoFilter) {}

TEST(LlvmLibcTestFilterTest, CheckCorrectFilter) {
  ASSERT_EQ(
      __llvm_libc::testing::Test::runTests("LlvmLibcTestFilterTest.NoFilter"),
      0);
  ASSERT_EQ(__llvm_libc::testing::Test::runTests(
                "LlvmLibcTestFilterTest.IncorrFilter"),
            1);
  ASSERT_EQ(__llvm_libc::testing::Test::runTests(
                "LlvmLibcTestFilterTest.CorrectFilter"),
            0);
  ASSERT_EQ(__llvm_libc::testing::Test::runTests(
                "LlvmLibcTestFilterTest.CorrectFilter2"),
            0);
}

int main() {
  __llvm_libc::testing::Test::runTests(
      "LlvmLibcTestFilterTest.CheckCorrectFilter");
  return 0;
}
