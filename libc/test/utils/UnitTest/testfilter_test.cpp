//===-- Tests for Test Filter functionality -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/UnitTest/LibcTest.h"

using LIBC_NAMESPACE::testing::TestOptions;

TEST(LlvmLibcTestFilterTest, CorrectFilter) {}

TEST(LlvmLibcTestFilterTest, CorrectFilter2) {}

TEST(LlvmLibcTestFilterTest, IncorrectFilter) {}

TEST(LlvmLibcTestFilterTest, NoFilter) {}

TEST(LlvmLibcTestFilterTest, CheckCorrectFilter) {
  TestOptions Options;
  Options.TestFilter = "LlvmLibcTestFilterTest.NoFilter";
  ASSERT_EQ(LIBC_NAMESPACE::testing::Test::runTests(Options), 0);

  Options.TestFilter = "LlvmLibcTestFilterTest.IncorrFilter";
  ASSERT_EQ(LIBC_NAMESPACE::testing::Test::runTests(Options), 1);

  Options.TestFilter = "LlvmLibcTestFilterTest.CorrectFilter";
  ASSERT_EQ(LIBC_NAMESPACE::testing::Test::runTests(Options), 0);

  Options.TestFilter = "LlvmLibcTestFilterTest.CorrectFilter2";
  ASSERT_EQ(LIBC_NAMESPACE::testing::Test::runTests(Options), 0);
}

int main() {
  TestOptions Options{"LlvmLibcTestFilterTest.NoFilter", /*PrintColor=*/true};
  LIBC_NAMESPACE::testing::Test::runTests(Options);
  return 0;
}
