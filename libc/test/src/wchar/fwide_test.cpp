//===-- Unittests for fwide -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fclose.h"
#include "src/stdio/fopen.h"
#include "src/wchar/fwide.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcFwideTest, QueryInitial) {
  auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("fwide_query.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  // Initial orientation should be unoriented (0)
  EXPECT_EQ(LIBC_NAMESPACE::fwide(file, 0), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST(LlvmLibcFwideTest, OrientWide) {
  auto FILENAME = libc_make_test_file_path(APPEND_LIBC_TEST("fwide_wide.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  // Setting mode > 0 should return > 0 (wide oriented)
  EXPECT_GT(LIBC_NAMESPACE::fwide(file, 1), 0);

  // Subsequent orientation queries/attempts should still return > 0
  EXPECT_GT(LIBC_NAMESPACE::fwide(file, 0), 0);
  EXPECT_GT(LIBC_NAMESPACE::fwide(file, -1), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST(LlvmLibcFwideTest, OrientByte) {
  auto FILENAME = libc_make_test_file_path(APPEND_LIBC_TEST("fwide_byte.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  // Setting mode < 0 should return < 0 (byte oriented)
  EXPECT_LT(LIBC_NAMESPACE::fwide(file, -1), 0);

  // Subsequent orientation queries/attempts should still return < 0
  EXPECT_LT(LIBC_NAMESPACE::fwide(file, 0), 0);
  EXPECT_LT(LIBC_NAMESPACE::fwide(file, 1), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}
