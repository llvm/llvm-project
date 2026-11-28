//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for putc_unlocked.
///
//===----------------------------------------------------------------------===//

#include "hdr/stdio_macros.h"
#include "src/__support/CPP/scope.h"
#include "src/stdio/fclose.h"
#include "src/stdio/ferror.h"
#include "src/stdio/flockfile.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fread.h"
#include "src/stdio/funlockfile.h"
#include "src/stdio/putc_unlocked.h"
#include "src/string/memcmp.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LIBC_NAMESPACE::cpp::scope_exit;

using LlvmLibcPutcUnlockedTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcPutcUnlockedTest, WriteBoundaryAndMultipleChars) {
  const auto FILENAME = libc_make_test_file_path("putc_unlocked.test");
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  constexpr int INPUT_CHARS[] = {0,  1,     'A', 127, 128, 255,   -1,   -128,
                                 -2, 0x141, 'a', 'b', 'c', 0x100, 0x1ff};
  constexpr size_t NUM_CHARS = sizeof(INPUT_CHARS) / sizeof(INPUT_CHARS[0]);
  unsigned char expected[NUM_CHARS];

  {
    scope_exit close_file(
        [&] { ASSERT_THAT(LIBC_NAMESPACE::fclose(file), Succeeds()); });

    for (size_t i = 0; i < NUM_CHARS; ++i) {
      expected[i] = static_cast<unsigned char>(INPUT_CHARS[i]);
      int expected_ret = static_cast<int>(expected[i]);
      ASSERT_THAT(LIBC_NAMESPACE::putc_unlocked(INPUT_CHARS[i], file),
                  Succeeds(expected_ret));
    }
  }

  // Data should be visible after fclose.
  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);
  scope_exit close_file(
      [&] { ASSERT_THAT(LIBC_NAMESPACE::fclose(file), Succeeds()); });

  unsigned char read_buf[NUM_CHARS];
  ASSERT_THAT(LIBC_NAMESPACE::fread(read_buf, 1, NUM_CHARS, file),
              Succeeds(NUM_CHARS));
  EXPECT_EQ(LIBC_NAMESPACE::memcmp(read_buf, expected, NUM_CHARS), 0);
}

TEST_F(LlvmLibcPutcUnlockedTest, ReadOnlyStream) {
  const auto FILENAME = libc_make_test_file_path("putc_unlocked_readonly.test");
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  ASSERT_THAT(LIBC_NAMESPACE::fclose(file), Succeeds());

  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);
  scope_exit close_file(
      [&] { ASSERT_THAT(LIBC_NAMESPACE::fclose(file), Succeeds()); });

  ASSERT_EQ(LIBC_NAMESPACE::ferror(file), 0);
  ASSERT_THAT(LIBC_NAMESPACE::putc_unlocked('X', file), Fails(EBADF, EOF));
  EXPECT_NE(LIBC_NAMESPACE::ferror(file), 0);
}

TEST_F(LlvmLibcPutcUnlockedTest, WriteUnderLock) {
  const auto FILENAME = libc_make_test_file_path("putc_unlocked_lock.test");
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  constexpr int INPUT_CHARS[] = {'L', 'O', 'C', 'K', 0, 128, 255};
  constexpr size_t NUM_CHARS = sizeof(INPUT_CHARS) / sizeof(INPUT_CHARS[0]);
  unsigned char expected[NUM_CHARS];

  {
    scope_exit close_file(
        [&] { ASSERT_THAT(LIBC_NAMESPACE::fclose(file), Succeeds()); });

    LIBC_NAMESPACE::flockfile(file);
    scope_exit unlock_file([&] { LIBC_NAMESPACE::funlockfile(file); });

    for (size_t i = 0; i < NUM_CHARS; ++i) {
      expected[i] = static_cast<unsigned char>(INPUT_CHARS[i]);
      int expected_ret = static_cast<int>(expected[i]);
      ASSERT_THAT(LIBC_NAMESPACE::putc_unlocked(INPUT_CHARS[i], file),
                  Succeeds(expected_ret));
    }
  }

  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);
  scope_exit close_file(
      [&] { ASSERT_THAT(LIBC_NAMESPACE::fclose(file), Succeeds()); });

  unsigned char read_buf[NUM_CHARS];
  ASSERT_THAT(LIBC_NAMESPACE::fread(read_buf, 1, NUM_CHARS, file),
              Succeeds(NUM_CHARS));
  EXPECT_EQ(LIBC_NAMESPACE::memcmp(read_buf, expected, NUM_CHARS), 0);
}
