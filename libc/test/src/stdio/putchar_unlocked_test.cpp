//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for putchar_unlocked.
///
//===----------------------------------------------------------------------===//

#include "hdr/stdio_macros.h"
#include "src/__support/CPP/scope.h"
#include "src/stdio/fclose.h"
#include "src/stdio/ferror.h"
#include "src/stdio/fflush.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fread.h"
#include "src/stdio/freopen.h"
#include "src/stdio/putchar_unlocked.h"
#include "src/stdio/stdout.h"
#include "src/string/memcmp.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::scope_exit;
using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

using LlvmLibcPutcharUnlockedTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcPutcharUnlockedTest, WriteToStdout) {
  const auto FILENAME =
      libc_make_test_file_path("putchar_unlocked_stdout.test");

  // Redirect stdout to FILENAME
  ::FILE *reopened =
      LIBC_NAMESPACE::freopen(FILENAME, "w", LIBC_NAMESPACE::stdout);
  ASSERT_EQ(reopened, static_cast<::FILE *>(LIBC_NAMESPACE::stdout));

  constexpr int INPUT_CHARS[] = {'H',   'e', 'l', 'l', 'o',  ',',  ' ',   'W',
                                 'o',   'r', 'l', 'd', '!',  '\n', 0,     1,
                                 127,   128, 255, -1,  -128, -2,   0x141, 0x100,
                                 0x1ff, 'e', 'n', 'd', '\0'};
  constexpr size_t NUM_CHARS = sizeof(INPUT_CHARS) / sizeof(INPUT_CHARS[0]);
  unsigned char expected[NUM_CHARS];

  for (size_t i = 0; i < NUM_CHARS; ++i) {
    expected[i] = static_cast<unsigned char>(INPUT_CHARS[i]);
    int expected_ret = static_cast<int>(expected[i]);
    ASSERT_THAT(LIBC_NAMESPACE::putchar_unlocked(INPUT_CHARS[i]),
                Succeeds(expected_ret));
  }

  ASSERT_THAT(LIBC_NAMESPACE::fflush(LIBC_NAMESPACE::stdout), Succeeds(0));

  // Read the file back with fopen/fread
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);
  scope_exit close_file(
      [&] { ASSERT_THAT(LIBC_NAMESPACE::fclose(file), Succeeds()); });

  unsigned char read_buf[NUM_CHARS];
  ASSERT_THAT(LIBC_NAMESPACE::fread(read_buf, 1, NUM_CHARS, file),
              Succeeds(NUM_CHARS));
  EXPECT_EQ(LIBC_NAMESPACE::memcmp(read_buf, expected, NUM_CHARS), 0);
}

TEST_F(LlvmLibcPutcharUnlockedTest, ReadOnlyStream) {
  const auto FILENAME =
      libc_make_test_file_path("putchar_unlocked_readonly.test");

  // Ensure test file exists
  ::FILE *init = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(init == nullptr);
  ASSERT_THAT(LIBC_NAMESPACE::fclose(init), Succeeds());

  // Redirect stdout to FILENAME in read-only mode
  ::FILE *reopened =
      LIBC_NAMESPACE::freopen(FILENAME, "r", LIBC_NAMESPACE::stdout);
  ASSERT_EQ(reopened, static_cast<::FILE *>(LIBC_NAMESPACE::stdout));

  ASSERT_EQ(LIBC_NAMESPACE::ferror(LIBC_NAMESPACE::stdout), 0);
  ASSERT_THAT(LIBC_NAMESPACE::putchar_unlocked('X'), Fails(EBADF, EOF));
  EXPECT_NE(LIBC_NAMESPACE::ferror(LIBC_NAMESPACE::stdout), 0);
}
