//===-- Unittests for fopen / fclose --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/scope.h"
#include "src/__support/File/file.h"
#include "src/__support/macros/properties/os.h"
#include "src/stdio/fclose.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fread.h"
#include "src/stdio/fwrite.h"

#include "test/UnitTest/Test.h"

#ifdef LIBC_TARGET_OS_IS_LINUX
#include "hdr/fcntl_macros.h"
#include "src/fcntl/fcntl.h"
#include "src/stdio/fileno.h"
#include "src/stdio/remove.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#endif

using LIBC_NAMESPACE::cpp::scope_exit;

#ifdef LIBC_TARGET_OS_IS_LINUX
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
#endif

TEST(LlvmLibcFOpenTest, PrintToFile) {
  size_t result;

  static constexpr char STRING[] = "A simple string written to a file\n";
  {
    FILE *file = LIBC_NAMESPACE::fopen("testdata/test.txt", "w");
    ASSERT_FALSE(file == nullptr);
    scope_exit close_file([&] { ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file)); });

    result = LIBC_NAMESPACE::fwrite(STRING, 1, sizeof(STRING) - 1, file);
    EXPECT_GE(result, size_t(0));
  }

  {
    FILE *file = LIBC_NAMESPACE::fopen("testdata/test.txt", "r");
    ASSERT_FALSE(file == nullptr);
    scope_exit close_file([&] { ASSERT_EQ(0, LIBC_NAMESPACE::fclose(file)); });

    static char data[64] = {0};
    ASSERT_EQ(LIBC_NAMESPACE::fread(data, 1, sizeof(STRING) - 1, file),
              sizeof(STRING) - 1);
    data[sizeof(STRING) - 1] = '\0';
    ASSERT_STREQ(data, STRING);
  }
}

#ifdef LIBC_TARGET_OS_IS_LINUX
class LlvmLibcFOpenModeTest
    : public LIBC_NAMESPACE::testing::ErrnoCheckingTest {
protected:
  void check_exclusive_create(const char *mode) {
    const auto FILENAME = libc_make_test_file_path("fopen_exclusive.test");
    constexpr char CONTENT[] = "Preserve this content";

    // Remove a file left by an interrupted test run.
    LIBC_NAMESPACE::remove(FILENAME);
    libc_errno = 0;
    FILE *file = LIBC_NAMESPACE::fopen(FILENAME, mode);
    ASSERT_NE(file, nullptr);
    scope_exit remove_file(
        [&] { EXPECT_THAT(LIBC_NAMESPACE::remove(FILENAME), Succeeds(0)); });
    {
      scope_exit close_file(
          [&] { EXPECT_THAT(LIBC_NAMESPACE::fclose(file), Succeeds(0)); });
      ASSERT_EQ(LIBC_NAMESPACE::fwrite(CONTENT, 1, sizeof(CONTENT) - 1, file),
                sizeof(CONTENT) - 1);
    }

    FILE *existing = LIBC_NAMESPACE::fopen(FILENAME, mode);
    EXPECT_THAT(existing, Fails(EEXIST, static_cast<void *>(nullptr)));
    if (existing != nullptr)
      EXPECT_THAT(LIBC_NAMESPACE::fclose(existing), Succeeds(0));

    file = LIBC_NAMESPACE::fopen(FILENAME, "r");
    ASSERT_NE(file, nullptr);
    scope_exit close_file(
        [&] { EXPECT_THAT(LIBC_NAMESPACE::fclose(file), Succeeds(0)); });
    char buffer[sizeof(CONTENT)] = {};
    ASSERT_EQ(LIBC_NAMESPACE::fread(buffer, 1, sizeof(buffer) - 1, file),
              sizeof(CONTENT) - 1);
    EXPECT_STREQ(buffer, CONTENT);
  }
};

TEST_F(LlvmLibcFOpenModeTest, ExclusiveWrite) { check_exclusive_create("wx"); }

TEST_F(LlvmLibcFOpenModeTest, ExclusiveAppend) { check_exclusive_create("ax"); }

TEST_F(LlvmLibcFOpenModeTest, CloseOnExec) {
  {
    FILE *file = LIBC_NAMESPACE::fopen("/dev/null", "w");
    ASSERT_NE(file, nullptr);
    scope_exit close_file(
        [&] { EXPECT_THAT(LIBC_NAMESPACE::fclose(file), Succeeds(0)); });
    EXPECT_THAT(LIBC_NAMESPACE::fcntl(LIBC_NAMESPACE::fileno(file), F_GETFD),
                Succeeds(0));
  }
  {
    FILE *file = LIBC_NAMESPACE::fopen("/dev/null", "we");
    ASSERT_NE(file, nullptr);
    scope_exit close_file(
        [&] { EXPECT_THAT(LIBC_NAMESPACE::fclose(file), Succeeds(0)); });
    EXPECT_THAT(LIBC_NAMESPACE::fcntl(LIBC_NAMESPACE::fileno(file), F_GETFD),
                Succeeds(FD_CLOEXEC));
  }
}

TEST_F(LlvmLibcFOpenModeTest, ReadIgnoresExclusiveModifier) {
  const auto FILENAME = libc_make_test_file_path("fopen_read_exclusive.test");
  FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_NE(file, nullptr);
  scope_exit remove_file(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::remove(FILENAME), Succeeds(0)); });
  ASSERT_THAT(LIBC_NAMESPACE::fclose(file), Succeeds(0));

  file = LIBC_NAMESPACE::fopen(FILENAME, "rx");
  ASSERT_NE(file, nullptr);
  EXPECT_THAT(LIBC_NAMESPACE::fclose(file), Succeeds(0));
}
#endif
