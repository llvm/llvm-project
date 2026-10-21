//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for exit.
///
//===----------------------------------------------------------------------===//

#include "src/__support/macros/config.h"
#include "src/stdlib/exit.h"
#include "test/UnitTest/Test.h"

#ifdef LIBC_COPT_EXIT_FLUSH_STREAMS
#include "hdr/types/FILE.h"
#include "hdr/types/size_t.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/CPP/string_view.h"
#include "src/stdio/fclose.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fputs.h"
#include "src/stdio/fread.h"
#include "src/stdio/freopen.h"
#include "src/stdio/remove.h"
#include "src/stdio/stdout.h"
#include "src/stdlib/atexit.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#endif

TEST(LlvmLibcStdlib, exit) {
  EXPECT_EXITS([] { LIBC_NAMESPACE::exit(1); }, 1);
  EXPECT_EXITS([] { LIBC_NAMESPACE::exit(65); }, 65);
}

#ifdef LIBC_COPT_EXIT_FLUSH_STREAMS

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

struct LlvmLibcExitTest : public LIBC_NAMESPACE::testing::ErrnoCheckingTest {
  static constexpr size_t VERIFY_BUFFER_SIZE = 128;

  void verify_file_content(const char *filepath,
                           LIBC_NAMESPACE::cpp::string_view expected) {
    ASSERT_LT(expected.size(), VERIFY_BUFFER_SIZE);

    ::FILE *file = LIBC_NAMESPACE::fopen(filepath, "r");
    ASSERT_NE(file, nullptr);
    LIBC_NAMESPACE::cpp::scope_exit close_file(
        [&] { EXPECT_THAT(LIBC_NAMESPACE::fclose(file), Succeeds(0)); });
    char read_buf[VERIFY_BUFFER_SIZE] = {};
    ASSERT_THAT(
        LIBC_NAMESPACE::fread(read_buf, 1, VERIFY_BUFFER_SIZE - 1, file),
        Succeeds(expected.size()));
    ASSERT_EQ(LIBC_NAMESPACE::cpp::string_view(read_buf), expected);
  }
};

TEST_F(LlvmLibcExitTest, FlushesOpenStreams) {
  const auto FILENAME = libc_make_test_file_path("exit_flush.test");
  const char *fn = FILENAME;
  constexpr char CONTENT[] = "unflushed data";

  auto test = [=] {
    ::FILE *file = LIBC_NAMESPACE::fopen(fn, "w");
    if (file == nullptr)
      __builtin_trap();
    if (LIBC_NAMESPACE::fputs(CONTENT, file) < 0)
      __builtin_trap();
    // Deliberately neither flushed nor closed: exit has to do it.
    LIBC_NAMESPACE::exit(0);
  };
  EXPECT_EXITS(test, 0);
  LIBC_NAMESPACE::cpp::scope_exit cleanup(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::remove(fn), Succeeds(0)); });

  verify_file_content(fn, CONTENT);
}

// stdout is line buffered and not on the open-file list, so it gets its own
// test; the content has no newline so that only exit can flush it.
TEST_F(LlvmLibcExitTest, FlushesStdout) {
  const auto FILENAME = libc_make_test_file_path("exit_flush_stdout.test");
  const char *fn = FILENAME;
  constexpr char CONTENT[] = "unflushed stdout data";

  auto test = [=] {
    if (LIBC_NAMESPACE::freopen(fn, "w", LIBC_NAMESPACE::stdout) == nullptr)
      __builtin_trap();
    if (LIBC_NAMESPACE::fputs(CONTENT, LIBC_NAMESPACE::stdout) < 0)
      __builtin_trap();
    LIBC_NAMESPACE::exit(0);
  };
  EXPECT_EXITS(test, 0);
  LIBC_NAMESPACE::cpp::scope_exit cleanup(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::remove(fn), Succeeds(0)); });

  verify_file_content(fn, CONTENT);
}

static ::FILE *atexit_file = nullptr;

TEST_F(LlvmLibcExitTest, FlushesAfterAtexitHandlers) {
  const auto FILENAME = libc_make_test_file_path("exit_flush_atexit.test");
  const char *fn = FILENAME;

  auto test = [=] {
    atexit_file = LIBC_NAMESPACE::fopen(fn, "w");
    if (atexit_file == nullptr)
      __builtin_trap();
    int status = LIBC_NAMESPACE::atexit(+[] {
      if (LIBC_NAMESPACE::fputs("atexit", atexit_file) < 0)
        __builtin_trap();
    });
    if (status != 0)
      __builtin_trap();
    if (LIBC_NAMESPACE::fputs("main ", atexit_file) < 0)
      __builtin_trap();
    LIBC_NAMESPACE::exit(0);
  };
  EXPECT_EXITS(test, 0);
  LIBC_NAMESPACE::cpp::scope_exit cleanup(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::remove(fn), Succeeds(0)); });

  verify_file_content(fn, "main atexit");
}

#endif // LIBC_COPT_EXIT_FLUSH_STREAMS
