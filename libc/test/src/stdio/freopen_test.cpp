//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for freopen.
///
//===----------------------------------------------------------------------===//

#include "src/fcntl/fcntl.h"
#include "src/stdio/clearerr.h"
#include "src/stdio/fclose.h"
#include "src/stdio/feof.h"
#include "src/stdio/ferror.h"
#include "src/stdio/fflush.h"
#include "src/stdio/fileno.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fread.h"
#include "src/stdio/freopen.h"
#include "src/stdio/fwrite.h"
#include "src/stdio/stdout.h"
#include "src/unistd/close.h"
#include "src/wchar/fwide.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include "hdr/fcntl_macros.h"
#include "hdr/stdio_macros.h"
#include "src/__support/macros/properties/os.h"

using LlvmLibcFreopenTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST_F(LlvmLibcFreopenTest, ReopenFile) {
  const auto FILENAME_A =
      libc_make_test_file_path(APPEND_LIBC_TEST("freopen_a.test"));
  const auto FILENAME_B =
      libc_make_test_file_path(APPEND_LIBC_TEST("freopen_b.test"));

  // Step 1: Open file A and write initial content.
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME_A, "w");
  ASSERT_FALSE(file == nullptr);

  constexpr char CONTENT_A[] = "File A Content";
  ASSERT_EQ(LIBC_NAMESPACE::fwrite(CONTENT_A, 1, sizeof(CONTENT_A) - 1, file),
            sizeof(CONTENT_A) - 1);
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Step 2: Open file A for reading.
  file = LIBC_NAMESPACE::fopen(FILENAME_A, "r");
  ASSERT_FALSE(file == nullptr);

  // Step 3: Use freopen to redirect stream from file A to file B for writing.
  ::FILE *reopened_file = LIBC_NAMESPACE::freopen(FILENAME_B, "w", file);
  ASSERT_NE(reopened_file, static_cast<::FILE *>(nullptr));
  ASSERT_EQ(reopened_file, file);

  // Step 4: Write to reopened stream (file B).
  constexpr char CONTENT_B[] = "File B Content Written via freopen";
  ASSERT_EQ(LIBC_NAMESPACE::fwrite(CONTENT_B, 1, sizeof(CONTENT_B) - 1,
                                   reopened_file),
            sizeof(CONTENT_B) - 1);
  ASSERT_EQ(LIBC_NAMESPACE::fclose(reopened_file), 0);

  // Step 5: Verify file B content.
  file = LIBC_NAMESPACE::fopen(FILENAME_B, "r");
  ASSERT_FALSE(file == nullptr);
  char read_buf[sizeof(CONTENT_B)] = {0};
  ASSERT_EQ(LIBC_NAMESPACE::fread(read_buf, 1, sizeof(CONTENT_B) - 1, file),
            sizeof(CONTENT_B) - 1);
  ASSERT_STREQ(read_buf, CONTENT_B);
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST_F(LlvmLibcFreopenTest, NullFilenameModeChange) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("freopen_null_filename.test"));

  // Step 1: Open file with write-update mode.
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w+");
  ASSERT_FALSE(file == nullptr);

  constexpr char INITIAL_CONTENT[] = "Initial Data ";
  ASSERT_EQ(LIBC_NAMESPACE::fwrite(INITIAL_CONTENT, 1,
                                   sizeof(INITIAL_CONTENT) - 1, file),
            sizeof(INITIAL_CONTENT) - 1);

  // Step 2: Change mode with filename == nullptr to append.
  ::FILE *reopened = LIBC_NAMESPACE::freopen(nullptr, "a", file);
  ASSERT_NE(reopened, static_cast<::FILE *>(nullptr));
  ASSERT_EQ(reopened, file);

  // Step 3: Write appended content.
  constexpr char APPENDED_CONTENT[] = "Appended Data";
  ASSERT_EQ(LIBC_NAMESPACE::fwrite(APPENDED_CONTENT, 1,
                                   sizeof(APPENDED_CONTENT) - 1, reopened),
            sizeof(APPENDED_CONTENT) - 1);
  ASSERT_EQ(LIBC_NAMESPACE::fclose(reopened), 0);

  // Step 4: Verify combined content.
  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);
  char read_buf[64] = {0};
  size_t read_bytes =
      LIBC_NAMESPACE::fread(read_buf, 1, sizeof(read_buf) - 1, file);
  read_buf[read_bytes] = '\0';
  ASSERT_STREQ(read_buf, "Initial Data Appended Data");
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST_F(LlvmLibcFreopenTest, NullFilenameInvalidModeChange) {
  const auto FILENAME = libc_make_test_file_path(
      APPEND_LIBC_TEST("freopen_invalid_mode_change.test"));

  // Open file read-only.
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  // Attempt incompatible mode change (r to w with filename == nullptr).
  // This failing freopen deallocates the stream.
  ASSERT_THAT(LIBC_NAMESPACE::freopen(nullptr, "w", file),
              Fails(EBADF, static_cast<void *>(nullptr)));

  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  // Attempt incompatible mode change (r to w+ with filename == nullptr).
  // This failing freopen also deallocates the stream.
  ASSERT_THAT(LIBC_NAMESPACE::freopen(nullptr, "w+", file),
              Fails(EBADF, static_cast<void *>(nullptr)));
}

#if defined(LIBC_TARGET_OS_IS_POSIX) || defined(LIBC_TARGET_OS_IS_LINUX)
TEST_F(LlvmLibcFreopenTest, InvalidModeFailure) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("freopen_invalid_mode.test"));

  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  int old_fd = LIBC_NAMESPACE::fileno(file);
  ASSERT_GT(old_fd, 0);

  ASSERT_THAT(LIBC_NAMESPACE::freopen(FILENAME, "invalid_mode_str", file),
              Fails(EINVAL, static_cast<void *>(nullptr)));

  // Per POSIX spec, original stream fd was closed on filename != nullptr
  // freopen attempt, and the stream object was deallocated.
  ASSERT_EQ(LIBC_NAMESPACE::fcntl(old_fd, F_GETFL), -1);
  ASSERT_ERRNO_EQ(EBADF);
}
#endif

#if defined(LIBC_TARGET_OS_IS_POSIX) || defined(LIBC_TARGET_OS_IS_LINUX)
TEST_F(LlvmLibcFreopenTest, NonExistentFileFailure) {
  auto EXISTING_FILE =
      libc_make_test_file_path(APPEND_LIBC_TEST("freopen_existing.test"));
  auto NON_EXISTENT_FILE =
      libc_make_test_file_path(APPEND_LIBC_TEST("freopen_does_not_exist.test"));

  ::FILE *file = LIBC_NAMESPACE::fopen(EXISTING_FILE, "w");
  ASSERT_FALSE(file == nullptr);

  int old_fd = LIBC_NAMESPACE::fileno(file);
  ASSERT_GT(old_fd, 0);

  // Attempt to freopen a non-existent file in read mode.
  ASSERT_THAT(LIBC_NAMESPACE::freopen(NON_EXISTENT_FILE, "r", file),
              Fails(ENOENT, static_cast<void *>(nullptr)));

  // Per POSIX spec: The original stream fd is closed even if open fails,
  // and the stream object is deallocated.
  ASSERT_EQ(LIBC_NAMESPACE::fcntl(old_fd, F_GETFL), -1);
  ASSERT_ERRNO_EQ(EBADF);
}
#endif // LIBC_TARGET_OS_IS_POSIX || LIBC_TARGET_OS_IS_LINUX

TEST_F(LlvmLibcFreopenTest, FlushBeforeReopenTest) {
  const auto FILENAME_A =
      libc_make_test_file_path(APPEND_LIBC_TEST("freopen_flush_a.test"));
  const auto FILENAME_B =
      libc_make_test_file_path(APPEND_LIBC_TEST("freopen_flush_b.test"));

  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME_A, "w");
  ASSERT_FALSE(file == nullptr);

  constexpr char DIRTY_DATA[] = "Buffered data before freopen";
  ASSERT_EQ(LIBC_NAMESPACE::fwrite(DIRTY_DATA, 1, sizeof(DIRTY_DATA) - 1, file),
            sizeof(DIRTY_DATA) - 1);

  // freopen must flush unwritten buffered data to FILENAME_A before reopening
  ::FILE *reopened = LIBC_NAMESPACE::freopen(FILENAME_B, "w", file);
  ASSERT_NE(reopened, static_cast<::FILE *>(nullptr));
  ASSERT_EQ(LIBC_NAMESPACE::fclose(reopened), 0);

  // Verify FILENAME_A received the flushed dirty buffer
  file = LIBC_NAMESPACE::fopen(FILENAME_A, "r");
  ASSERT_FALSE(file == nullptr);
  char read_buf[sizeof(DIRTY_DATA)] = {0};
  ASSERT_EQ(LIBC_NAMESPACE::fread(read_buf, 1, sizeof(DIRTY_DATA) - 1, file),
            sizeof(DIRTY_DATA) - 1);
  ASSERT_STREQ(read_buf, DIRTY_DATA);
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST_F(LlvmLibcFreopenTest, ClearFlagsTest) {
  const auto FILENAME_A =
      libc_make_test_file_path(APPEND_LIBC_TEST("freopen_flags_a.test"));
  const auto FILENAME_B =
      libc_make_test_file_path(APPEND_LIBC_TEST("freopen_flags_b.test"));

  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME_A, "w");
  ASSERT_FALSE(file == nullptr);
  constexpr char SHORT_DATA[] = "X";
  ASSERT_EQ(LIBC_NAMESPACE::fwrite(SHORT_DATA, 1, 1, file),
            static_cast<size_t>(1));
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Trigger EOF on file
  file = LIBC_NAMESPACE::fopen(FILENAME_A, "r");
  ASSERT_FALSE(file == nullptr);
  char buf[4];
  LIBC_NAMESPACE::fread(buf, 1, sizeof(buf), file);
  ASSERT_NE(LIBC_NAMESPACE::feof(file), 0);

  // freopen must clear EOF and error indicators
  ::FILE *reopened = LIBC_NAMESPACE::freopen(FILENAME_B, "w", file);
  ASSERT_NE(reopened, static_cast<::FILE *>(nullptr));
  ASSERT_EQ(LIBC_NAMESPACE::feof(reopened), 0);
  ASSERT_EQ(LIBC_NAMESPACE::ferror(reopened), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(reopened), 0);
}

#if defined(LIBC_TARGET_OS_IS_POSIX) || defined(LIBC_TARGET_OS_IS_LINUX)
TEST_F(LlvmLibcFreopenTest, NullFilenameBadFdTest) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("freopen_bad_fd.test"));

  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  int fd = LIBC_NAMESPACE::fileno(file);
  ASSERT_GT(fd, 0);
  // Manually close underlying fd to simulate bad file descriptor state
  ASSERT_EQ(LIBC_NAMESPACE::close(fd), 0);

  // freopen with filename == nullptr on invalid fd should return nullptr +
  // EBADF
  ASSERT_THAT(LIBC_NAMESPACE::freopen(nullptr, "a", file),
              Fails(EBADF, static_cast<void *>(nullptr)));
}
#endif // LIBC_TARGET_OS_IS_POSIX || LIBC_TARGET_OS_IS_LINUX

TEST_F(LlvmLibcFreopenTest, ResetOrientationTest) {
  const auto FILENAME_A =
      libc_make_test_file_path(APPEND_LIBC_TEST("freopen_orient_a.test"));
  const auto FILENAME_B =
      libc_make_test_file_path(APPEND_LIBC_TEST("freopen_orient_b.test"));

  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME_A, "w");
  ASSERT_FALSE(file == nullptr);

  // Set wide orientation
  ASSERT_GT(LIBC_NAMESPACE::fwide(file, 1), 0);

  // freopen must reset orientation to 0 (unoriented)
  ::FILE *reopened = LIBC_NAMESPACE::freopen(FILENAME_B, "w", file);
  ASSERT_NE(reopened, static_cast<::FILE *>(nullptr));
  ASSERT_EQ(LIBC_NAMESPACE::fwide(reopened, 0), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(reopened), 0);
}

#if defined(LIBC_TARGET_OS_IS_POSIX) || defined(LIBC_TARGET_OS_IS_LINUX)
TEST_F(LlvmLibcFreopenTest, StdoutRedirectionTest) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("freopen_stdout.test"));

  int stdout_fd = LIBC_NAMESPACE::fileno(LIBC_NAMESPACE::stdout);
  ASSERT_EQ(stdout_fd, 1);

  // Redirect stdout to FILENAME
  ::FILE *reopened =
      LIBC_NAMESPACE::freopen(FILENAME, "w", LIBC_NAMESPACE::stdout);
  ASSERT_NE(reopened, static_cast<::FILE *>(nullptr));
  ASSERT_EQ(reopened, static_cast<::FILE *>(LIBC_NAMESPACE::stdout));

  // Verify fileno(stdout) is preserved as 1
  ASSERT_EQ(LIBC_NAMESPACE::fileno(LIBC_NAMESPACE::stdout), 1);

  constexpr char MSG[] = "Redirected Stdout";
  ASSERT_EQ(
      LIBC_NAMESPACE::fwrite(MSG, 1, sizeof(MSG) - 1, LIBC_NAMESPACE::stdout),
      sizeof(MSG) - 1);
  ASSERT_EQ(LIBC_NAMESPACE::fflush(LIBC_NAMESPACE::stdout), 0);

  // Verify file content
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);
  char read_buf[sizeof(MSG)] = {0};
  ASSERT_EQ(LIBC_NAMESPACE::fread(read_buf, 1, sizeof(MSG) - 1, file),
            sizeof(MSG) - 1);
  ASSERT_STREQ(read_buf, MSG);
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}
#endif

// TODO: hermetic tests are failing with this error:
/*
/usr/bin/x86_64-linux-gnu-ld.bfd:
libc/test/src/stdio/libc.test.src.stdio.freopen_test.__hermetic__.__build__:
hidden symbol
`_ZN22__llvm_libc_23_0_0_git7testing4Test17testProcessKilledEPNS_9testutils14FunctionCallerEiPKcS6_NS0_8internal8LocationE'
isn't defined /usr/bin/x86_64-linux-gnu-ld.bfd: final link failed: bad value
*/
// TEST_F(LlvmLibcFreopenTest, NullStreamFailure) {
//   const auto FILENAME =
//       libc_make_test_file_path(APPEND_LIBC_TEST("freopen_null_stream.test"));
//   const char *fn = FILENAME;

//   EXPECT_DEATH([=] { LIBC_NAMESPACE::freopen(fn, "r", nullptr); },
//                WITH_SIGNAL(-1));
// }
