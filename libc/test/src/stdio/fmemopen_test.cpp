//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for fmemopen.
///
//===----------------------------------------------------------------------===//

#include "hdr/stdio_macros.h"
#include "hdr/types/off_t.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/scope.h"
#include "src/stdio/clearerr.h"
#include "src/stdio/fclose.h"
#include "src/stdio/feof.h"
#include "src/stdio/ferror.h"
#include "src/stdio/fflush.h"
#include "src/stdio/fgetc.h"
#include "src/stdio/fmemopen.h"
#include "src/stdio/fread.h"
#include "src/stdio/fseek.h"
#include "src/stdio/fseeko.h"
#include "src/stdio/ftell.h"
#include "src/stdio/fwrite.h"
#include "src/stdio/setvbuf.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/MemoryMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcFMemOpenTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;
using MemoryView = LIBC_NAMESPACE::testing::MemoryView;
using LIBC_NAMESPACE::cpp::scope_exit;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST_F(LlvmLibcFMemOpenTest, InitialPositionAndEnd) {
  struct Mode {
    const char *name;
    long position;
    long end;
  };
  const Mode modes[] = {
      {"r", 0, 8}, {"rb", 0, 8}, {"r+", 0, 8}, {"rb+", 0, 8}, {"r+b", 0, 8},
      {"w", 0, 0}, {"wb", 0, 0}, {"w+", 0, 0}, {"wb+", 0, 0}, {"w+b", 0, 0},
      {"a", 2, 2}, {"ab", 2, 2}, {"a+", 2, 2}, {"ab+", 2, 2}, {"a+b", 2, 2},
  };
  for (const auto &mode : modes) {
    char storage[] = {'A', 'B', '\0', 'x', 'x', 'x', 'x', 'x'};
    ::FILE *f = LIBC_NAMESPACE::fmemopen(storage, sizeof(storage), mode.name);
    ASSERT_TRUE(f != nullptr);
    scope_exit close([&] { EXPECT_EQ(0, LIBC_NAMESPACE::fclose(f)); });
    EXPECT_EQ(mode.position, LIBC_NAMESPACE::ftell(f));
    EXPECT_EQ(mode.name[0] == 'w' ? '\0' : 'A', storage[0]);
    EXPECT_EQ(0, LIBC_NAMESPACE::fseek(f, 0, SEEK_END));
    EXPECT_EQ(mode.end, LIBC_NAMESPACE::ftell(f));
  }
}

TEST_F(LlvmLibcFMemOpenTest, ReadIncludesNullAndStopsAtEnd) {
  char storage[] = {'A', '\0', 'B', 'C'};
  char output[] = {'?', '?', '?', '?', '?', '?'};
  ::FILE *f = LIBC_NAMESPACE::fmemopen(storage, sizeof(storage), "r");
  ASSERT_TRUE(f != nullptr);
  scope_exit close([&] { EXPECT_EQ(0, LIBC_NAMESPACE::fclose(f)); });
  EXPECT_EQ(size_t(4), LIBC_NAMESPACE::fread(output, 1, sizeof(output), f));
  const char expected[] = {'A', '\0', 'B', 'C', '?', '?'};
  EXPECT_MEM_EQ(MemoryView(expected, sizeof(expected)),
                MemoryView(output, sizeof(output)));
  EXPECT_NE(0, LIBC_NAMESPACE::feof(f));
  EXPECT_EQ(0, LIBC_NAMESPACE::ferror(f));
  EXPECT_EQ(size_t(0), LIBC_NAMESPACE::fread(output, 1, 1, f));
  EXPECT_EQ(4L, LIBC_NAMESPACE::ftell(f));
  EXPECT_EQ(0, LIBC_NAMESPACE::fseek(f, 0, SEEK_SET));
  EXPECT_EQ(0, LIBC_NAMESPACE::feof(f));
  EXPECT_EQ(int('A'), LIBC_NAMESPACE::fgetc(f));
}

TEST_F(LlvmLibcFMemOpenTest, WriteAndOverwrite) {
  char storage[8];
  ::FILE *f = LIBC_NAMESPACE::fmemopen(storage, sizeof(storage), "w+");
  ASSERT_TRUE(f != nullptr);
  scope_exit close([&] { EXPECT_EQ(0, LIBC_NAMESPACE::fclose(f)); });
  EXPECT_EQ(size_t(5), LIBC_NAMESPACE::fwrite("ABCDE", 1, 5, f));
  EXPECT_EQ(0, LIBC_NAMESPACE::fflush(f));
  EXPECT_STREQ("ABCDE", storage);
  EXPECT_EQ(0, LIBC_NAMESPACE::fseek(f, 1, SEEK_SET));
  EXPECT_EQ(size_t(1), LIBC_NAMESPACE::fwrite("X", 1, 1, f));
  EXPECT_EQ(0, LIBC_NAMESPACE::fflush(f));
  EXPECT_STREQ("AXCDE", storage);
  EXPECT_EQ(0, LIBC_NAMESPACE::fseek(f, 0, SEEK_END));
  EXPECT_EQ(5L, LIBC_NAMESPACE::ftell(f));
  EXPECT_EQ(size_t(1), LIBC_NAMESPACE::fwrite("F", 1, 1, f));
  // Seeking flushes output and allows an update stream to switch to reading.
  EXPECT_EQ(0, LIBC_NAMESPACE::fseek(f, 0, SEEK_SET));
  char output[8];
  EXPECT_EQ(size_t(6), LIBC_NAMESPACE::fread(output, 1, sizeof(output), f));
  EXPECT_MEM_EQ(MemoryView("AXCDEF", 6), MemoryView(output, 6));
}

TEST_F(LlvmLibcFMemOpenTest, ExactCapacityAndClose) {
  const int buffer_modes[] = {_IONBF, _IOFBF, _IOLBF};
  for (int buffering : buffer_modes) {
    char guarded[] = {'L', '?', '?', '?', '?', 'R'};
    ::FILE *f = LIBC_NAMESPACE::fmemopen(guarded + 1, 4, "w");
    ASSERT_TRUE(f != nullptr);
    EXPECT_EQ(0, LIBC_NAMESPACE::setvbuf(f, nullptr, buffering, 16));
    EXPECT_EQ(size_t(4), LIBC_NAMESPACE::fwrite("ABCD", 1, 4, f));
    EXPECT_EQ(0, LIBC_NAMESPACE::fclose(f));
    const char expected[] = {'L', 'A', 'B', 'C', 'D', 'R'};
    EXPECT_MEM_EQ(MemoryView(expected, sizeof(expected)),
                  MemoryView(guarded, sizeof(guarded)));
  }
}

TEST_F(LlvmLibcFMemOpenTest, CloseFlushesPendingOutput) {
  char storage[8];
  ::FILE *f = LIBC_NAMESPACE::fmemopen(storage, sizeof(storage), "w");
  ASSERT_TRUE(f != nullptr);
  EXPECT_EQ(size_t(3), LIBC_NAMESPACE::fwrite("ABC", 1, 3, f));
  EXPECT_EQ(0, LIBC_NAMESPACE::fclose(f));
  EXPECT_STREQ("ABC", storage);
}

TEST_F(LlvmLibcFMemOpenTest, AppendAfterSeekAndRead) {
  const int buffer_modes[] = {_IONBF, _IOFBF};
  for (int buffering : buffer_modes) {
    char storage[8] = "AB";
    ::FILE *f = LIBC_NAMESPACE::fmemopen(storage, sizeof(storage), "a+");
    ASSERT_TRUE(f != nullptr);
    scope_exit close([&] { EXPECT_EQ(0, LIBC_NAMESPACE::fclose(f)); });
    ASSERT_EQ(0, LIBC_NAMESPACE::setvbuf(f, nullptr, buffering, 16));
    EXPECT_EQ(0, LIBC_NAMESPACE::fseek(f, 0, SEEK_SET));
    EXPECT_EQ(int('A'), LIBC_NAMESPACE::fgetc(f));
    EXPECT_EQ(0, LIBC_NAMESPACE::fseek(f, 0, SEEK_CUR));
    EXPECT_EQ(size_t(1), LIBC_NAMESPACE::fwrite("X", 1, 1, f));
    EXPECT_EQ(0, LIBC_NAMESPACE::fflush(f));
    EXPECT_EQ(3L, LIBC_NAMESPACE::ftell(f));
    EXPECT_STREQ("ABX", storage);
    // Even a position beyond the current end must not create an append hole.
    EXPECT_EQ(0, LIBC_NAMESPACE::fseek(f, 7, SEEK_SET));
    EXPECT_EQ(size_t(1), LIBC_NAMESPACE::fwrite("Y", 1, 1, f));
    EXPECT_EQ(0, LIBC_NAMESPACE::fflush(f));
    EXPECT_STREQ("ABXY", storage);
  }
}

TEST_F(LlvmLibcFMemOpenTest, AppendTracksEndAcrossNullBytes) {
  char storage[8] = "";
  ::FILE *f = LIBC_NAMESPACE::fmemopen(storage, sizeof(storage), "a+");
  ASSERT_TRUE(f != nullptr);
  scope_exit close([&] { EXPECT_EQ(0, LIBC_NAMESPACE::fclose(f)); });
  EXPECT_EQ(0L, LIBC_NAMESPACE::ftell(f));
  const char first[] = {'A', '\0', 'B'};
  EXPECT_EQ(sizeof(first), LIBC_NAMESPACE::fwrite(first, 1, sizeof(first), f));
  EXPECT_EQ(0, LIBC_NAMESPACE::fflush(f));
  EXPECT_EQ(size_t(1), LIBC_NAMESPACE::fwrite("C", 1, 1, f));
  EXPECT_EQ(0, LIBC_NAMESPACE::fflush(f));
  const char expected[] = {'A', '\0', 'B', 'C', '\0'};
  EXPECT_MEM_EQ(MemoryView(expected, sizeof(expected)),
                MemoryView(storage, sizeof(expected)));
  EXPECT_EQ(0, LIBC_NAMESPACE::fseek(f, 0, SEEK_END));
  EXPECT_EQ(4L, LIBC_NAMESPACE::ftell(f));
}

TEST_F(LlvmLibcFMemOpenTest, SeekBoundsAndEnd) {
  char storage[8];
  ::FILE *f = LIBC_NAMESPACE::fmemopen(storage, sizeof(storage), "w+");
  ASSERT_TRUE(f != nullptr);
  scope_exit close([&] { EXPECT_EQ(0, LIBC_NAMESPACE::fclose(f)); });

  EXPECT_EQ(size_t(3), LIBC_NAMESPACE::fwrite("ABC", 1, 3, f));
  ASSERT_THAT(LIBC_NAMESPACE::fseek(f, 8, SEEK_SET), Succeeds());
  EXPECT_EQ(8L, LIBC_NAMESPACE::ftell(f));

  ASSERT_THAT(LIBC_NAMESPACE::fseek(f, 9, SEEK_SET), Fails(EINVAL));
  EXPECT_EQ(8L, LIBC_NAMESPACE::ftell(f));

  ASSERT_THAT(LIBC_NAMESPACE::fseek(f, -1, SEEK_SET), Fails(EINVAL));

  ASSERT_THAT(LIBC_NAMESPACE::fseek(f, 0, SEEK_END), Succeeds());
  EXPECT_EQ(3L, LIBC_NAMESPACE::ftell(f));

  ASSERT_THAT(LIBC_NAMESPACE::fseek(f, 2, SEEK_END), Succeeds());
  EXPECT_EQ(5L, LIBC_NAMESPACE::ftell(f));

  // Reading past the data end must not advance the memory stream's position.
  EXPECT_EQ(EOF, LIBC_NAMESPACE::fgetc(f));
  EXPECT_NE(0, LIBC_NAMESPACE::feof(f));
  EXPECT_EQ(0, LIBC_NAMESPACE::ferror(f));

  ASSERT_THAT(LIBC_NAMESPACE::fseek(f, 0, SEEK_CUR), Succeeds());
  EXPECT_EQ(5L, LIBC_NAMESPACE::ftell(f));

  ASSERT_THAT(LIBC_NAMESPACE::fseek(f, -1, SEEK_END), Succeeds());
  EXPECT_EQ(2L, LIBC_NAMESPACE::ftell(f));

  const off_t offsets[] = {LIBC_NAMESPACE::cpp::numeric_limits<off_t>::min(),
                           LIBC_NAMESPACE::cpp::numeric_limits<off_t>::max()};
  for (off_t offset : offsets) {
    ASSERT_THAT(LIBC_NAMESPACE::fseeko(f, offset, SEEK_CUR), Fails(EINVAL));
    EXPECT_EQ(2L, LIBC_NAMESPACE::ftell(f));
  }

  ASSERT_THAT(LIBC_NAMESPACE::fseek(f, 6, SEEK_SET), Succeeds());
  EXPECT_EQ(size_t(1), LIBC_NAMESPACE::fwrite("Z", 1, 1, f));

  ASSERT_THAT(LIBC_NAMESPACE::fseek(f, 0, SEEK_END), Succeeds());
  EXPECT_EQ(7L, LIBC_NAMESPACE::ftell(f));
  EXPECT_EQ('Z', storage[6]);
  EXPECT_EQ('\0', storage[7]);

  ASSERT_THAT(LIBC_NAMESPACE::fseek(f, 0, -1), Fails(EINVAL));
  // The contents of the gap between the old end and position 6 are unspecified.
}

TEST_F(LlvmLibcFMemOpenTest, InternalBuffer) {
  const char *update_modes[] = {"w+", "a+", "r+"};
  for (const char *mode : update_modes) {
    ::FILE *f = LIBC_NAMESPACE::fmemopen(nullptr, 16, mode);
    ASSERT_TRUE(f != nullptr);
    scope_exit close([&] { EXPECT_EQ(0, LIBC_NAMESPACE::fclose(f)); });
    EXPECT_EQ(0L, LIBC_NAMESPACE::ftell(f));
    EXPECT_EQ(0, LIBC_NAMESPACE::fseek(f, 0, SEEK_END));
    EXPECT_EQ(mode[0] == 'r' ? 16L : 0L, LIBC_NAMESPACE::ftell(f));
    EXPECT_EQ(0, LIBC_NAMESPACE::fseek(f, 0, SEEK_SET));
    EXPECT_EQ(size_t(3), LIBC_NAMESPACE::fwrite("ABC", 1, 3, f));
    EXPECT_EQ(0, LIBC_NAMESPACE::fseek(f, 0, SEEK_SET));
    char output[3];
    EXPECT_EQ(sizeof(output),
              LIBC_NAMESPACE::fread(output, 1, sizeof(output), f));
    EXPECT_MEM_EQ(MemoryView("ABC", 3), MemoryView(output, 3));
  }
  // Non-update modes are accepted, just as with an externally supplied buffer.
  const char *other_modes[] = {"r", "w", "a"};
  for (const char *mode : other_modes) {
    ::FILE *f = LIBC_NAMESPACE::fmemopen(nullptr, 4, mode);
    ASSERT_TRUE(f != nullptr);
    EXPECT_EQ(0L, LIBC_NAMESPACE::ftell(f));
    EXPECT_EQ(0, LIBC_NAMESPACE::fclose(f));
  }
}

TEST_F(LlvmLibcFMemOpenTest, ZeroCapacity) {
  char guard = 'X';
  void *buffers[] = {&guard, nullptr};
  for (void *buf : buffers) {
    const char *modes[] = {"r+", "w+", "a+"};
    for (const char *mode : modes) {
      ::FILE *f = LIBC_NAMESPACE::fmemopen(buf, 0, mode);
      ASSERT_TRUE(f != nullptr);
      scope_exit close([&] { EXPECT_EQ(0, LIBC_NAMESPACE::fclose(f)); });
      ASSERT_EQ(0, LIBC_NAMESPACE::setvbuf(f, nullptr, _IOFBF, 1));
      EXPECT_EQ(0L, LIBC_NAMESPACE::ftell(f));
      EXPECT_EQ(EOF, LIBC_NAMESPACE::fgetc(f));
      EXPECT_NE(0, LIBC_NAMESPACE::feof(f));
      EXPECT_EQ(0, LIBC_NAMESPACE::fseek(f, 0, SEEK_SET));
      EXPECT_EQ(size_t(0), LIBC_NAMESPACE::fwrite("ABC", 1, 3, f));
      ASSERT_ERRNO_EQ(ENOSPC);
      EXPECT_NE(0, LIBC_NAMESPACE::ferror(f));
      EXPECT_EQ(0, LIBC_NAMESPACE::fseek(f, 0, SEEK_END));
      EXPECT_EQ(0L, LIBC_NAMESPACE::ftell(f));
      EXPECT_NE(0, LIBC_NAMESPACE::fseek(f, 1, SEEK_SET));
      ASSERT_ERRNO_EQ(EINVAL);
      EXPECT_EQ('X', guard);
    }
  }
}

TEST_F(LlvmLibcFMemOpenTest, ShortWriteAndRecovery) {
  char guarded[] = {'L', '?', '?', '?', '?', 'R'};
  ::FILE *f = LIBC_NAMESPACE::fmemopen(guarded + 1, 4, "w+");
  ASSERT_TRUE(f != nullptr);
  scope_exit close([&] { EXPECT_EQ(0, LIBC_NAMESPACE::fclose(f)); });
  ASSERT_EQ(0, LIBC_NAMESPACE::setvbuf(f, nullptr, _IOFBF, 1));
  EXPECT_EQ(size_t(4), LIBC_NAMESPACE::fwrite("ABCDEF", 1, 6, f));
  ASSERT_ERRNO_EQ(ENOSPC);
  EXPECT_NE(0, LIBC_NAMESPACE::ferror(f));
  EXPECT_EQ(4L, LIBC_NAMESPACE::ftell(f));
  EXPECT_EQ(size_t(0), LIBC_NAMESPACE::fwrite("XYZ", 1, 3, f));
  ASSERT_ERRNO_EQ(ENOSPC);
  LIBC_NAMESPACE::clearerr(f);
  EXPECT_EQ(0, LIBC_NAMESPACE::ferror(f));
  EXPECT_EQ(0, LIBC_NAMESPACE::fseek(f, 0, SEEK_SET));
  EXPECT_EQ(size_t(1), LIBC_NAMESPACE::fwrite("X", 1, 1, f));
  EXPECT_EQ(0, LIBC_NAMESPACE::fflush(f));
  const char expected[] = {'L', 'X', 'B', 'C', 'D', 'R'};
  EXPECT_MEM_EQ(MemoryView(expected, sizeof(expected)),
                MemoryView(guarded, sizeof(guarded)));
}
