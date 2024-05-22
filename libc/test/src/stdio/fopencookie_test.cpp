//===-- Unittests for the fopencookie function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/clearerr.h"
#include "src/stdio/fclose.h"
#include "src/stdio/feof.h"
#include "src/stdio/ferror.h"
#include "src/stdio/fflush.h"
#include "src/stdio/fopencookie.h"
#include "src/stdio/fread.h"
#include "src/stdio/fseek.h"
#include "src/stdio/fwrite.h"
#include "test/UnitTest/MemoryMatcher.h"
#include "test/UnitTest/Test.h"

#include "src/errno/libc_errno.h"
#include <stdio.h>
#include <stdlib.h>

using MemoryView = LIBC_NAMESPACE::testing::MemoryView;

struct StringStream {
  char *buf;
  size_t bufsize; // Size of buf
  size_t endpos;  // 1 more than current fill size
  size_t offset;  // Current read/write location
};

ssize_t write_ss(void *cookie, const char *buf, size_t size) {
  auto *ss = reinterpret_cast<StringStream *>(cookie);
  if (ss->offset + size > ss->bufsize)
    ss->buf =
        reinterpret_cast<char *>(realloc(ss->buf, (ss->offset + size) * 2));
  for (size_t i = 0; i < size; ++i, ss->offset += 1)
    ss->buf[ss->offset] = buf[i];
  if (ss->offset > ss->endpos)
    ss->endpos = ss->offset;
  return size;
}

ssize_t read_ss(void *cookie, char *buf, size_t size) {
  auto *ss = reinterpret_cast<StringStream *>(cookie);
  ssize_t copysize = size;
  if (ss->offset + size > ss->endpos) {
    // You cannot copy more than what you have available.
    copysize = ss->endpos - ss->offset;
    if (copysize < 0)
      copysize = 0; // A seek could have moved offset past the endpos
  }
  for (size_t i = 0; i < size_t(copysize); ++i, ++ss->offset)
    buf[i] = ss->buf[ss->offset];
  return copysize;
}

int seek_ss(void *cookie, off64_t *offset, int whence) {
  auto *ss = reinterpret_cast<StringStream *>(cookie);
  off64_t new_offset;
  if (whence == SEEK_SET) {
    new_offset = *offset;
  } else if (whence == SEEK_CUR) {
    new_offset = *offset + ss->offset;
  } else if (whence == SEEK_END) {
    new_offset = *offset + ss->endpos;
  } else {
    libc_errno = EINVAL;
    return -1;
  }
  if (new_offset < 0 || size_t(new_offset) > ss->bufsize)
    return -1;
  ss->offset = new_offset;
  *offset = new_offset;
  return 0;
}

int close_ss(void *cookie) {
  auto *ss = reinterpret_cast<StringStream *>(cookie);
  free(ss->buf);
  ss->buf = nullptr;
  ss->bufsize = ss->endpos = ss->offset = 0;
  return 0;
}

constexpr cookie_io_functions_t STRING_STREAM_FUNCS = {&read_ss, &write_ss,
                                                       &seek_ss, &close_ss};

TEST(LlvmLibcFOpenCookie, ReadOnlyCookieTest) {
  constexpr char CONTENT[] = "Hello,readonly!";
  auto *ss = reinterpret_cast<StringStream *>(malloc(sizeof(StringStream)));
  ss->buf = reinterpret_cast<char *>(malloc(sizeof(CONTENT)));
  ss->bufsize = sizeof(CONTENT);
  ss->offset = 0;
  ss->endpos = ss->bufsize;
  for (size_t i = 0; i < sizeof(CONTENT); ++i)
    ss->buf[i] = CONTENT[i];

  ::FILE *f = LIBC_NAMESPACE::fopencookie(ss, "r", STRING_STREAM_FUNCS);
  ASSERT_TRUE(f != nullptr);
  char read_data[sizeof(CONTENT)];
  ASSERT_EQ(sizeof(CONTENT),
            LIBC_NAMESPACE::fread(read_data, 1, sizeof(CONTENT), f));
  ASSERT_STREQ(read_data, CONTENT);

  // Reading another time should trigger eof.
  ASSERT_NE(sizeof(CONTENT),
            LIBC_NAMESPACE::fread(read_data, 1, sizeof(CONTENT), f));
  ASSERT_NE(LIBC_NAMESPACE::feof(f), 0);

  ASSERT_EQ(0, LIBC_NAMESPACE::fseek(f, 0, SEEK_SET));
  // Should be an error to write.
  ASSERT_EQ(size_t(0), LIBC_NAMESPACE::fwrite(CONTENT, 1, sizeof(CONTENT), f));
  ASSERT_NE(LIBC_NAMESPACE::ferror(f), 0);
  ASSERT_NE(libc_errno, 0);
  libc_errno = 0;

  LIBC_NAMESPACE::clearerr(f);
  ASSERT_EQ(LIBC_NAMESPACE::ferror(f), 0);

  ASSERT_EQ(0, LIBC_NAMESPACE::fclose(f));
  free(ss);
}

TEST(LlvmLibcFOpenCookie, WriteOnlyCookieTest) {
  size_t INIT_BUFSIZE = 32;
  auto *ss = reinterpret_cast<StringStream *>(malloc(sizeof(StringStream)));
  ss->buf = reinterpret_cast<char *>(malloc(INIT_BUFSIZE));
  ss->bufsize = INIT_BUFSIZE;
  ss->offset = 0;
  ss->endpos = 0;

  ::FILE *f = LIBC_NAMESPACE::fopencookie(ss, "w", STRING_STREAM_FUNCS);
  ASSERT_TRUE(f != nullptr);

  constexpr char WRITE_DATA[] = "Hello,writeonly!";
  ASSERT_EQ(sizeof(WRITE_DATA),
            LIBC_NAMESPACE::fwrite(WRITE_DATA, 1, sizeof(WRITE_DATA), f));
  // Flushing will ensure the data to be written to the string stream.
  ASSERT_EQ(0, LIBC_NAMESPACE::fflush(f));
  ASSERT_STREQ(WRITE_DATA, ss->buf);

  ASSERT_EQ(0, LIBC_NAMESPACE::fseek(f, 0, SEEK_SET));
  char read_data[sizeof(WRITE_DATA)];
  // Should be an error to read.
  ASSERT_EQ(size_t(0),
            LIBC_NAMESPACE::fread(read_data, 1, sizeof(WRITE_DATA), f));
  ASSERT_NE(LIBC_NAMESPACE::ferror(f), 0);
  ASSERT_EQ(libc_errno, EBADF);
  libc_errno = 0;

  LIBC_NAMESPACE::clearerr(f);
  ASSERT_EQ(LIBC_NAMESPACE::ferror(f), 0);

  ASSERT_EQ(0, LIBC_NAMESPACE::fclose(f));
  free(ss);
}

TEST(LlvmLibcFOpenCookie, AppendOnlyCookieTest) {
  constexpr char INITIAL_CONTENT[] = "1234567890987654321";
  constexpr char WRITE_DATA[] = "append";
  auto *ss = reinterpret_cast<StringStream *>(malloc(sizeof(StringStream)));
  ss->buf = reinterpret_cast<char *>(malloc(sizeof(INITIAL_CONTENT)));
  ss->bufsize = sizeof(INITIAL_CONTENT);
  ss->offset = ss->bufsize; // We want to open the file in append mode.
  ss->endpos = ss->bufsize;
  for (size_t i = 0; i < sizeof(INITIAL_CONTENT); ++i)
    ss->buf[i] = INITIAL_CONTENT[i];

  ::FILE *f = LIBC_NAMESPACE::fopencookie(ss, "a", STRING_STREAM_FUNCS);
  ASSERT_TRUE(f != nullptr);

  constexpr size_t READ_SIZE = 5;
  char read_data[READ_SIZE];
  // This is not a readable file.
  ASSERT_EQ(LIBC_NAMESPACE::fread(read_data, 1, READ_SIZE, f), size_t(0));
  ASSERT_NE(LIBC_NAMESPACE::ferror(f), 0);
  EXPECT_NE(libc_errno, 0);
  libc_errno = 0;

  LIBC_NAMESPACE::clearerr(f);
  ASSERT_EQ(LIBC_NAMESPACE::ferror(f), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fwrite(WRITE_DATA, 1, sizeof(WRITE_DATA), f),
            sizeof(WRITE_DATA));
  EXPECT_EQ(LIBC_NAMESPACE::fflush(f), 0);
  EXPECT_EQ(ss->endpos, sizeof(WRITE_DATA) + sizeof(INITIAL_CONTENT));

  ASSERT_EQ(LIBC_NAMESPACE::fclose(f), 0);
  free(ss);
}

TEST(LlvmLibcFOpenCookie, ReadUpdateCookieTest) {
  const char INITIAL_CONTENT[] = "1234567890987654321";
  auto *ss = reinterpret_cast<StringStream *>(malloc(sizeof(StringStream)));
  ss->buf = reinterpret_cast<char *>(malloc(sizeof(INITIAL_CONTENT)));
  ss->bufsize = sizeof(INITIAL_CONTENT);
  ss->offset = 0;
  ss->endpos = ss->bufsize;
  for (size_t i = 0; i < sizeof(INITIAL_CONTENT); ++i)
    ss->buf[i] = INITIAL_CONTENT[i];

  ::FILE *f = LIBC_NAMESPACE::fopencookie(ss, "r+", STRING_STREAM_FUNCS);
  ASSERT_TRUE(f != nullptr);

  constexpr size_t READ_SIZE = sizeof(INITIAL_CONTENT) / 2;
  char read_data[READ_SIZE];
  ASSERT_EQ(READ_SIZE, LIBC_NAMESPACE::fread(read_data, 1, READ_SIZE, f));

  MemoryView src1(INITIAL_CONTENT, READ_SIZE), dst1(read_data, READ_SIZE);
  EXPECT_MEM_EQ(src1, dst1);

  ASSERT_EQ(LIBC_NAMESPACE::fseek(f, 0, SEEK_SET), 0);
  constexpr char WRITE_DATA[] = "hello, file";
  ASSERT_EQ(sizeof(WRITE_DATA),
            LIBC_NAMESPACE::fwrite(WRITE_DATA, 1, sizeof(WRITE_DATA), f));
  ASSERT_EQ(LIBC_NAMESPACE::fflush(f), 0);
  EXPECT_STREQ(ss->buf, WRITE_DATA);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(f), 0);
  free(ss);
}

TEST(LlvmLibcFOpenCookie, WriteUpdateCookieTest) {
  constexpr char WRITE_DATA[] = "hello, file";
  auto *ss = reinterpret_cast<StringStream *>(malloc(sizeof(StringStream)));
  ss->buf = reinterpret_cast<char *>(malloc(sizeof(WRITE_DATA)));
  ss->bufsize = sizeof(WRITE_DATA);
  ss->offset = 0;
  ss->endpos = 0;

  ::FILE *f = LIBC_NAMESPACE::fopencookie(ss, "w+", STRING_STREAM_FUNCS);
  ASSERT_TRUE(f != nullptr);

  ASSERT_EQ(sizeof(WRITE_DATA),
            LIBC_NAMESPACE::fwrite(WRITE_DATA, 1, sizeof(WRITE_DATA), f));

  ASSERT_EQ(LIBC_NAMESPACE::fseek(f, 0, SEEK_SET), 0);

  char read_data[sizeof(WRITE_DATA)];
  ASSERT_EQ(LIBC_NAMESPACE::fread(read_data, 1, sizeof(read_data), f),
            sizeof(read_data));
  EXPECT_STREQ(read_data, WRITE_DATA);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(f), 0);
  free(ss);
}
