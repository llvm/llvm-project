//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for line_reader.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/size_t.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/File/file.h"
#include "src/__support/File/line_reader.h"
#include "src/__support/dynamic_buffer.h"
#include "src/stdio/remove.h"
#include "test/UnitTest/MemoryMatcher.h"
#include "test/UnitTest/Test.h"

namespace {

using MemoryView = LIBC_NAMESPACE::testing::MemoryView;

LIBC_NAMESPACE::File *open_temp(const char *path,
                                LIBC_NAMESPACE::cpp::string_view content,
                                const char *mode = "r") {
  auto w = LIBC_NAMESPACE::openfile(path, "w");
  if (!w.has_value())
    return nullptr;
  w.value()->write(content.data(), content.size());
  w.value()->close();
  auto r = LIBC_NAMESPACE::openfile(path, mode);
  return r.has_value() ? r.value() : nullptr;
}

} // namespace

TEST(LlvmLibcLineReaderTest, EmptySpanAndWriteOnlyStream) {
  {
    const char *path = libc_make_test_file_path("line_reader_empty_span.test");
    LIBC_NAMESPACE::File *f = open_temp(path, "\na\n");
    ASSERT_NE(f, nullptr);
    LIBC_NAMESPACE::cpp::scope_exit cleanup([&]() {
      f->close();
      LIBC_NAMESPACE::remove(path);
    });

    LIBC_NAMESPACE::cpp::span<char> empty;
    auto blank = LIBC_NAMESPACE::read_line(f, empty);
    ASSERT_TRUE(blank.has_value());
    EXPECT_EQ(blank->bytes_read, size_t{0});
    EXPECT_FALSE(blank->truncated);
    EXPECT_FALSE(blank->eof);

    auto trunc = LIBC_NAMESPACE::read_line(f, empty);
    ASSERT_TRUE(trunc.has_value());
    EXPECT_EQ(trunc->bytes_read, size_t{0});
    EXPECT_TRUE(trunc->truncated);
    EXPECT_FALSE(trunc->eof);

    auto eof = LIBC_NAMESPACE::read_line(f, empty);
    ASSERT_TRUE(eof.has_value());
    EXPECT_EQ(eof->bytes_read, size_t{0});
    EXPECT_FALSE(eof->truncated);
    EXPECT_TRUE(eof->eof);
  }

  {
    const char *path = libc_make_test_file_path("line_reader_wonly.test");
    LIBC_NAMESPACE::File *f = open_temp(path, "hello\n", "w");
    ASSERT_NE(f, nullptr);
    LIBC_NAMESPACE::cpp::scope_exit cleanup([&]() {
      f->close();
      LIBC_NAMESPACE::remove(path);
    });

    char buf[8];
    auto bad_span = LIBC_NAMESPACE::read_line(f, buf);
    ASSERT_FALSE(bad_span.has_value());
    EXPECT_EQ(bad_span.error(), EBADF);

    LIBC_NAMESPACE::ScopedDynamicBuffer dyn_buf;
    auto bad_dyn = LIBC_NAMESPACE::read_line(f, dyn_buf);
    ASSERT_FALSE(bad_dyn.has_value());
    EXPECT_EQ(bad_dyn.error(), EBADF);
    EXPECT_EQ(dyn_buf.capacity(), size_t{0});
  }
}

TEST(LlvmLibcLineReaderTest, FixedSpanReadTruncateAndEof) {
  const char *path = libc_make_test_file_path("line_reader_span.test");
  LIBC_NAMESPACE::File *f = open_temp(path, "abcd\n\nefghi\nok");
  ASSERT_NE(f, nullptr);
  LIBC_NAMESPACE::cpp::scope_exit cleanup([&]() {
    f->close();
    LIBC_NAMESPACE::remove(path);
  });

  // 4-byte span: "abcd\n" fits, "\n" is blank, "efghi\n" truncates and drains,
  // "ok" has no trailing newline, then EOF.
  char buf[4];

  auto r1 = LIBC_NAMESPACE::read_line(f, buf);
  ASSERT_TRUE(r1.has_value());
  EXPECT_EQ(r1->bytes_read, size_t{4});
  EXPECT_FALSE(r1->truncated);
  EXPECT_FALSE(r1->eof);
  EXPECT_EQ(LIBC_NAMESPACE::cpp::string_view(buf, r1->bytes_read),
            LIBC_NAMESPACE::cpp::string_view("abcd"));

  auto r2 = LIBC_NAMESPACE::read_line(f, buf);
  ASSERT_TRUE(r2.has_value());
  EXPECT_EQ(r2->bytes_read, size_t{0});
  EXPECT_FALSE(r2->truncated);
  EXPECT_FALSE(r2->eof);

  auto r3 = LIBC_NAMESPACE::read_line(f, buf);
  ASSERT_TRUE(r3.has_value());
  EXPECT_EQ(r3->bytes_read, size_t{4});
  EXPECT_TRUE(r3->truncated);
  EXPECT_EQ(LIBC_NAMESPACE::cpp::string_view(buf, r3->bytes_read),
            LIBC_NAMESPACE::cpp::string_view("efgh"));

  auto r4 = LIBC_NAMESPACE::read_line(f, buf);
  ASSERT_TRUE(r4.has_value());
  EXPECT_EQ(r4->bytes_read, size_t{2});
  EXPECT_FALSE(r4->truncated);
  EXPECT_FALSE(r4->eof);
  EXPECT_EQ(LIBC_NAMESPACE::cpp::string_view(buf, r4->bytes_read),
            LIBC_NAMESPACE::cpp::string_view("ok"));

  auto r5 = LIBC_NAMESPACE::read_line(f, buf);
  ASSERT_TRUE(r5.has_value());
  EXPECT_EQ(r5->bytes_read, size_t{0});
  EXPECT_FALSE(r5->truncated);
  EXPECT_TRUE(r5->eof);
}

TEST(LlvmLibcLineReaderTest, EmbeddedNullAndCarriageReturnPreserved) {
  constexpr char RAW[] = "ab\0cd\r\n";
  const char *path = libc_make_test_file_path("line_reader_nul.test");
  LIBC_NAMESPACE::File *f =
      open_temp(path, LIBC_NAMESPACE::cpp::string_view(RAW, sizeof(RAW) - 1));
  ASSERT_NE(f, nullptr);
  LIBC_NAMESPACE::cpp::scope_exit cleanup([&]() {
    f->close();
    LIBC_NAMESPACE::remove(path);
  });

  char buf[16];
  auto r = LIBC_NAMESPACE::read_line(f, buf);
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(r->bytes_read, size_t{6});
  EXPECT_MEM_EQ(MemoryView("ab\0cd\r", 6), MemoryView(buf, r->bytes_read));
}

TEST(LlvmLibcLineReaderTest, DynamicBufferGrowthReuseAndEmptyEof) {
  LIBC_NAMESPACE::ScopedDynamicBuffer dyn_buf;

  // Immediate EOF on a fresh DynamicBuffer does not allocate.
  {
    const char *empty_path = libc_make_test_file_path("line_reader_empty.test");
    LIBC_NAMESPACE::File *empty_f = open_temp(empty_path, "");
    ASSERT_NE(empty_f, nullptr);
    LIBC_NAMESPACE::cpp::scope_exit cleanup_empty([&]() {
      empty_f->close();
      LIBC_NAMESPACE::remove(empty_path);
    });

    auto eof = LIBC_NAMESPACE::read_line(empty_f, dyn_buf);
    ASSERT_TRUE(eof.has_value());
    EXPECT_TRUE(eof->eof);
    EXPECT_EQ(dyn_buf.capacity(), size_t{0});
  }

  constexpr size_t LONG_LEN = 2500;
  char content[LONG_LEN + 8];
  for (size_t i = 0; i < LONG_LEN; ++i)
    content[i] = static_cast<char>('a' + (i % 26));
  content[LONG_LEN] = '\n';
  content[LONG_LEN + 1] = 'h';
  content[LONG_LEN + 2] = 'i';

  const char *long_path = libc_make_test_file_path("line_reader_long.test");
  LIBC_NAMESPACE::File *long_f = open_temp(
      long_path, LIBC_NAMESPACE::cpp::string_view(content, LONG_LEN + 3));
  ASSERT_NE(long_f, nullptr);
  LIBC_NAMESPACE::cpp::scope_exit cleanup_long([&]() {
    long_f->close();
    LIBC_NAMESPACE::remove(long_path);
  });

  auto r1 = LIBC_NAMESPACE::read_line(long_f, dyn_buf);
  ASSERT_TRUE(r1.has_value());
  EXPECT_EQ(r1->bytes_read, LONG_LEN);
  EXPECT_FALSE(r1->truncated);
  EXPECT_MEM_EQ(MemoryView(content, LONG_LEN),
                MemoryView(dyn_buf.span().data(), r1->bytes_read));

  const size_t CAP = dyn_buf.capacity();
  auto r2 = LIBC_NAMESPACE::read_line(long_f, dyn_buf);
  ASSERT_TRUE(r2.has_value());
  EXPECT_EQ(r2->bytes_read, size_t{2});
  EXPECT_EQ(dyn_buf.capacity(), CAP);
  EXPECT_EQ(
      LIBC_NAMESPACE::cpp::string_view(dyn_buf.span().data(), r2->bytes_read),
      LIBC_NAMESPACE::cpp::string_view("hi"));
}
