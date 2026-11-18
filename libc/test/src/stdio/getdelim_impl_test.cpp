//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for the getdelim_impl helper.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/func/free.h"
#include "hdr/func/malloc.h"
#include "hdr/stdint_proxy.h"
#include "hdr/stdio_macros.h"
#include "hdr/types/size_t.h"
#include "hdr/types/ssize_t.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/File/file.h"
#include "src/__support/File/file_io_result.h"
#include "src/__support/File/file_mode.h"
#include "src/__support/File/platform_file.h"
#include "src/__support/error_or.h"
#include "src/stdio/generic/getdelim_impl.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "test/UnitTest/Test.h"

namespace {

using LIBC_NAMESPACE::CustomFile;
using LIBC_NAMESPACE::File;
using LIBC_NAMESPACE::FileIOResult;
using LIBC_NAMESPACE::FileMode;
using LIBC_NAMESPACE::cpp::scope_exit;
using LIBC_NAMESPACE::cpp::span;
using LIBC_NAMESPACE::cpp::string_view;

class TestFile : public CustomFile {
  span<const char> data;
  size_t pos = 0;
  size_t fail_at_byte = LIBC_NAMESPACE::cpp::numeric_limits<size_t>::max();
  int fail_errno = 0;
  uint8_t stream_buf[32]{};

  static FileIOResult read_func(File *f, void *dst, size_t len) {
    auto *tf = static_cast<TestFile *>(f);
    if (tf->fail_at_byte <= tf->pos)
      return {0, tf->fail_errno};

    if (tf->pos >= tf->data.size())
      return 0;

    size_t remaining = tf->data.size() - tf->pos;
    size_t count = len < remaining ? len : remaining;
    if (tf->fail_at_byte < tf->pos + count)
      count = tf->fail_at_byte - tf->pos;

    LIBC_NAMESPACE::inline_memcpy(dst, tf->data.data() + tf->pos, count);
    tf->pos += count;
    return count;
  }

  static int close_func(File *) { return 0; }

public:
  explicit TestFile(span<const char> bytes, int bufmode = _IOFBF)
      : CustomFile({/*w=*/nullptr, &read_func, /*s=*/nullptr, &close_func},
                   bufmode == _IONBF ? nullptr : stream_buf,
                   bufmode == _IONBF ? 0 : sizeof(stream_buf), bufmode,
                   /*owned=*/false, FileMode("r")),
        data(bytes) {}

  explicit TestFile(string_view str, int bufmode = _IOFBF)
      : TestFile(span<const char>(str.data(), str.size()), bufmode) {}

  ~TestFile() { (void)close(); }

  void set_fail_at(size_t byte_offset, int err_val) {
    fail_at_byte = byte_offset;
    fail_errno = err_val;
  }

  void set_data(string_view str) {
    data = span<const char>(str.data(), str.size());
  }

  size_t raw_pos() const { return pos; }
};

} // namespace

TEST(LlvmLibcGetdelimImplTest, CustomColonDelimiter) {
  TestFile file(string_view("root:x:0:0\n"));
  char *line = nullptr;
  size_t cap = 0;
  scope_exit cleanup([&] { ::free(line); });

  auto res = LIBC_NAMESPACE::internal::getdelim_impl(&line, &cap, ':', &file);
  ASSERT_TRUE(res.has_value());
  EXPECT_EQ(res.value(), ssize_t(5));
  EXPECT_STREQ(line, "root:");
  EXPECT_GE(cap, size_t(6));

  res = LIBC_NAMESPACE::internal::getdelim_impl(&line, &cap, ':', &file);
  ASSERT_TRUE(res.has_value());
  EXPECT_EQ(res.value(), ssize_t(2));
  EXPECT_STREQ(line, "x:");

  res = LIBC_NAMESPACE::internal::getdelim_impl(&line, &cap, ':', &file);
  ASSERT_TRUE(res.has_value());
  EXPECT_EQ(res.value(), ssize_t(2));
  EXPECT_STREQ(line, "0:");

  res = LIBC_NAMESPACE::internal::getdelim_impl(&line, &cap, ':', &file);
  ASSERT_TRUE(res.has_value());
  EXPECT_EQ(res.value(), ssize_t(2));
  EXPECT_STREQ(line, "0\n");
  EXPECT_TRUE(file.iseof());
  EXPECT_FALSE(file.error());

  res = LIBC_NAMESPACE::internal::getdelim_impl(&line, &cap, ':', &file);
  ASSERT_TRUE(res.has_value());
  EXPECT_EQ(res.value(), ssize_t(-1));
}

TEST(LlvmLibcGetdelimImplTest, NullByteDelimiter) {
  constexpr char RAW[] = "first\0second\0third";
  TestFile file(span<const char>(RAW, sizeof(RAW) - 1));
  char *line = nullptr;
  size_t cap = 0;
  scope_exit cleanup([&] { ::free(line); });

  auto res = LIBC_NAMESPACE::internal::getdelim_impl(&line, &cap, '\0', &file);
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(res.value(), ssize_t(6));
  EXPECT_EQ(string_view(line, 6), string_view("first\0", 6));
  EXPECT_EQ(line[6], '\0');

  res = LIBC_NAMESPACE::internal::getdelim_impl(&line, &cap, '\0', &file);
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(res.value(), ssize_t(7));
  EXPECT_EQ(string_view(line, 7), string_view("second\0", 7));
  EXPECT_EQ(line[7], '\0');

  res = LIBC_NAMESPACE::internal::getdelim_impl(&line, &cap, '\0', &file);
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(res.value(), ssize_t(5));
  EXPECT_STREQ(line, "third");

  res = LIBC_NAMESPACE::internal::getdelim_impl(&line, &cap, '\0', &file);
  ASSERT_TRUE(res.has_value());
  EXPECT_EQ(res.value(), ssize_t(-1));
}

TEST(LlvmLibcGetdelimImplTest, HighBitBytesAndDelimiter) {
  constexpr char RAW[] = "ab\xff"
                         "cd\n";
  TestFile newline_file(span<const char>(RAW, sizeof(RAW) - 1));
  char *line = nullptr;
  size_t cap = 0;
  scope_exit cleanup([&] { ::free(line); });

  // Byte 0xFF in input must not be confused with EOF.
  auto res =
      LIBC_NAMESPACE::internal::getdelim_impl(&line, &cap, '\n', &newline_file);
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(res.value(), ssize_t(6));
  EXPECT_EQ(string_view(line, 6), string_view(RAW, 6));

  // Byte 0xFF passed as unsigned char value works as a delimiter.
  TestFile ff_file(span<const char>(RAW, sizeof(RAW) - 1));
  const int ff_delim = static_cast<int>(static_cast<unsigned char>('\xff'));
  res =
      LIBC_NAMESPACE::internal::getdelim_impl(&line, &cap, ff_delim, &ff_file);
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(res.value(), ssize_t(3));
  EXPECT_EQ(string_view(line, 3), string_view("ab\xff", 3));
  EXPECT_EQ(line[3], '\0');

  res =
      LIBC_NAMESPACE::internal::getdelim_impl(&line, &cap, ff_delim, &ff_file);
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(res.value(), ssize_t(3));
  EXPECT_STREQ(line, "cd\n");
}

TEST(LlvmLibcGetdelimImplTest, InvalidArgumentsSetErrorIndicator) {
  TestFile file1(string_view("hello\n"));
  size_t cap = 0;
  auto res1 =
      LIBC_NAMESPACE::internal::getdelim_impl(nullptr, &cap, '\n', &file1);
  ASSERT_FALSE(res1.has_value());
  EXPECT_EQ(res1.error(), EINVAL);
  EXPECT_TRUE(file1.error());
  EXPECT_EQ(file1.raw_pos(), size_t(0));

  TestFile file2(string_view("hello\n"));
  char *line = nullptr;
  auto res2 =
      LIBC_NAMESPACE::internal::getdelim_impl(&line, nullptr, '\n', &file2);
  ASSERT_FALSE(res2.has_value());
  EXPECT_EQ(res2.error(), EINVAL);
  EXPECT_TRUE(file2.error());
  EXPECT_EQ(line, nullptr);
  EXPECT_EQ(file2.raw_pos(), size_t(0));
}

TEST(LlvmLibcGetdelimImplTest, WideOrientedStreamFailsEvenIfEofSet) {
  TestFile wide_file(string_view(""));
  wchar_t wc = 0;
  // Wide read on empty file sets orientation to WIDE and latches EOF.
  (void)wide_file.read(&wc, 1);
  ASSERT_EQ(wide_file.get_orientation(), File::Orientation::WIDE);
  ASSERT_TRUE(wide_file.iseof());

  char *line = nullptr;
  size_t cap = 0;
  auto res =
      LIBC_NAMESPACE::internal::getdelim_impl(&line, &cap, '\n', &wide_file);
  ASSERT_FALSE(res.has_value());
  EXPECT_EQ(res.error(), EINVAL);
  EXPECT_TRUE(wide_file.error());
  EXPECT_EQ(line, nullptr);
  EXPECT_EQ(cap, size_t(0));
}

TEST(LlvmLibcGetdelimImplTest, StickyEofShortCircuitsWithoutReading) {
  TestFile file(string_view(""), _IONBF);
  char *line = nullptr;
  size_t cap = 0;

  // First call hits EOF and sets the stream's EOF indicator.
  auto res = LIBC_NAMESPACE::internal::getdelim_impl(&line, &cap, '\n', &file);
  ASSERT_TRUE(res.has_value());
  EXPECT_EQ(res.value(), ssize_t(-1));
  ASSERT_TRUE(file.iseof());

  // Make new data available without clearing EOF; getdelim_impl must not read.
  file.set_data(string_view("unread data\n"));
  res = LIBC_NAMESPACE::internal::getdelim_impl(&line, &cap, '\n', &file);
  ASSERT_TRUE(res.has_value());
  EXPECT_EQ(res.value(), ssize_t(-1));
  EXPECT_EQ(file.raw_pos(), size_t(0));
  EXPECT_FALSE(file.error());
  EXPECT_EQ(line, nullptr);
  EXPECT_EQ(cap, size_t(0));
}

TEST(LlvmLibcGetdelimImplTest, ReadErrorAtStartPreservesNullBuffer) {
  TestFile file(string_view("hello\n"));
  file.set_fail_at(0, EIO);

  char *line = nullptr;
  size_t cap = 0;
  auto res = LIBC_NAMESPACE::internal::getdelim_impl(&line, &cap, '\n', &file);
  ASSERT_FALSE(res.has_value());
  EXPECT_EQ(res.error(), EIO);
  EXPECT_TRUE(file.error());
  EXPECT_EQ(line, nullptr);
  EXPECT_EQ(cap, size_t(0));
}

TEST(LlvmLibcGetdelimImplTest, ReadErrorMidwayAfterGrowthUpdatesCallerBuffer) {
  constexpr char CHARS[] = "abcXYZ0123_!";
  char input[160];
  for (size_t i = 0; i < sizeof(input); ++i)
    input[i] = CHARS[i % (sizeof(CHARS) - 1)];

  TestFile file(span<const char>(input, sizeof(input)));
  file.set_fail_at(100, EIO);

  size_t cap = 4;
  char *line = static_cast<char *>(::malloc(cap));
  ASSERT_NE(line, nullptr);
  scope_exit cleanup([&] { ::free(line); });

  auto res = LIBC_NAMESPACE::internal::getdelim_impl(&line, &cap, '\n', &file);
  ASSERT_FALSE(res.has_value());
  EXPECT_EQ(res.error(), EIO);
  EXPECT_TRUE(file.error());
  // The original 4-byte buffer was freed during growth; line and cap must
  // reflect the newly grown buffer so cleanup's free(line) is valid.
  EXPECT_NE(line, nullptr);
  EXPECT_GE(cap, size_t(128));
}

TEST(LlvmLibcGetdelimImplTest, ExactMaxCapacitySucceedsAndExceedingOverflows) {
  char *line = nullptr;
  size_t cap = 0;
  scope_exit cleanup([&] { ::free(line); });

  // MaxCapacity == 6 holds exactly 5 chars ("abcd\n") plus NUL terminator.
  TestFile ok_file(string_view("abcd\n"));
  auto ok_res =
      LIBC_NAMESPACE::internal::getdelim_impl<6>(&line, &cap, '\n', &ok_file);
  ASSERT_TRUE(ok_res.has_value());
  EXPECT_EQ(ok_res.value(), ssize_t(5));
  EXPECT_STREQ(line, "abcd\n");
  EXPECT_EQ(cap, size_t(6));
  EXPECT_FALSE(ok_file.error());

  // 6 chars ("abcde\n") requires capacity 7 > MaxCapacity == 6 -> EOVERFLOW.
  TestFile overflow_file(string_view("abcde\n"));
  auto err_res = LIBC_NAMESPACE::internal::getdelim_impl<6>(&line, &cap, '\n',
                                                            &overflow_file);
  ASSERT_FALSE(err_res.has_value());
  EXPECT_EQ(err_res.error(), EOVERFLOW);
  EXPECT_TRUE(overflow_file.error());
  EXPECT_NE(line, nullptr);
  EXPECT_EQ(cap, size_t(6));
}

TEST(LlvmLibcGetdelimImplTest, CapacityClampingAndSizeMaxOverflowSafety) {
  constexpr size_t MAX_SIZE =
      LIBC_NAMESPACE::cpp::numeric_limits<size_t>::max();

  // Doubling near SIZE_MAX must saturate to MaxCapacity without wrapping.
  using MaxBuf = LIBC_NAMESPACE::internal::LineBuffer<MAX_SIZE>;
  static_assert(MaxBuf::next_capacity(0) == 64);
  static_assert(MaxBuf::next_capacity(MAX_SIZE / 2) == (MAX_SIZE / 2) * 2);
  static_assert(MaxBuf::next_capacity((MAX_SIZE / 2) + 1) == MAX_SIZE);
  static_assert(MaxBuf::next_capacity(MAX_SIZE - 1) == MAX_SIZE);

  using MinBuf = LIBC_NAMESPACE::internal::LineBuffer<2>;
  static_assert(MinBuf::next_capacity(0) == 2);

  char *raw = nullptr;
  size_t raw_cap = 0;
  {
    MaxBuf buf(raw, raw_cap);
    ASSERT_TRUE(buf.push_back('x'));
    buf.null_terminate();
  }
  ASSERT_NE(raw, nullptr);
  EXPECT_STREQ(raw, "x");
  ::free(raw);

  // Runtime check of current_capacity > MaxCapacity / 2 clamping: with
  // MaxCapacity == 71, initial allocation is 64 and the next growth clamps to
  // 71 instead of 128.
  constexpr char CHARS[] = "abcXYZ0123_!";
  char input[70];
  for (size_t i = 0; i < sizeof(input) - 1; ++i)
    input[i] = CHARS[i % (sizeof(CHARS) - 1)];
  input[sizeof(input) - 1] = '\n';

  TestFile clamped_file(span<const char>(input, sizeof(input)));
  char *line = nullptr;
  size_t cap = 0;
  scope_exit cleanup([&] { ::free(line); });

  auto res = LIBC_NAMESPACE::internal::getdelim_impl<71>(&line, &cap, '\n',
                                                         &clamped_file);
  ASSERT_TRUE(res.has_value());
  EXPECT_EQ(res.value(), ssize_t(70));
  EXPECT_EQ(cap, size_t(71));
  EXPECT_EQ(string_view(line, 70), string_view(input, 70));
  EXPECT_EQ(line[70], '\0');
}
