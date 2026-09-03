//===-- Unittests for the printf String Writer ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/printf_core/writer.h"

#include "src/__support/CPP/string_view.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "test/UnitTest/Test.h"

namespace {

using LIBC_NAMESPACE::cpp::string_view;
using LIBC_NAMESPACE::printf_core::make_drop_overflow_writer;
using LIBC_NAMESPACE::printf_core::make_writer;
using LIBC_NAMESPACE::printf_core::overflow_write_flush_to_sink;
using LIBC_NAMESPACE::printf_core::OverflowMode;
using LIBC_NAMESPACE::printf_core::WRITE_OK;
using LIBC_NAMESPACE::printf_core::WriteBuffer;
using LIBC_NAMESPACE::printf_core::Writer;

TEST(LlvmLibcPrintfWriterTest, Constructor) {
  char str[10];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  EXPECT_EQ(writer.get_write_buffer().buff, str);
}

TEST(LlvmLibcPrintfWriterTest, Write) {
  char str[4] = {'D', 'E', 'F', 'G'};
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<char> &wb = writer.get_write_buffer();
  writer.write({"abc", 3});

  EXPECT_EQ(str[3], 'G');

  // The string must be null terminated manually since the writer cannot tell
  // when it's done.
  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ("abc", str);
  ASSERT_EQ(writer.get_chars_written(), size_t{3});
}

TEST(LlvmLibcPrintfWriterTest, WriteMultipleTimes) {
  char str[10];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<char> &wb = writer.get_write_buffer();
  writer.write({"abc", 3});
  writer.write({"DEF", 3});
  writer.write({"1234", 3});

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ("abcDEF123", str);
  ASSERT_EQ(writer.get_chars_written(), size_t{9});
}

TEST(LlvmLibcPrintfWriterTest, WriteChars) {
  char str[4] = {'D', 'E', 'F', 'G'};
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<char> &wb = writer.get_write_buffer();
  writer.write('a', 3);

  EXPECT_EQ(str[3], 'G');
  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ("aaa", str);
  ASSERT_EQ(writer.get_chars_written(), size_t{3});
}

TEST(LlvmLibcPrintfWriterTest, WriteCharsMultipleTimes) {
  char str[10];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<char> &wb = writer.get_write_buffer();
  writer.write('a', 3);
  writer.write('D', 3);
  writer.write('1', 3);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ("aaaDDD111", str);
  ASSERT_EQ(writer.get_chars_written(), size_t{9});
}

TEST(LlvmLibcPrintfWriterTest, WriteManyChars) {
  char str[100];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<char> &wb = writer.get_write_buffer();
  writer.write('Z', 99);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ("ZZZZZZZZZZ"
               "ZZZZZZZZZZ"
               "ZZZZZZZZZZ"
               "ZZZZZZZZZZ"
               "ZZZZZZZZZZ"
               "ZZZZZZZZZZ"
               "ZZZZZZZZZZ"
               "ZZZZZZZZZZ"
               "ZZZZZZZZZZ"
               "ZZZZZZZZZ",
               str);
  ASSERT_EQ(writer.get_chars_written(), size_t{99});
}

TEST(LlvmLibcPrintfWriterTest, MixedWrites) {
  char str[13];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<char> &wb = writer.get_write_buffer();
  writer.write('a', 3);
  writer.write({"DEF", 3});
  writer.write('1', 3);
  writer.write({"456", 3});

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ("aaaDEF111456", str);
  ASSERT_EQ(writer.get_chars_written(), size_t{12});
}

TEST(LlvmLibcPrintfWriterTest, WriteWithMaxLength) {
  char str[11];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<char> &wb = writer.get_write_buffer();
  writer.write({"abcDEF123456", 12});

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ("abcDEF1234", str);
  ASSERT_EQ(writer.get_chars_written(), size_t{12});
}

TEST(LlvmLibcPrintfWriterTest, WriteCharsWithMaxLength) {
  char str[11];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<char> &wb = writer.get_write_buffer();
  writer.write('1', 15);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ("1111111111", str);
  ASSERT_EQ(writer.get_chars_written(), size_t{15});
}

TEST(LlvmLibcPrintfWriterTest, MixedWriteWithMaxLength) {
  char str[11];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<char> &wb = writer.get_write_buffer();
  writer.write('a', 3);
  writer.write({"DEF", 3});
  writer.write('1', 3);
  writer.write({"456", 3});

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ("aaaDEF1114", str);
  ASSERT_EQ(writer.get_chars_written(), size_t{12});
}

TEST(LlvmLibcPrintfWriterTest, StringWithMaxLengthOne) {
  char str[1];
  // This is because the max length should be at most 1 less than the size of
  // the buffer it's writing to.
  Writer writer = make_drop_overflow_writer(str, 0);
  WriteBuffer<char> &wb = writer.get_write_buffer();
  writer.write('a', 3);
  writer.write({"DEF", 3});
  writer.write('1', 3);
  writer.write({"456", 3});

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ("", str);
  ASSERT_EQ(writer.get_chars_written(), size_t{12});
}

TEST(LlvmLibcPrintfWriterTest, NullStringWithZeroMaxLength) {
  Writer writer = make_drop_overflow_writer<char>(nullptr, 0);
  writer.write('a', 3);
  writer.write({"DEF", 3});
  writer.write('1', 3);
  writer.write({"456", 3});

  ASSERT_EQ(writer.get_chars_written(), size_t{12});
}

struct OutBuff {
  char *out_str;
  size_t cur_pos = 0;
};

int copy_to_out(string_view new_str, void *raw_out_buff) {
  OutBuff *out_buff = reinterpret_cast<OutBuff *>(raw_out_buff);

  LIBC_NAMESPACE::inline_memcpy(out_buff->out_str + out_buff->cur_pos,
                                new_str.data(), new_str.size());

  out_buff->cur_pos += new_str.size();
  return WRITE_OK;
}

TEST(LlvmLibcPrintfWriterTest, WriteWithMaxLengthWithCallback) {
  char str[16];

  OutBuff out_buff = {str, 0};

  char wb_buff[8];
  Writer writer =
      make_writer(wb_buff, sizeof(wb_buff) - 1,
                  &overflow_write_flush_to_sink<char, copy_to_out>, &out_buff);
  writer.write({"abcDEF123456", 12});

  writer.get_write_buffer().flush_to_sink<copy_to_out>(&out_buff);
  str[out_buff.cur_pos] = '\0';

  ASSERT_STREQ("abcDEF123456", str);
  ASSERT_EQ(writer.get_chars_written(), size_t{12});
}

TEST(LlvmLibcPrintfWriterTest, WriteCharsWithMaxLengthWithCallback) {
  char str[16];

  OutBuff out_buff = {str, 0};

  char wb_buff[8];
  Writer writer =
      make_writer(wb_buff, sizeof(wb_buff) - 1,
                  &overflow_write_flush_to_sink<char, copy_to_out>, &out_buff);
  writer.write('1', 15);

  writer.get_write_buffer().flush_to_sink<copy_to_out>(&out_buff);
  str[out_buff.cur_pos] = '\0';

  ASSERT_STREQ("111111111111111", str);
  ASSERT_EQ(writer.get_chars_written(), size_t{15});
}

TEST(LlvmLibcPrintfWriterTest, MixedWriteWithMaxLengthWithCallback) {
  char str[16];

  OutBuff out_buff = {str, 0};

  char wb_buff[8];
  Writer writer =
      make_writer(wb_buff, sizeof(wb_buff) - 1,
                  &overflow_write_flush_to_sink<char, copy_to_out>, &out_buff);
  writer.write('a', 3);
  writer.write({"DEF", 3});
  writer.write('1', 3);
  writer.write({"456", 3});

  writer.get_write_buffer().flush_to_sink<copy_to_out>(&out_buff);
  str[out_buff.cur_pos] = '\0';

  ASSERT_STREQ("aaaDEF111456", str);
  ASSERT_EQ(writer.get_chars_written(), size_t{12});
}

TEST(LlvmLibcPrintfWriterTest, ZeroLengthBufferWithCallback) {
  char str[16];

  OutBuff out_buff = {str, 0};

  char wb_buff[1];
  Writer writer =
      make_writer(wb_buff, sizeof(wb_buff) - 1,
                  &overflow_write_flush_to_sink<char, copy_to_out>, &out_buff);
  writer.write('a', 3);
  writer.write({"DEF", 3});
  writer.write('1', 3);
  writer.write({"456", 3});

  writer.get_write_buffer().flush_to_sink<copy_to_out>(&out_buff);
  str[out_buff.cur_pos] = '\0';

  ASSERT_STREQ("aaaDEF111456", str);
  ASSERT_EQ(writer.get_chars_written(), size_t{12});
}

TEST(LlvmLibcPrintfWriterTest, NullStringWithZeroMaxLengthWithCallback) {
  char str[16];

  OutBuff out_buff = {str, 0};

  Writer writer =
      make_writer(static_cast<char *>(nullptr), 0,
                  &overflow_write_flush_to_sink<char, copy_to_out>, &out_buff);
  writer.write('a', 3);
  writer.write({"DEF", 3});
  writer.write('1', 3);
  writer.write({"456", 3});

  writer.get_write_buffer().flush_to_sink<copy_to_out>(&out_buff);
  str[out_buff.cur_pos] = '\0';

  ASSERT_EQ(writer.get_chars_written(), size_t{12});
  ASSERT_STREQ("aaaDEF111456", str);
}

} // namespace
