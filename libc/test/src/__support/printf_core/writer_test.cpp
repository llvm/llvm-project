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
#include "test/UnitTest/CharLiteralUtils.h"
#include "test/UnitTest/Test.h"

namespace {

using LIBC_NAMESPACE::cpp::basic_string_view;
using LIBC_NAMESPACE::printf_core::make_drop_overflow_writer;
using LIBC_NAMESPACE::printf_core::make_writer;
using LIBC_NAMESPACE::printf_core::overflow_write_flush_to_sink;
using LIBC_NAMESPACE::printf_core::OverflowMode;
using LIBC_NAMESPACE::printf_core::WRITE_OK;
using LIBC_NAMESPACE::printf_core::WriteBuffer;
using LIBC_NAMESPACE::printf_core::Writer;

using TestCharTypes = LIBC_NAMESPACE::testing::TypeList<char, wchar_t>;

TYPED_TEST(LlvmLibcPrintfWriterTest, Constructor, TestCharTypes) {
  using CharT = ParamType;
  constexpr size_t BUFFER_SIZE = 10;
  CharT str[BUFFER_SIZE];
  Writer writer = make_drop_overflow_writer(str, BUFFER_SIZE - 1);
  EXPECT_EQ(writer.get_write_buffer().buff, str);
}

TYPED_TEST(LlvmLibcPrintfWriterTest, Write, TestCharTypes) {
  using CharT = ParamType;
  constexpr size_t BUFFER_SIZE = 4;
  CharT str[BUFFER_SIZE] = {ENCODED(CharT, 'D'), ENCODED(CharT, 'E'),
                            ENCODED(CharT, 'F'), ENCODED(CharT, 'G')};
  Writer writer = make_drop_overflow_writer(str, BUFFER_SIZE - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  writer.write(ENCODED(CharT, "abc"));

  EXPECT_EQ(str[3], ENCODED(CharT, 'G'));

  // The string must be null terminated manually since the writer cannot tell
  // when it's done.
  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(ENCODED(CharT, "abc"), str);
  ASSERT_EQ(writer.get_chars_written(), size_t{3});
}

TYPED_TEST(LlvmLibcPrintfWriterTest, WriteMultipleTimes, TestCharTypes) {
  using CharT = ParamType;
  constexpr size_t BUFFER_SIZE = 10;
  CharT str[BUFFER_SIZE];
  Writer writer = make_drop_overflow_writer(str, BUFFER_SIZE - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  writer.write(ENCODED(CharT, "abc"));
  writer.write(ENCODED(CharT, "DEF"));
  writer.write(ENCODED(CharT, "123"));

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(ENCODED(CharT, "abcDEF123"), str);
  ASSERT_EQ(writer.get_chars_written(), size_t{9});
}

TYPED_TEST(LlvmLibcPrintfWriterTest, WriteChars, TestCharTypes) {
  using CharT = ParamType;
  constexpr size_t BUFFER_SIZE = 4;
  CharT str[BUFFER_SIZE] = {ENCODED(CharT, 'D'), ENCODED(CharT, 'E'),
                            ENCODED(CharT, 'F'), ENCODED(CharT, 'G')};
  Writer writer = make_drop_overflow_writer(str, BUFFER_SIZE - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  writer.write(ENCODED(CharT, 'a'), 3);

  EXPECT_EQ(str[3], ENCODED(CharT, 'G'));
  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(ENCODED(CharT, "aaa"), str);
  ASSERT_EQ(writer.get_chars_written(), size_t{3});
}

TYPED_TEST(LlvmLibcPrintfWriterTest, WriteCharsMultipleTimes, TestCharTypes) {
  using CharT = ParamType;
  constexpr size_t BUFFER_SIZE = 10;
  CharT str[BUFFER_SIZE];
  Writer writer = make_drop_overflow_writer(str, BUFFER_SIZE - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  writer.write(ENCODED(CharT, 'a'), 3);
  writer.write(ENCODED(CharT, 'D'), 3);
  writer.write(ENCODED(CharT, '1'), 3);

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(ENCODED(CharT, "aaaDDD111"), str);
  ASSERT_EQ(writer.get_chars_written(), size_t{9});
}

TYPED_TEST(LlvmLibcPrintfWriterTest, WriteManyChars, TestCharTypes) {
  using CharT = ParamType;
  constexpr size_t BUFFER_SIZE = 100;
  CharT str[BUFFER_SIZE];
  Writer writer = make_drop_overflow_writer(str, BUFFER_SIZE - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  writer.write(ENCODED(CharT, 'Z'), 99);

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(ENCODED(CharT, "ZZZZZZZZZZ"
                              "ZZZZZZZZZZ"
                              "ZZZZZZZZZZ"
                              "ZZZZZZZZZZ"
                              "ZZZZZZZZZZ"
                              "ZZZZZZZZZZ"
                              "ZZZZZZZZZZ"
                              "ZZZZZZZZZZ"
                              "ZZZZZZZZZZ"
                              "ZZZZZZZZZ"),
               str);
  ASSERT_EQ(writer.get_chars_written(), size_t{99});
}

TYPED_TEST(LlvmLibcPrintfWriterTest, MixedWrites, TestCharTypes) {
  using CharT = ParamType;
  constexpr size_t BUFFER_SIZE = 13;
  CharT str[BUFFER_SIZE];
  Writer writer = make_drop_overflow_writer(str, BUFFER_SIZE - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  writer.write(ENCODED(CharT, 'a'), 3);
  writer.write(ENCODED(CharT, "DEF"));
  writer.write(ENCODED(CharT, '1'), 3);
  writer.write(ENCODED(CharT, "456"));

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(ENCODED(CharT, "aaaDEF111456"), str);
  ASSERT_EQ(writer.get_chars_written(), size_t{12});
}

TYPED_TEST(LlvmLibcPrintfWriterTest, WriteWithMaxLength, TestCharTypes) {
  using CharT = ParamType;
  constexpr size_t BUFFER_SIZE = 11;
  CharT str[BUFFER_SIZE];
  Writer writer = make_drop_overflow_writer(str, BUFFER_SIZE - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  writer.write(ENCODED(CharT, "abcDEF123456"));

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(ENCODED(CharT, "abcDEF1234"), str);
  ASSERT_EQ(writer.get_chars_written(), size_t{12});
}

TYPED_TEST(LlvmLibcPrintfWriterTest, WriteCharsWithMaxLength, TestCharTypes) {
  using CharT = ParamType;
  constexpr size_t BUFFER_SIZE = 11;
  CharT str[BUFFER_SIZE];
  Writer writer = make_drop_overflow_writer(str, BUFFER_SIZE - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  writer.write(ENCODED(CharT, '1'), 15);

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(ENCODED(CharT, "1111111111"), str);
  ASSERT_EQ(writer.get_chars_written(), size_t{15});
}

TYPED_TEST(LlvmLibcPrintfWriterTest, MixedWriteWithMaxLength, TestCharTypes) {
  using CharT = ParamType;
  constexpr size_t BUFFER_SIZE = 11;
  CharT str[BUFFER_SIZE];
  Writer writer = make_drop_overflow_writer(str, BUFFER_SIZE - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  writer.write(ENCODED(CharT, 'a'), 3);
  writer.write(ENCODED(CharT, "DEF"));
  writer.write(ENCODED(CharT, '1'), 3);
  writer.write(ENCODED(CharT, "456"));

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(ENCODED(CharT, "aaaDEF1114"), str);
  ASSERT_EQ(writer.get_chars_written(), size_t{12});
}

TYPED_TEST(LlvmLibcPrintfWriterTest, StringWithMaxLengthOne, TestCharTypes) {
  using CharT = ParamType;
  CharT str[1];
  // This is because the max length should be at most 1 less than the size of
  // the buffer it's writing to.
  Writer writer = make_drop_overflow_writer(str, 0);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  writer.write(ENCODED(CharT, 'a'), 3);
  writer.write(ENCODED(CharT, "DEF"));
  writer.write(ENCODED(CharT, '1'), 3);
  writer.write(ENCODED(CharT, "456"));

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(ENCODED(CharT, ""), str);
  ASSERT_EQ(writer.get_chars_written(), size_t{12});
}

TYPED_TEST(LlvmLibcPrintfWriterTest, NullStringWithZeroMaxLength,
           TestCharTypes) {
  using CharT = ParamType;
  Writer writer = make_drop_overflow_writer<CharT>(nullptr, 0);
  writer.write(ENCODED(CharT, 'a'), 3);
  writer.write(ENCODED(CharT, "DEF"));
  writer.write(ENCODED(CharT, '1'), 3);
  writer.write(ENCODED(CharT, "456"));

  ASSERT_EQ(writer.get_chars_written(), size_t{12});
}

template <typename CharT> struct OutBuff {
  CharT *out_str;
  size_t cur_pos = 0;
};

template <typename CharT>
int copy_to_out(basic_string_view<CharT> new_str, void *raw_out_buff) {
  auto *out_buff = static_cast<OutBuff<CharT> *>(raw_out_buff);

  LIBC_NAMESPACE::inline_memcpy(out_buff->out_str + out_buff->cur_pos,
                                new_str.data(), new_str.size() * sizeof(CharT));

  out_buff->cur_pos += new_str.size();
  return WRITE_OK;
}

TYPED_TEST(LlvmLibcPrintfWriterTest, WriteWithMaxLengthWithCallback,
           TestCharTypes) {
  using CharT = ParamType;
  CharT str[16];

  OutBuff<CharT> out_buff = {str, 0};

  constexpr size_t WB_BUFFER_SIZE = 8;
  CharT wb_buff[WB_BUFFER_SIZE];
  Writer writer =
      make_writer(wb_buff, WB_BUFFER_SIZE - 1,
                  &overflow_write_flush_to_sink<CharT, copy_to_out>, &out_buff);
  writer.write(ENCODED(CharT, "abcDEF123456"));

  writer.get_write_buffer().template flush_to_sink<copy_to_out>(&out_buff);
  str[out_buff.cur_pos] = ENCODED(CharT, '\0');

  ASSERT_STREQ(ENCODED(CharT, "abcDEF123456"), str);
  ASSERT_EQ(writer.get_chars_written(), size_t{12});
}

TYPED_TEST(LlvmLibcPrintfWriterTest, WriteCharsWithMaxLengthWithCallback,
           TestCharTypes) {
  using CharT = ParamType;
  CharT str[16];

  OutBuff<CharT> out_buff = {str, 0};

  constexpr size_t WB_BUFFER_SIZE = 8;
  CharT wb_buff[WB_BUFFER_SIZE];
  Writer writer =
      make_writer(wb_buff, WB_BUFFER_SIZE - 1,
                  &overflow_write_flush_to_sink<CharT, copy_to_out>, &out_buff);
  writer.write(ENCODED(CharT, '1'), 15);

  writer.get_write_buffer().template flush_to_sink<copy_to_out>(&out_buff);
  str[out_buff.cur_pos] = ENCODED(CharT, '\0');

  ASSERT_STREQ(ENCODED(CharT, "111111111111111"), str);
  ASSERT_EQ(writer.get_chars_written(), size_t{15});
}

TYPED_TEST(LlvmLibcPrintfWriterTest, MixedWriteWithMaxLengthWithCallback,
           TestCharTypes) {
  using CharT = ParamType;
  CharT str[16];

  OutBuff<CharT> out_buff = {str, 0};

  constexpr size_t WB_BUFFER_SIZE = 8;
  CharT wb_buff[WB_BUFFER_SIZE];
  Writer writer =
      make_writer(wb_buff, WB_BUFFER_SIZE - 1,
                  &overflow_write_flush_to_sink<CharT, copy_to_out>, &out_buff);
  writer.write(ENCODED(CharT, 'a'), 3);
  writer.write(ENCODED(CharT, "DEF"));
  writer.write(ENCODED(CharT, '1'), 3);
  writer.write(ENCODED(CharT, "456"));

  writer.get_write_buffer().template flush_to_sink<copy_to_out>(&out_buff);
  str[out_buff.cur_pos] = ENCODED(CharT, '\0');

  ASSERT_STREQ(ENCODED(CharT, "aaaDEF111456"), str);
  ASSERT_EQ(writer.get_chars_written(), size_t{12});
}

TYPED_TEST(LlvmLibcPrintfWriterTest, ZeroLengthBufferWithCallback,
           TestCharTypes) {
  using CharT = ParamType;
  CharT str[16];

  OutBuff<CharT> out_buff = {str, 0};

  constexpr size_t WB_BUFFER_SIZE = 1;
  CharT wb_buff[WB_BUFFER_SIZE];
  Writer writer =
      make_writer(wb_buff, WB_BUFFER_SIZE - 1,
                  &overflow_write_flush_to_sink<CharT, copy_to_out>, &out_buff);
  writer.write(ENCODED(CharT, 'a'), 3);
  writer.write(ENCODED(CharT, "DEF"));
  writer.write(ENCODED(CharT, '1'), 3);
  writer.write(ENCODED(CharT, "456"));

  writer.get_write_buffer().template flush_to_sink<copy_to_out>(&out_buff);
  str[out_buff.cur_pos] = ENCODED(CharT, '\0');

  ASSERT_STREQ(ENCODED(CharT, "aaaDEF111456"), str);
  ASSERT_EQ(writer.get_chars_written(), size_t{12});
}

TYPED_TEST(LlvmLibcPrintfWriterTest, NullStringWithZeroMaxLengthWithCallback,
           TestCharTypes) {
  using CharT = ParamType;
  CharT str[16];

  OutBuff<CharT> out_buff = {str, 0};

  Writer writer =
      make_writer(static_cast<CharT *>(nullptr), 0,
                  &overflow_write_flush_to_sink<CharT, copy_to_out>, &out_buff);
  writer.write(ENCODED(CharT, 'a'), 3);
  writer.write(ENCODED(CharT, "DEF"));
  writer.write(ENCODED(CharT, '1'), 3);
  writer.write(ENCODED(CharT, "456"));

  writer.get_write_buffer().template flush_to_sink<copy_to_out>(&out_buff);
  str[out_buff.cur_pos] = '\0';

  ASSERT_EQ(writer.get_chars_written(), size_t{12});
  ASSERT_STREQ(ENCODED(CharT, "aaaDEF111456"), str);
}

} // namespace
