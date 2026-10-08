//===-- Unittests for the printf Converter --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/printf_core/converter.h"
#include "src/__support/printf_core/core_structs.h"
#include "src/__support/printf_core/writer.h"

#include "test/UnitTest/CharLiteralUtils.h"
#include "test/UnitTest/Test.h"

namespace {

using LIBC_NAMESPACE::printf_core::FormatSection;
using LIBC_NAMESPACE::printf_core::make_drop_overflow_writer;
using LIBC_NAMESPACE::printf_core::Mode;
using LIBC_NAMESPACE::printf_core::OverflowMode;
using LIBC_NAMESPACE::printf_core::WriteBuffer;
using LIBC_NAMESPACE::printf_core::Writer;

#if defined(LIBC_TYPES_WCHAR_T_IS_UTF32)
using TestCharTypes = LIBC_NAMESPACE::testing::TypeList<char, wchar_t>;
#else
using TestCharTypes = LIBC_NAMESPACE::testing::TypeList<char>;
#endif

TYPED_TEST(LlvmLibcPrintfConverterTest, SimpleRawConversion, TestCharTypes) {
  using CharT = ParamType;
  CharT str[60];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();

  FormatSection<CharT> raw_section;
  raw_section.has_conv = false;
  raw_section.raw_string = ENCODED(CharT, "abc");

  LIBC_NAMESPACE::printf_core::convert(&writer, raw_section);

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(str, ENCODED(CharT, "abc"));
  ASSERT_EQ(writer.get_chars_written(), size_t{3});
}

TYPED_TEST(LlvmLibcPrintfConverterTest, PercentConversion, TestCharTypes) {
  using CharT = ParamType;
  CharT str[60];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();

  FormatSection<CharT> simple_conv;
  simple_conv.has_conv = true;
  simple_conv.raw_string = ENCODED(CharT, "%%");
  simple_conv.conv_name = ENCODED(CharT, '%');

  LIBC_NAMESPACE::printf_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(str, ENCODED(CharT, "%"));
  ASSERT_EQ(writer.get_chars_written(), size_t{1});
}

TYPED_TEST(LlvmLibcPrintfConverterTest, CharConversionSimple, TestCharTypes) {
  using CharT = ParamType;
  CharT str[60];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  FormatSection<CharT> simple_conv;
  simple_conv.has_conv = true;
  // If has_conv is true, the raw string is ignored. They are not being parsed
  // and match the actual conversion taking place so that you can compare these
  // tests with other implmentations. The raw strings are completely optional.
  simple_conv.raw_string = ENCODED(CharT, "%c");
  simple_conv.conv_name = ENCODED(CharT, 'c');
  simple_conv.conv_val_raw = 'D';

  LIBC_NAMESPACE::printf_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(str, ENCODED(CharT, "D"));
  ASSERT_EQ(writer.get_chars_written(), size_t{1});
}

TYPED_TEST(LlvmLibcPrintfConverterTest, CharConversionRightJustified,
           TestCharTypes) {
  using CharT = ParamType;
  CharT str[60];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  FormatSection<CharT> right_justified_conv;
  right_justified_conv.has_conv = true;
  right_justified_conv.raw_string = ENCODED(CharT, "%4c");
  right_justified_conv.conv_name = ENCODED(CharT, 'c');
  right_justified_conv.min_width = 4;
  right_justified_conv.conv_val_raw = 'E';
  LIBC_NAMESPACE::printf_core::convert(&writer, right_justified_conv);

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(str, ENCODED(CharT, "   E"));
  ASSERT_EQ(writer.get_chars_written(), size_t{4});
}

TYPED_TEST(LlvmLibcPrintfConverterTest, CharConversionLeftJustified,
           TestCharTypes) {
  using CharT = ParamType;
  CharT str[60];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  FormatSection<CharT> left_justified_conv;
  left_justified_conv.has_conv = true;
  left_justified_conv.raw_string = ENCODED(CharT, "%-4c");
  left_justified_conv.conv_name = ENCODED(CharT, 'c');
  left_justified_conv.flags =
      LIBC_NAMESPACE::printf_core::FormatFlags::LEFT_JUSTIFIED;
  left_justified_conv.min_width = 4;
  left_justified_conv.conv_val_raw = 'F';
  LIBC_NAMESPACE::printf_core::convert(&writer, left_justified_conv);

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(str, ENCODED(CharT, "F   "));
  ASSERT_EQ(writer.get_chars_written(), size_t{4});
}

#if !defined(LIBC_COPT_PRINTF_DISABLE_WIDE)

TYPED_TEST(LlvmLibcPrintfConverterTest, WideCharConversionSimple,
           TestCharTypes) {
  using CharT = ParamType;
  CharT str[60];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  FormatSection<CharT> simple_conv;
  simple_conv.has_conv = true;
  // If has_conv is true, the raw string is ignored. They are not being parsed
  // and match the actual conversion taking place so that you can compare these
  // tests with other implmentations. The raw strings are completely optional.
  simple_conv.raw_string = ENCODED(CharT, "%lc");
  simple_conv.length_modifier = LIBC_NAMESPACE::printf_core::LengthModifier::l;
  simple_conv.conv_name = ENCODED(CharT, 'c');
  simple_conv.conv_val_raw = L'D';

  LIBC_NAMESPACE::printf_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(str, ENCODED(CharT, "D"));
  ASSERT_EQ(writer.get_chars_written(), size_t{1});
}

TYPED_TEST(LlvmLibcPrintfConverterTest, WideCharConversionRightJustified,
           TestCharTypes) {
  using CharT = ParamType;
  CharT str[60];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  FormatSection<CharT> right_justified_conv;
  right_justified_conv.has_conv = true;
  right_justified_conv.raw_string = ENCODED(CharT, "%4lc");
  right_justified_conv.length_modifier =
      LIBC_NAMESPACE::printf_core::LengthModifier::l;
  right_justified_conv.conv_name = ENCODED(CharT, 'c');
  right_justified_conv.min_width = 4;
  right_justified_conv.conv_val_raw = L'E';
  LIBC_NAMESPACE::printf_core::convert(&writer, right_justified_conv);

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(str, ENCODED(CharT, "   E"));
  ASSERT_EQ(writer.get_chars_written(), size_t{4});
}

TYPED_TEST(LlvmLibcPrintfConverterTest, WideCharConversionLeftJustified,
           TestCharTypes) {
  using CharT = ParamType;
  CharT str[60];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  FormatSection<CharT> left_justified_conv;
  left_justified_conv.has_conv = true;
  left_justified_conv.raw_string = ENCODED(CharT, "%-4lc");
  left_justified_conv.length_modifier =
      LIBC_NAMESPACE::printf_core::LengthModifier::l;
  left_justified_conv.conv_name = ENCODED(CharT, 'c');
  left_justified_conv.flags =
      LIBC_NAMESPACE::printf_core::FormatFlags::LEFT_JUSTIFIED;
  left_justified_conv.min_width = 4;
  left_justified_conv.conv_val_raw = L'F';
  LIBC_NAMESPACE::printf_core::convert(&writer, left_justified_conv);

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(str, ENCODED(CharT, "F   "));
  ASSERT_EQ(writer.get_chars_written(), size_t{4});
}

#endif // !LIBC_COPT_PRINTF_DISABLE_WIDE

TYPED_TEST(LlvmLibcPrintfConverterTest, StringConversionSimple, TestCharTypes) {
  using CharT = ParamType;
  CharT str[60];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();

  FormatSection<CharT> simple_conv;
  simple_conv.has_conv = true;
  simple_conv.raw_string = ENCODED(CharT, "%s");
  simple_conv.conv_name = ENCODED(CharT, 's');
  simple_conv.conv_val_ptr = const_cast<char *>("DEF");

  LIBC_NAMESPACE::printf_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(str, ENCODED(CharT, "DEF"));
  ASSERT_EQ(writer.get_chars_written(), size_t{3});
}

TYPED_TEST(LlvmLibcPrintfConverterTest, StringConversionPrecisionHigh,
           TestCharTypes) {
  using CharT = ParamType;
  CharT str[60];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  FormatSection<CharT> high_precision_conv;
  high_precision_conv.has_conv = true;
  high_precision_conv.raw_string = ENCODED(CharT, "%.4s");
  high_precision_conv.conv_name = ENCODED(CharT, 's');
  high_precision_conv.precision = 4;
  high_precision_conv.conv_val_ptr = const_cast<char *>("456");
  LIBC_NAMESPACE::printf_core::convert(&writer, high_precision_conv);

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(str, ENCODED(CharT, "456"));
  ASSERT_EQ(writer.get_chars_written(), size_t{3});
}

TYPED_TEST(LlvmLibcPrintfConverterTest, StringConversionPrecisionLow,
           TestCharTypes) {
  using CharT = ParamType;
  CharT str[60];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  FormatSection<CharT> low_precision_conv;
  low_precision_conv.has_conv = true;
  low_precision_conv.raw_string = ENCODED(CharT, "%.2s");
  low_precision_conv.conv_name = ENCODED(CharT, 's');
  low_precision_conv.precision = 2;
  low_precision_conv.conv_val_ptr = const_cast<char *>("xyz");
  LIBC_NAMESPACE::printf_core::convert(&writer, low_precision_conv);

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(str, ENCODED(CharT, "xy"));
  ASSERT_EQ(writer.get_chars_written(), size_t{2});
}

TYPED_TEST(LlvmLibcPrintfConverterTest, StringConversionRightJustified,
           TestCharTypes) {
  using CharT = ParamType;
  CharT str[60];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  FormatSection<CharT> right_justified_conv;
  right_justified_conv.has_conv = true;
  right_justified_conv.raw_string = ENCODED(CharT, "%4s");
  right_justified_conv.conv_name = ENCODED(CharT, 's');
  right_justified_conv.min_width = 4;
  right_justified_conv.conv_val_ptr = const_cast<char *>("789");
  LIBC_NAMESPACE::printf_core::convert(&writer, right_justified_conv);

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(str, ENCODED(CharT, " 789"));
  ASSERT_EQ(writer.get_chars_written(), size_t{4});
}

TYPED_TEST(LlvmLibcPrintfConverterTest, StringConversionLeftJustified,
           TestCharTypes) {
  using CharT = ParamType;
  CharT str[60];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  FormatSection<CharT> left_justified_conv;
  left_justified_conv.has_conv = true;
  left_justified_conv.raw_string = ENCODED(CharT, "%-4s");
  left_justified_conv.conv_name = ENCODED(CharT, 's');
  left_justified_conv.flags =
      LIBC_NAMESPACE::printf_core::FormatFlags::LEFT_JUSTIFIED;
  left_justified_conv.min_width = 4;
  left_justified_conv.conv_val_ptr = const_cast<char *>("ghi");
  LIBC_NAMESPACE::printf_core::convert(&writer, left_justified_conv);

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(str, ENCODED(CharT, "ghi "));
  ASSERT_EQ(writer.get_chars_written(), size_t{4});
}

#if !defined(LIBC_COPT_PRINTF_DISABLE_WIDE)

TYPED_TEST(LlvmLibcPrintfConverterTest, WideStringConversionSimple,
           TestCharTypes) {
  using CharT = ParamType;
  CharT str[60];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();

  FormatSection<CharT> simple_conv;
  simple_conv.has_conv = true;
  simple_conv.raw_string = ENCODED(CharT, "%ls");
  simple_conv.length_modifier = LIBC_NAMESPACE::printf_core::LengthModifier::l;
  simple_conv.conv_name = ENCODED(CharT, 's');
  simple_conv.conv_val_ptr = const_cast<wchar_t *>(L"DEF");

  LIBC_NAMESPACE::printf_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(str, ENCODED(CharT, "DEF"));
  ASSERT_EQ(writer.get_chars_written(), size_t{3});
}

TYPED_TEST(LlvmLibcPrintfConverterTest, WideStringConversionPrecisionHigh,
           TestCharTypes) {
  using CharT = ParamType;
  CharT str[60];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  FormatSection<CharT> high_precision_conv;
  high_precision_conv.has_conv = true;
  high_precision_conv.raw_string = ENCODED(CharT, "%.4ls");
  high_precision_conv.length_modifier =
      LIBC_NAMESPACE::printf_core::LengthModifier::l;
  high_precision_conv.conv_name = ENCODED(CharT, 's');
  high_precision_conv.precision = 4;
  high_precision_conv.conv_val_ptr = const_cast<wchar_t *>(L"456");
  LIBC_NAMESPACE::printf_core::convert(&writer, high_precision_conv);

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(str, ENCODED(CharT, "456"));
  ASSERT_EQ(writer.get_chars_written(), size_t{3});
}

TYPED_TEST(LlvmLibcPrintfConverterTest, WideStringConversionPrecisionLow,
           TestCharTypes) {
  using CharT = ParamType;
  CharT str[60];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  FormatSection<CharT> low_precision_conv;
  low_precision_conv.has_conv = true;
  low_precision_conv.raw_string = ENCODED(CharT, "%.2ls");
  low_precision_conv.length_modifier =
      LIBC_NAMESPACE::printf_core::LengthModifier::l;
  low_precision_conv.conv_name = ENCODED(CharT, 's');
  low_precision_conv.precision = 2;
  low_precision_conv.conv_val_ptr = const_cast<wchar_t *>(L"xyz");
  LIBC_NAMESPACE::printf_core::convert(&writer, low_precision_conv);

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(str, ENCODED(CharT, "xy"));
  ASSERT_EQ(writer.get_chars_written(), size_t{2});
}

TYPED_TEST(LlvmLibcPrintfConverterTest, WideStringConversionRightJustified,
           TestCharTypes) {
  using CharT = ParamType;
  CharT str[60];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  FormatSection<CharT> right_justified_conv;
  right_justified_conv.has_conv = true;
  right_justified_conv.raw_string = ENCODED(CharT, "%4ls");
  right_justified_conv.length_modifier =
      LIBC_NAMESPACE::printf_core::LengthModifier::l;
  right_justified_conv.conv_name = ENCODED(CharT, 's');
  right_justified_conv.min_width = 4;
  right_justified_conv.conv_val_ptr = const_cast<wchar_t *>(L"789");
  LIBC_NAMESPACE::printf_core::convert(&writer, right_justified_conv);

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(str, ENCODED(CharT, " 789"));
  ASSERT_EQ(writer.get_chars_written(), size_t{4});
}

TYPED_TEST(LlvmLibcPrintfConverterTest, WideStringConversionLeftJustified,
           TestCharTypes) {
  using CharT = ParamType;
  CharT str[60];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  FormatSection<CharT> left_justified_conv;
  left_justified_conv.has_conv = true;
  left_justified_conv.raw_string = ENCODED(CharT, "%-4ls");
  left_justified_conv.length_modifier =
      LIBC_NAMESPACE::printf_core::LengthModifier::l;
  left_justified_conv.conv_name = ENCODED(CharT, 's');
  left_justified_conv.flags =
      LIBC_NAMESPACE::printf_core::FormatFlags::LEFT_JUSTIFIED;
  left_justified_conv.min_width = 4;
  left_justified_conv.conv_val_ptr = const_cast<wchar_t *>(L"ghi");
  LIBC_NAMESPACE::printf_core::convert(&writer, left_justified_conv);

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(str, ENCODED(CharT, "ghi "));
  ASSERT_EQ(writer.get_chars_written(), size_t{4});
}

#endif // !LIBC_COPT_PRINTF_DISABLE_WIDE

TYPED_TEST(LlvmLibcPrintfConverterTest, IntConversionSimple, TestCharTypes) {
  using CharT = ParamType;
  CharT str[60];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  FormatSection<CharT> section;
  section.has_conv = true;
  section.raw_string = ENCODED(CharT, "%d");
  section.conv_name = ENCODED(CharT, 'd');
  section.conv_val_raw = 12345;
  LIBC_NAMESPACE::printf_core::convert(&writer, section);

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(str, ENCODED(CharT, "12345"));
  ASSERT_EQ(writer.get_chars_written(), size_t{5});
}

TYPED_TEST(LlvmLibcPrintfConverterTest, HexConversion, TestCharTypes) {
  using CharT = ParamType;
  CharT str[60];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  FormatSection<CharT> section;
  section.has_conv = true;
  section.raw_string = ENCODED(CharT, "%#018x");
  section.conv_name = ENCODED(CharT, 'x');
  section.flags = static_cast<LIBC_NAMESPACE::printf_core::FormatFlags>(
      LIBC_NAMESPACE::printf_core::FormatFlags::ALTERNATE_FORM |
      LIBC_NAMESPACE::printf_core::FormatFlags::LEADING_ZEROES);
  section.min_width = 18;
  section.conv_val_raw = 0x123456ab;
  LIBC_NAMESPACE::printf_core::convert(&writer, section);

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');
  ASSERT_STREQ(str, ENCODED(CharT, "0x00000000123456ab"));
  ASSERT_EQ(writer.get_chars_written(), size_t{18});
}

TYPED_TEST(LlvmLibcPrintfConverterTest, BinaryConversion, TestCharTypes) {
  using CharT = ParamType;
  CharT str[60];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();
  FormatSection<CharT> section;
  section.has_conv = true;
  section.raw_string = ENCODED(CharT, "%b");
  section.conv_name = ENCODED(CharT, 'b');
  section.conv_val_raw = 42;
  LIBC_NAMESPACE::printf_core::convert(&writer, section);

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');

  ASSERT_STREQ(str, ENCODED(CharT, "101010"));
  ASSERT_EQ(writer.get_chars_written(), size_t{6});
}

TYPED_TEST(LlvmLibcPrintfConverterTest, PointerConversion, TestCharTypes) {
  using CharT = ParamType;
  CharT str[60];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();

  FormatSection<CharT> section;
  section.has_conv = true;
  section.raw_string = ENCODED(CharT, "%p");
  section.conv_name = ENCODED(CharT, 'p');
  section.conv_val_ptr = (void *)(0x123456ab);
  LIBC_NAMESPACE::printf_core::convert(&writer, section);

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');
  ASSERT_STREQ(str, ENCODED(CharT, "0x123456ab"));
  ASSERT_EQ(writer.get_chars_written(), size_t{10});
}

TYPED_TEST(LlvmLibcPrintfConverterTest, OctConversion, TestCharTypes) {
  using CharT = ParamType;
  CharT str[60];
  Writer writer = make_drop_overflow_writer(str, sizeof(str) - 1);
  WriteBuffer<CharT> &wb = writer.get_write_buffer();

  FormatSection<CharT> section;
  section.has_conv = true;
  section.raw_string = ENCODED(CharT, "%o");
  section.conv_name = ENCODED(CharT, 'o');
  section.conv_val_raw = 01234;
  LIBC_NAMESPACE::printf_core::convert(&writer, section);

  wb.buff[wb.buff_cur] = ENCODED(CharT, '\0');
  ASSERT_STREQ(str, ENCODED(CharT, "1234"));
  ASSERT_EQ(writer.get_chars_written(), size_t{4});
}

} // namespace
