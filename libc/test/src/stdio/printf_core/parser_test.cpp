//===-- Unittests for the printf Parser -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/arg_list.h"
#include "src/stdio/printf_core/parser.h"

#include <stdarg.h>

#include "test/UnitTest/PrintfMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::string_view;
using LIBC_NAMESPACE::internal::ArgList;

void init(const char *__restrict str, ...) {
  va_list vlist;
  va_start(vlist, str);
  ArgList v(vlist);
  va_end(vlist);

  LIBC_NAMESPACE::printf_core::Parser<ArgList> parser(str, v);
}

void evaluate(LIBC_NAMESPACE::printf_core::FormatSection *format_arr,
              const char *__restrict str, ...) {
  va_list vlist;
  va_start(vlist, str);
  ArgList v(vlist);
  va_end(vlist);

  LIBC_NAMESPACE::printf_core::Parser<ArgList> parser(str, v);

  for (auto cur_section = parser.get_next_section();
       !cur_section.raw_string.empty();
       cur_section = parser.get_next_section()) {
    *format_arr = cur_section;
    ++format_arr;
  }
}

TEST(LlvmLibcPrintfParserTest, Constructor) { init("test", 1, 2); }

TEST(LlvmLibcPrintfParserTest, EvalRaw) {
  LIBC_NAMESPACE::printf_core::FormatSection format_arr[10];
  const char *str = "test";
  evaluate(format_arr, str);

  LIBC_NAMESPACE::printf_core::FormatSection expected;
  expected.has_conv = false;

  expected.raw_string = {str, 4};

  ASSERT_PFORMAT_EQ(expected, format_arr[0]);
  // TODO: add checks that the format_arr after the last one has length 0
}

TEST(LlvmLibcPrintfParserTest, EvalSimple) {
  LIBC_NAMESPACE::printf_core::FormatSection format_arr[10];
  const char *str = "test %% test";
  evaluate(format_arr, str);

  LIBC_NAMESPACE::printf_core::FormatSection expected0, expected1, expected2;
  expected0.has_conv = false;

  expected0.raw_string = {str, 5};

  ASSERT_PFORMAT_EQ(expected0, format_arr[0]);

  expected1.has_conv = true;

  expected1.raw_string = {str + 5, 2};
  expected1.conv_name = '%';

  ASSERT_PFORMAT_EQ(expected1, format_arr[1]);

  expected2.has_conv = false;

  expected2.raw_string = {str + 7, 5};

  ASSERT_PFORMAT_EQ(expected2, format_arr[2]);
}

TEST(LlvmLibcPrintfParserTest, EvalOneArg) {
  LIBC_NAMESPACE::printf_core::FormatSection format_arr[10];
  const char *str = "%d";
  int arg1 = 12345;
  evaluate(format_arr, str, arg1);

  LIBC_NAMESPACE::printf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = {str, 2};
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  ASSERT_PFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcPrintfParserTest, EvalBadArg) {
  LIBC_NAMESPACE::printf_core::FormatSection format_arr[10];
  const char *str = "%\0abc";
  int arg1 = 12345;
  evaluate(format_arr, str, arg1);

  LIBC_NAMESPACE::printf_core::FormatSection expected;
  expected.has_conv = false;
  expected.raw_string = {str, 1};

  ASSERT_PFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcPrintfParserTest, EvalOneArgWithFlags) {
  LIBC_NAMESPACE::printf_core::FormatSection format_arr[10];
  const char *str = "%+-0 #d";
  int arg1 = 12345;
  evaluate(format_arr, str, arg1);

  LIBC_NAMESPACE::printf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = {str, 7};
  expected.flags = static_cast<LIBC_NAMESPACE::printf_core::FormatFlags>(
      LIBC_NAMESPACE::printf_core::FormatFlags::FORCE_SIGN |
      LIBC_NAMESPACE::printf_core::FormatFlags::LEFT_JUSTIFIED |
      LIBC_NAMESPACE::printf_core::FormatFlags::LEADING_ZEROES |
      LIBC_NAMESPACE::printf_core::FormatFlags::SPACE_PREFIX |
      LIBC_NAMESPACE::printf_core::FormatFlags::ALTERNATE_FORM);
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  ASSERT_PFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcPrintfParserTest, EvalOneArgWithWidth) {
  LIBC_NAMESPACE::printf_core::FormatSection format_arr[10];
  const char *str = "%12d";
  int arg1 = 12345;
  evaluate(format_arr, str, arg1);

  LIBC_NAMESPACE::printf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = {str, 4};
  expected.min_width = 12;
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  ASSERT_PFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcPrintfParserTest, EvalOneArgWithPrecision) {
  LIBC_NAMESPACE::printf_core::FormatSection format_arr[10];
  const char *str = "%.34d";
  int arg1 = 12345;
  evaluate(format_arr, str, arg1);

  LIBC_NAMESPACE::printf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = {str, 5};
  expected.precision = 34;
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  ASSERT_PFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcPrintfParserTest, EvalOneArgWithTrivialPrecision) {
  LIBC_NAMESPACE::printf_core::FormatSection format_arr[10];
  const char *str = "%.d";
  int arg1 = 12345;
  evaluate(format_arr, str, arg1);

  LIBC_NAMESPACE::printf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = {str, 3};
  expected.precision = 0;
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  ASSERT_PFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcPrintfParserTest, EvalOneArgWithShortLengthModifier) {
  LIBC_NAMESPACE::printf_core::FormatSection format_arr[10];
  const char *str = "%hd";
  int arg1 = 12345;
  evaluate(format_arr, str, arg1);

  LIBC_NAMESPACE::printf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = {str, 3};
  expected.length_modifier = LIBC_NAMESPACE::printf_core::LengthModifier::h;
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  ASSERT_PFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcPrintfParserTest, EvalOneArgWithLongLengthModifier) {
  LIBC_NAMESPACE::printf_core::FormatSection format_arr[10];
  const char *str = "%lld";
  long long arg1 = 12345;
  evaluate(format_arr, str, arg1);

  LIBC_NAMESPACE::printf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = {str, 4};
  expected.length_modifier = LIBC_NAMESPACE::printf_core::LengthModifier::ll;
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  ASSERT_PFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcPrintfParserTest, EvalOneArgWithBitWidthLengthModifier) {
  LIBC_NAMESPACE::printf_core::FormatSection format_arr[10];
  const char *str = "%w32d";
  long long arg1 = 12345;
  evaluate(format_arr, str, arg1);

  LIBC_NAMESPACE::printf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = {str, 5};
  expected.length_modifier = LIBC_NAMESPACE::printf_core::LengthModifier::w;
  expected.bit_width = 32;
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  ASSERT_PFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcPrintfParserTest, EvalOneArgWithFastBitWidthLengthModifier) {
  LIBC_NAMESPACE::printf_core::FormatSection format_arr[10];
  const char *str = "%wf32d";
  long long arg1 = 12345;
  evaluate(format_arr, str, arg1);

  LIBC_NAMESPACE::printf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = {str, 6};
  expected.length_modifier = LIBC_NAMESPACE::printf_core::LengthModifier::wf;
  expected.bit_width = 32;
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  ASSERT_PFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcPrintfParserTest, EvalOneArgWithAllOptions) {
  LIBC_NAMESPACE::printf_core::FormatSection format_arr[10];
  const char *str = "% -056.78jd";
  intmax_t arg1 = 12345;
  evaluate(format_arr, str, arg1);

  LIBC_NAMESPACE::printf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = {str, 11};
  expected.flags = static_cast<LIBC_NAMESPACE::printf_core::FormatFlags>(
      LIBC_NAMESPACE::printf_core::FormatFlags::LEFT_JUSTIFIED |
      LIBC_NAMESPACE::printf_core::FormatFlags::LEADING_ZEROES |
      LIBC_NAMESPACE::printf_core::FormatFlags::SPACE_PREFIX);
  expected.min_width = 56;
  expected.precision = 78;
  expected.length_modifier = LIBC_NAMESPACE::printf_core::LengthModifier::j;
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  ASSERT_PFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcPrintfParserTest, EvalThreeArgs) {
  LIBC_NAMESPACE::printf_core::FormatSection format_arr[10];
  const char *str = "%d%f%s";
  int arg1 = 12345;
  double arg2 = 123.45;
  const char *arg3 = "12345";
  evaluate(format_arr, str, arg1, arg2, arg3);

  LIBC_NAMESPACE::printf_core::FormatSection expected0, expected1, expected2;
  expected0.has_conv = true;

  expected0.raw_string = {str, 2};
  expected0.conv_val_raw = arg1;
  expected0.conv_name = 'd';

  ASSERT_PFORMAT_EQ(expected0, format_arr[0]);

  expected1.has_conv = true;

  expected1.raw_string = {str + 2, 2};
  expected1.conv_val_raw = LIBC_NAMESPACE::cpp::bit_cast<uint64_t>(arg2);
  expected1.conv_name = 'f';

  ASSERT_PFORMAT_EQ(expected1, format_arr[1]);

  expected2.has_conv = true;

  expected2.raw_string = {str + 4, 2};
  expected2.conv_val_ptr = const_cast<char *>(arg3);
  expected2.conv_name = 's';

  ASSERT_PFORMAT_EQ(expected2, format_arr[2]);
}

TEST(LlvmLibcPrintfParserTest, EvalOneArgWithOverflowingWidthAndPrecision) {
  LIBC_NAMESPACE::printf_core::FormatSection format_arr[10];
  const char *str = "%-999999999999.999999999999d";
  int arg1 = 12345;
  evaluate(format_arr, str, arg1);

  LIBC_NAMESPACE::printf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = {str, 28};
  expected.flags = LIBC_NAMESPACE::printf_core::FormatFlags::LEFT_JUSTIFIED;
  expected.min_width = INT_MAX;
  expected.precision = INT_MAX;
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  ASSERT_PFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcPrintfParserTest,
     EvalOneArgWithOverflowingWidthAndPrecisionAsArgs) {
  LIBC_NAMESPACE::printf_core::FormatSection format_arr[10];
  const char *str = "%*.*d";
  int arg1 = INT_MIN; // INT_MIN = -2147483648 if int is 32 bits.
  int arg2 = INT_MIN;
  int arg3 = 12345;
  evaluate(format_arr, str, arg1, arg2, arg3);

  LIBC_NAMESPACE::printf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = {str, 5};
  expected.flags = LIBC_NAMESPACE::printf_core::FormatFlags::LEFT_JUSTIFIED;
  expected.min_width = INT_MAX;
  expected.precision = arg2;
  expected.conv_val_raw = arg3;
  expected.conv_name = 'd';

  ASSERT_PFORMAT_EQ(expected, format_arr[0]);
}

#ifndef LIBC_COPT_PRINTF_DISABLE_INDEX_MODE

TEST(LlvmLibcPrintfParserTest, IndexModeOneArg) {
  LIBC_NAMESPACE::printf_core::FormatSection format_arr[10];
  const char *str = "%1$d";
  int arg1 = 12345;
  evaluate(format_arr, str, arg1);

  LIBC_NAMESPACE::printf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = {str, 4};
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  ASSERT_PFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcPrintfParserTest, IndexModeThreeArgsSequential) {
  LIBC_NAMESPACE::printf_core::FormatSection format_arr[10];
  const char *str = "%1$d%2$f%3$s";
  int arg1 = 12345;
  double arg2 = 123.45;
  const char *arg3 = "12345";
  evaluate(format_arr, str, arg1, arg2, arg3);

  LIBC_NAMESPACE::printf_core::FormatSection expected0, expected1, expected2;
  expected0.has_conv = true;

  expected0.raw_string = {str, 4};
  expected0.conv_val_raw = arg1;
  expected0.conv_name = 'd';

  ASSERT_PFORMAT_EQ(expected0, format_arr[0]);

  expected1.has_conv = true;

  expected1.raw_string = {str + 4, 4};
  expected1.conv_val_raw = LIBC_NAMESPACE::cpp::bit_cast<uint64_t>(arg2);
  expected1.conv_name = 'f';

  ASSERT_PFORMAT_EQ(expected1, format_arr[1]);

  expected2.has_conv = true;

  expected2.raw_string = {str + 8, 4};
  expected2.conv_val_ptr = const_cast<char *>(arg3);
  expected2.conv_name = 's';

  ASSERT_PFORMAT_EQ(expected2, format_arr[2]);
}

TEST(LlvmLibcPrintfParserTest, IndexModeThreeArgsReverse) {
  LIBC_NAMESPACE::printf_core::FormatSection format_arr[10];
  const char *str = "%3$d%2$f%1$s";
  int arg1 = 12345;
  double arg2 = 123.45;
  const char *arg3 = "12345";
  evaluate(format_arr, str, arg3, arg2, arg1);

  LIBC_NAMESPACE::printf_core::FormatSection expected0, expected1, expected2;
  expected0.has_conv = true;

  expected0.raw_string = {str, 4};
  expected0.conv_val_raw = arg1;
  expected0.conv_name = 'd';

  ASSERT_PFORMAT_EQ(expected0, format_arr[0]);

  expected1.has_conv = true;

  expected1.raw_string = {str + 4, 4};
  expected1.conv_val_raw = LIBC_NAMESPACE::cpp::bit_cast<uint64_t>(arg2);
  expected1.conv_name = 'f';

  ASSERT_PFORMAT_EQ(expected1, format_arr[1]);

  expected2.has_conv = true;

  expected2.raw_string = {str + 8, 4};
  expected2.conv_val_ptr = const_cast<char *>(arg3);
  expected2.conv_name = 's';

  ASSERT_PFORMAT_EQ(expected2, format_arr[2]);
}

TEST(LlvmLibcPrintfParserTest, IndexModeTenArgsRandom) {
  LIBC_NAMESPACE::printf_core::FormatSection format_arr[10];
  const char *str = "%6$d%3$d%7$d%2$d%8$d%1$d%4$d%9$d%5$d%10$d";
  int args[10] = {6, 4, 2, 7, 9, 1, 3, 5, 8, 10};
  evaluate(format_arr, str, args[0], args[1], args[2], args[3], args[4],
           args[5], args[6], args[7], args[8], args[9]);

  for (size_t i = 0; i < 10; ++i) {
    LIBC_NAMESPACE::printf_core::FormatSection expected;
    expected.has_conv = true;

    expected.raw_string = {str + (4 * i),
                           static_cast<size_t>(4 + (i >= 9 ? 1 : 0))};
    expected.conv_val_raw = i + 1;
    expected.conv_name = 'd';
    EXPECT_PFORMAT_EQ(expected, format_arr[i]);
  }
}

TEST(LlvmLibcPrintfParserTest, IndexModeComplexParsing) {
  LIBC_NAMESPACE::printf_core::FormatSection format_arr[10];
  const char *str = "normal text %3$llu %% %2$ *4$f %2$ .*4$f %1$1.1c";
  char arg1 = '1';
  double arg2 = 123.45;
  unsigned long long arg3 = 12345;
  int arg4 = 10;
  evaluate(format_arr, str, arg1, arg2, arg3, arg4);

  LIBC_NAMESPACE::printf_core::FormatSection expected0, expected1, expected2,
      expected3, expected4, expected5, expected6, expected7, expected8,
      expected9;

  expected0.has_conv = false;

  expected0.raw_string = {str, 12};

  EXPECT_PFORMAT_EQ(expected0, format_arr[0]);

  expected1.has_conv = true;

  expected1.raw_string = {str + 12, 6};
  expected1.length_modifier = LIBC_NAMESPACE::printf_core::LengthModifier::ll;
  expected1.conv_val_raw = arg3;
  expected1.conv_name = 'u';

  EXPECT_PFORMAT_EQ(expected1, format_arr[1]);

  expected2.has_conv = false;

  expected2.raw_string = {str + 18, 1};

  EXPECT_PFORMAT_EQ(expected2, format_arr[2]);

  expected3.has_conv = true;

  expected3.raw_string = {str + 19, 2};
  expected3.conv_name = '%';

  EXPECT_PFORMAT_EQ(expected3, format_arr[3]);

  expected4.has_conv = false;

  expected4.raw_string = {str + 21, 1};

  EXPECT_PFORMAT_EQ(expected4, format_arr[4]);

  expected5.has_conv = true;

  expected5.raw_string = {str + 22, 8};
  expected5.flags = LIBC_NAMESPACE::printf_core::FormatFlags::SPACE_PREFIX;
  expected5.min_width = arg4;
  expected5.conv_val_raw = LIBC_NAMESPACE::cpp::bit_cast<uint64_t>(arg2);
  expected5.conv_name = 'f';

  EXPECT_PFORMAT_EQ(expected5, format_arr[5]);

  expected6.has_conv = false;

  expected6.raw_string = {str + 30, 1};

  EXPECT_PFORMAT_EQ(expected6, format_arr[6]);

  expected7.has_conv = true;

  expected7.raw_string = {str + 31, 9};
  expected7.flags = LIBC_NAMESPACE::printf_core::FormatFlags::SPACE_PREFIX;
  expected7.precision = arg4;
  expected7.conv_val_raw = LIBC_NAMESPACE::cpp::bit_cast<uint64_t>(arg2);
  expected7.conv_name = 'f';

  EXPECT_PFORMAT_EQ(expected7, format_arr[7]);

  expected8.has_conv = false;

  expected8.raw_string = {str + 40, 1};

  EXPECT_PFORMAT_EQ(expected8, format_arr[8]);

  expected9.has_conv = true;

  expected9.raw_string = {str + 41, 7};
  expected9.min_width = 1;
  expected9.precision = 1;
  expected9.conv_val_raw = arg1;
  expected9.conv_name = 'c';

  EXPECT_PFORMAT_EQ(expected9, format_arr[9]);
}

TEST(LlvmLibcPrintfParserTest, IndexModeGapCheck) {
  LIBC_NAMESPACE::printf_core::FormatSection format_arr[10];
  const char *str = "%1$d%2$d%4$d";
  int arg1 = 1;
  int arg2 = 2;
  int arg3 = 3;
  int arg4 = 4;

  evaluate(format_arr, str, arg1, arg2, arg3, arg4);

  LIBC_NAMESPACE::printf_core::FormatSection expected0, expected1, expected2;

  expected0.has_conv = true;
  expected0.raw_string = {str, 4};
  expected0.conv_val_raw = arg1;
  expected0.conv_name = 'd';

  EXPECT_PFORMAT_EQ(expected0, format_arr[0]);

  expected1.has_conv = true;
  expected1.raw_string = {str + 4, 4};
  expected1.conv_val_raw = arg2;
  expected1.conv_name = 'd';

  EXPECT_PFORMAT_EQ(expected1, format_arr[1]);

  expected2.has_conv = false;
  expected2.raw_string = {str + 8, 4};

  EXPECT_PFORMAT_EQ(expected2, format_arr[2]);
}

TEST(LlvmLibcPrintfParserTest, IndexModeTrailingPercentCrash) {
  LIBC_NAMESPACE::printf_core::FormatSection format_arr[10];
  const char *str = "%2$d%";
  evaluate(format_arr, str, 1, 2);

  LIBC_NAMESPACE::printf_core::FormatSection expected0, expected1;
  expected0.has_conv = false;

  expected0.raw_string = {str, 4};
  EXPECT_PFORMAT_EQ(expected0, format_arr[0]);

  expected1.has_conv = false;

  expected1.raw_string = {str + 4, 1};
  EXPECT_PFORMAT_EQ(expected1, format_arr[1]);
}

TEST(LlvmLibcPrintfParserTest, DoublePercentIsAllowedInvalidIndex) {
  LIBC_NAMESPACE::printf_core::FormatSection format_arr[10];

  // Normally this conversion specifier would be raw (due to having a width
  // defined as an invalid argument) but since it's a % conversion it's allowed
  // by this specific parser. Any % conversion that is not just "%%" is
  // undefined, so this is implementation-specific behavior.
  // The goal is to be consistent. A conversion specifier of "%L%" is also
  // undefined, but is traditionally displayed as just "%". "%2%" is also
  // displayed as "%", even though if the width was respected it should be " %".
  // Finally, "%*%" traditionally is displayed as "%" but also traditionally
  // consumes an argument, since the * consumes an integer. Therefore, having
  // "%*2$%" display as "%" is consistent with other printf behavior.
  const char *str = "%*2$%";

  evaluate(format_arr, str, 1, 2);

  LIBC_NAMESPACE::printf_core::FormatSection expected0;
  expected0.has_conv = true;

  expected0.raw_string = str;
  expected0.conv_name = '%';
  EXPECT_PFORMAT_EQ(expected0, format_arr[0]);
}

#endif // LIBC_COPT_PRINTF_DISABLE_INDEX_MODE
