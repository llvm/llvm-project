//===-- Unittests for the printf Parser -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/Bit.h"
#include "src/__support/arg_list.h"
#include "src/stdio/printf_core/parser.h"

#include <stdarg.h>

#include "utils/UnitTest/Test.h"

class LlvmLibcPrintfParserTest : public __llvm_libc::testing::Test {
public:
  void assert_eq_fs(__llvm_libc::printf_core::FormatSection expected,
                    __llvm_libc::printf_core::FormatSection actual) {
    ASSERT_EQ(expected.has_conv, actual.has_conv);
    ASSERT_EQ(expected.raw_len, actual.raw_len);

    for (size_t i = 0; i < expected.raw_len; ++i) {
      EXPECT_EQ(expected.raw_string[i], actual.raw_string[i]);
    }

    if (expected.has_conv) {
      ASSERT_EQ(static_cast<uint8_t>(expected.flags),
                static_cast<uint8_t>(actual.flags));
      ASSERT_EQ(expected.min_width, actual.min_width);
      ASSERT_EQ(expected.precision, actual.precision);
      ASSERT_TRUE(expected.length_modifier == actual.length_modifier);
      ASSERT_EQ(expected.conv_name, actual.conv_name);

      if (expected.conv_name == 'p' || expected.conv_name == 'n' ||
          expected.conv_name == 's') {
        ASSERT_EQ(expected.conv_val_ptr, actual.conv_val_ptr);
      } else if (expected.conv_name != '%') {
        ASSERT_EQ(expected.conv_val_raw, actual.conv_val_raw);
      }
    }
  }
};

void init(const char *__restrict str, ...) {
  va_list vlist;
  va_start(vlist, str);
  __llvm_libc::internal::ArgList v(vlist);
  va_end(vlist);

  __llvm_libc::printf_core::Parser parser(str, v);
}

void evaluate(__llvm_libc::printf_core::FormatSection *format_arr,
              const char *__restrict str, ...) {
  va_list vlist;
  va_start(vlist, str);
  __llvm_libc::internal::ArgList v(vlist);
  va_end(vlist);

  __llvm_libc::printf_core::Parser parser(str, v);

  for (auto cur_section = parser.get_next_section(); cur_section.raw_len > 0;
       cur_section = parser.get_next_section()) {
    *format_arr = cur_section;
    ++format_arr;
  }
}

TEST_F(LlvmLibcPrintfParserTest, Constructor) { init("test", 1, 2); }

TEST_F(LlvmLibcPrintfParserTest, EvalRaw) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "test";
  evaluate(format_arr, str);

  __llvm_libc::printf_core::FormatSection expected;
  expected.has_conv = false;
  expected.raw_len = 4;
  expected.raw_string = str;

  assert_eq_fs(expected, format_arr[0]);
}

TEST_F(LlvmLibcPrintfParserTest, EvalSimple) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "test %% test";
  evaluate(format_arr, str);

  __llvm_libc::printf_core::FormatSection expected0, expected1, expected2;
  expected0.has_conv = false;
  expected0.raw_len = 5;
  expected0.raw_string = str;

  assert_eq_fs(expected0, format_arr[0]);

  expected1.has_conv = true;
  expected1.raw_len = 2;
  expected1.raw_string = str + 5;
  expected1.conv_name = '%';

  assert_eq_fs(expected1, format_arr[1]);

  expected2.has_conv = false;
  expected2.raw_len = 5;
  expected2.raw_string = str + 7;

  assert_eq_fs(expected2, format_arr[2]);
}

TEST_F(LlvmLibcPrintfParserTest, EvalOneArg) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "%d";
  int arg1 = 12345;
  evaluate(format_arr, str, arg1);

  __llvm_libc::printf_core::FormatSection expected;
  expected.has_conv = true;
  expected.raw_len = 2;
  expected.raw_string = str;
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  assert_eq_fs(expected, format_arr[0]);
}

TEST_F(LlvmLibcPrintfParserTest, EvalOneArgWithFlags) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "%+-0 #d";
  int arg1 = 12345;
  evaluate(format_arr, str, arg1);

  __llvm_libc::printf_core::FormatSection expected;
  expected.has_conv = true;
  expected.raw_len = 7;
  expected.raw_string = str;
  expected.flags = static_cast<__llvm_libc::printf_core::FormatFlags>(
      __llvm_libc::printf_core::FormatFlags::FORCE_SIGN |
      __llvm_libc::printf_core::FormatFlags::LEFT_JUSTIFIED |
      __llvm_libc::printf_core::FormatFlags::LEADING_ZEROES |
      __llvm_libc::printf_core::FormatFlags::SPACE_PREFIX |
      __llvm_libc::printf_core::FormatFlags::ALTERNATE_FORM);
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  assert_eq_fs(expected, format_arr[0]);
}

TEST_F(LlvmLibcPrintfParserTest, EvalOneArgWithWidth) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "%12d";
  int arg1 = 12345;
  evaluate(format_arr, str, arg1);

  __llvm_libc::printf_core::FormatSection expected;
  expected.has_conv = true;
  expected.raw_len = 4;
  expected.raw_string = str;
  expected.min_width = 12;
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  assert_eq_fs(expected, format_arr[0]);
}

TEST_F(LlvmLibcPrintfParserTest, EvalOneArgWithPrecision) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "%.34d";
  int arg1 = 12345;
  evaluate(format_arr, str, arg1);

  __llvm_libc::printf_core::FormatSection expected;
  expected.has_conv = true;
  expected.raw_len = 5;
  expected.raw_string = str;
  expected.precision = 34;
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  assert_eq_fs(expected, format_arr[0]);
}

TEST_F(LlvmLibcPrintfParserTest, EvalOneArgWithTrivialPrecision) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "%.d";
  int arg1 = 12345;
  evaluate(format_arr, str, arg1);

  __llvm_libc::printf_core::FormatSection expected;
  expected.has_conv = true;
  expected.raw_len = 3;
  expected.raw_string = str;
  expected.precision = 0;
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  assert_eq_fs(expected, format_arr[0]);
}

TEST_F(LlvmLibcPrintfParserTest, EvalOneArgWithShortLengthModifier) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "%hd";
  int arg1 = 12345;
  evaluate(format_arr, str, arg1);

  __llvm_libc::printf_core::FormatSection expected;
  expected.has_conv = true;
  expected.raw_len = 3;
  expected.raw_string = str;
  expected.length_modifier = __llvm_libc::printf_core::LengthModifier::h;
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  assert_eq_fs(expected, format_arr[0]);
}

TEST_F(LlvmLibcPrintfParserTest, EvalOneArgWithLongLengthModifier) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "%lld";
  int arg1 = 12345;
  evaluate(format_arr, str, arg1);

  __llvm_libc::printf_core::FormatSection expected;
  expected.has_conv = true;
  expected.raw_len = 4;
  expected.raw_string = str;
  expected.length_modifier = __llvm_libc::printf_core::LengthModifier::ll;
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  assert_eq_fs(expected, format_arr[0]);
}

TEST_F(LlvmLibcPrintfParserTest, EvalOneArgWithAllOptions) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "% -056.78jd";
  int arg1 = 12345;
  evaluate(format_arr, str, arg1);

  __llvm_libc::printf_core::FormatSection expected;
  expected.has_conv = true;
  expected.raw_len = 11;
  expected.raw_string = str;
  expected.flags = static_cast<__llvm_libc::printf_core::FormatFlags>(
      __llvm_libc::printf_core::FormatFlags::LEFT_JUSTIFIED |
      __llvm_libc::printf_core::FormatFlags::LEADING_ZEROES |
      __llvm_libc::printf_core::FormatFlags::SPACE_PREFIX);
  expected.min_width = 56;
  expected.precision = 78;
  expected.length_modifier = __llvm_libc::printf_core::LengthModifier::j;
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  assert_eq_fs(expected, format_arr[0]);
}

TEST_F(LlvmLibcPrintfParserTest, EvalThreeArgs) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "%d%f%s";
  int arg1 = 12345;
  double arg2 = 123.45;
  const char *arg3 = "12345";
  evaluate(format_arr, str, arg1, arg2, arg3);

  __llvm_libc::printf_core::FormatSection expected0, expected1, expected2;
  expected0.has_conv = true;
  expected0.raw_len = 2;
  expected0.raw_string = str;
  expected0.conv_val_raw = arg1;
  expected0.conv_name = 'd';

  assert_eq_fs(expected0, format_arr[0]);

  expected1.has_conv = true;
  expected1.raw_len = 2;
  expected1.raw_string = str + 2;
  expected1.conv_val_raw = __llvm_libc::bit_cast<uint64_t>(arg2);
  expected1.conv_name = 'f';

  assert_eq_fs(expected1, format_arr[1]);

  expected2.has_conv = true;
  expected2.raw_len = 2;
  expected2.raw_string = str + 4;
  expected2.conv_val_ptr = const_cast<char *>(arg3);
  expected2.conv_name = 's';

  assert_eq_fs(expected2, format_arr[2]);
}
