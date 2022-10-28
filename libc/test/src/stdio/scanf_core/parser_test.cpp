//===-- Unittests for the scanf Parser -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/bitset.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/arg_list.h"
#include "src/stdio/scanf_core/parser.h"

#include <stdarg.h>

#include "utils/UnitTest/ScanfMatcher.h"
#include "utils/UnitTest/Test.h"

using __llvm_libc::cpp::string_view;

void init(const char *__restrict str, ...) {
  va_list vlist;
  va_start(vlist, str);
  __llvm_libc::internal::ArgList v(vlist);
  va_end(vlist);

  __llvm_libc::scanf_core::Parser parser(str, v);
}

void evaluate(__llvm_libc::scanf_core::FormatSection *format_arr,
              const char *__restrict str, ...) {
  va_list vlist;
  va_start(vlist, str);
  __llvm_libc::internal::ArgList v(vlist);
  va_end(vlist);

  __llvm_libc::scanf_core::Parser parser(str, v);

  for (auto cur_section = parser.get_next_section();
       !cur_section.raw_string.empty();
       cur_section = parser.get_next_section()) {
    *format_arr = cur_section;
    ++format_arr;
  }
}

TEST(LlvmLibcScanfParserTest, Constructor) { init("test", 1, 2); }

TEST(LlvmLibcScanfParserTest, EvalRaw) {
  __llvm_libc::scanf_core::FormatSection format_arr[10];
  const char *str = "test";
  evaluate(format_arr, str);

  __llvm_libc::scanf_core::FormatSection expected;
  expected.has_conv = false;

  expected.raw_string = str;

  ASSERT_SFORMAT_EQ(expected, format_arr[0]);
  // TODO: add checks that the format_arr after the last one has length 0
}

TEST(LlvmLibcScanfParserTest, EvalSimple) {
  __llvm_libc::scanf_core::FormatSection format_arr[10];
  const char *str = "test %% test";
  evaluate(format_arr, str);

  __llvm_libc::scanf_core::FormatSection expected0, expected1, expected2;
  expected0.has_conv = false;

  expected0.raw_string = {str, 5};

  ASSERT_SFORMAT_EQ(expected0, format_arr[0]);

  expected1.has_conv = true;

  expected1.raw_string = {str + 5, 2};
  expected1.conv_name = '%';

  ASSERT_SFORMAT_EQ(expected1, format_arr[1]);

  expected2.has_conv = false;

  expected2.raw_string = {str + 7, 5};

  ASSERT_SFORMAT_EQ(expected2, format_arr[2]);
}

TEST(LlvmLibcScanfParserTest, EvalOneArg) {
  __llvm_libc::scanf_core::FormatSection format_arr[10];
  const char *str = "%d";
  int arg1 = 12345;
  evaluate(format_arr, str, &arg1);

  __llvm_libc::scanf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = str;
  expected.output_ptr = &arg1;
  expected.conv_name = 'd';

  ASSERT_SFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcScanfParserTest, EvalOneArgWithFlag) {
  __llvm_libc::scanf_core::FormatSection format_arr[10];
  const char *str = "%*d";
  // Since NO_WRITE is set, the argument shouldn't be used, but I've included
  // one anyways because in the case that it doesn't work it's better for it to
  // have a real argument to check against.
  int arg1 = 12345;
  evaluate(format_arr, str, &arg1);

  __llvm_libc::scanf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = str;
  expected.flags = __llvm_libc::scanf_core::FormatFlags::NO_WRITE;
  expected.conv_name = 'd';

  ASSERT_SFORMAT_EQ(expected, format_arr[0]);

  // If NO_WRITE is set, then the equality check ignores the pointer since it's
  // irrelevant, but in this case I want to make sure that it hasn't been set
  // and check it separately.
  ASSERT_EQ(expected.output_ptr, format_arr[0].output_ptr);
}

TEST(LlvmLibcScanfParserTest, EvalOneArgWithWidth) {
  __llvm_libc::scanf_core::FormatSection format_arr[10];
  const char *str = "%12d";
  int arg1 = 12345;
  evaluate(format_arr, str, &arg1);

  __llvm_libc::scanf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = str;
  expected.max_width = 12;
  expected.output_ptr = &arg1;
  expected.conv_name = 'd';

  ASSERT_SFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcScanfParserTest, EvalOneArgWithShortLengthModifier) {
  __llvm_libc::scanf_core::FormatSection format_arr[10];
  const char *str = "%hd";
  int arg1 = 12345;
  evaluate(format_arr, str, &arg1);

  __llvm_libc::scanf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = str;
  expected.length_modifier = __llvm_libc::scanf_core::LengthModifier::h;
  expected.output_ptr = &arg1;
  expected.conv_name = 'd';

  ASSERT_SFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcScanfParserTest, EvalOneArgWithLongLengthModifier) {
  __llvm_libc::scanf_core::FormatSection format_arr[10];
  const char *str = "%lld";
  long long arg1 = 12345;
  evaluate(format_arr, str, &arg1);

  __llvm_libc::scanf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = str;
  expected.length_modifier = __llvm_libc::scanf_core::LengthModifier::ll;
  expected.output_ptr = &arg1;
  expected.conv_name = 'd';

  ASSERT_SFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcScanfParserTest, EvalOneArgWithAllOptions) {
  __llvm_libc::scanf_core::FormatSection format_arr[10];
  const char *str = "%*56jd";
  intmax_t arg1 = 12345;
  evaluate(format_arr, str, &arg1);

  __llvm_libc::scanf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = str;
  expected.flags = __llvm_libc::scanf_core::FormatFlags::NO_WRITE;
  expected.max_width = 56;
  expected.length_modifier = __llvm_libc::scanf_core::LengthModifier::j;
  expected.conv_name = 'd';

  ASSERT_SFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcScanfParserTest, EvalSimpleBracketArg) {
  __llvm_libc::scanf_core::FormatSection format_arr[10];
  const char *str = "%[abc]";
  char arg1 = 'a';
  evaluate(format_arr, str, &arg1);

  __llvm_libc::scanf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = str;
  expected.conv_name = '[';
  expected.output_ptr = &arg1;

  __llvm_libc::cpp::bitset<256> scan_set;

  scan_set.set('a');
  scan_set.set('b');
  scan_set.set('c');

  expected.scan_set = scan_set;

  ASSERT_SFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcScanfParserTest, EvalBracketArgRange) {
  __llvm_libc::scanf_core::FormatSection format_arr[10];
  const char *str = "%[A-D]";
  char arg1 = 'a';
  evaluate(format_arr, str, &arg1);

  __llvm_libc::scanf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = str;
  expected.conv_name = '[';
  expected.output_ptr = &arg1;

  __llvm_libc::cpp::bitset<256> scan_set;

  scan_set.set('A');
  scan_set.set('B');
  scan_set.set('C');
  scan_set.set('D');

  expected.scan_set = scan_set;

  ASSERT_SFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcScanfParserTest, EvalBracketArgTwoRanges) {
  __llvm_libc::scanf_core::FormatSection format_arr[10];
  const char *str = "%[A-De-g]";
  char arg1 = 'a';
  evaluate(format_arr, str, &arg1);

  __llvm_libc::scanf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = str;
  expected.conv_name = '[';
  expected.output_ptr = &arg1;

  __llvm_libc::cpp::bitset<256> scan_set;

  scan_set.set('A');
  scan_set.set('B');
  scan_set.set('C');
  scan_set.set('D');
  scan_set.set_range('e', 'g');

  expected.scan_set = scan_set;

  ASSERT_SFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcScanfParserTest, EvalBracketArgJustHyphen) {
  __llvm_libc::scanf_core::FormatSection format_arr[10];
  const char *str = "%[-]";
  char arg1 = 'a';
  evaluate(format_arr, str, &arg1);

  __llvm_libc::scanf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = str;
  expected.conv_name = '[';
  expected.output_ptr = &arg1;

  __llvm_libc::cpp::bitset<256> scan_set;

  scan_set.set('-');

  expected.scan_set = scan_set;

  ASSERT_SFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcScanfParserTest, EvalBracketArgLeftHyphen) {
  __llvm_libc::scanf_core::FormatSection format_arr[10];
  const char *str = "%[-A]";
  char arg1 = 'a';
  evaluate(format_arr, str, &arg1);

  __llvm_libc::scanf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = str;
  expected.conv_name = '[';
  expected.output_ptr = &arg1;

  __llvm_libc::cpp::bitset<256> scan_set;

  scan_set.set('-');
  scan_set.set('A');

  expected.scan_set = scan_set;

  ASSERT_SFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcScanfParserTest, EvalBracketArgRightHyphen) {
  __llvm_libc::scanf_core::FormatSection format_arr[10];
  const char *str = "%[Z-]";
  char arg1 = 'a';
  evaluate(format_arr, str, &arg1);

  __llvm_libc::scanf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = str;
  expected.conv_name = '[';
  expected.output_ptr = &arg1;

  __llvm_libc::cpp::bitset<256> scan_set;

  scan_set.set('-');
  scan_set.set('Z');

  expected.scan_set = scan_set;

  ASSERT_SFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcScanfParserTest, EvalBracketArgInvertSimple) {
  __llvm_libc::scanf_core::FormatSection format_arr[10];
  const char *str = "%[^abc]";
  char arg1 = 'a';
  evaluate(format_arr, str, &arg1);

  __llvm_libc::scanf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = str;
  expected.conv_name = '[';
  expected.output_ptr = &arg1;

  __llvm_libc::cpp::bitset<256> scan_set;

  scan_set.set('a');
  scan_set.set('b');
  scan_set.set('c');
  scan_set.flip();

  expected.scan_set = scan_set;

  ASSERT_SFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcScanfParserTest, EvalBracketArgInvertRange) {
  __llvm_libc::scanf_core::FormatSection format_arr[10];
  const char *str = "%[^0-9]";
  char arg1 = 'a';
  evaluate(format_arr, str, &arg1);

  __llvm_libc::scanf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = str;
  expected.conv_name = '[';
  expected.output_ptr = &arg1;

  __llvm_libc::cpp::bitset<256> scan_set;

  scan_set.set_range('0', '9');
  scan_set.flip();

  expected.scan_set = scan_set;

  ASSERT_SFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcScanfParserTest, EvalBracketArgRightBracket) {
  __llvm_libc::scanf_core::FormatSection format_arr[10];
  const char *str = "%[]]";
  char arg1 = 'a';
  evaluate(format_arr, str, &arg1);

  __llvm_libc::scanf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = str;
  expected.conv_name = '[';
  expected.output_ptr = &arg1;

  __llvm_libc::cpp::bitset<256> scan_set;

  scan_set.set(']');

  expected.scan_set = scan_set;

  ASSERT_SFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcScanfParserTest, EvalBracketArgRightBracketRange) {
  __llvm_libc::scanf_core::FormatSection format_arr[10];
  const char *str = "%[]-a]";
  char arg1 = 'a';
  evaluate(format_arr, str, &arg1);

  __llvm_libc::scanf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = str;
  expected.conv_name = '[';
  expected.output_ptr = &arg1;

  __llvm_libc::cpp::bitset<256> scan_set;

  scan_set.set_range(']', 'a');

  expected.scan_set = scan_set;

  ASSERT_SFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcScanfParserTest, EvalBracketArgRightBracketInvert) {
  __llvm_libc::scanf_core::FormatSection format_arr[10];
  const char *str = "%[^]]";
  char arg1 = 'a';
  evaluate(format_arr, str, &arg1);

  __llvm_libc::scanf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = str;
  expected.conv_name = '[';
  expected.output_ptr = &arg1;

  __llvm_libc::cpp::bitset<256> scan_set;

  scan_set.set(']');
  scan_set.flip();

  expected.scan_set = scan_set;

  ASSERT_SFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcScanfParserTest, EvalBracketArgRightBracketInvertRange) {
  __llvm_libc::scanf_core::FormatSection format_arr[10];
  const char *str = "%[^]-^]";
  char arg1 = 'a';
  evaluate(format_arr, str, &arg1);

  __llvm_libc::scanf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = str;
  expected.conv_name = '[';
  expected.output_ptr = &arg1;

  __llvm_libc::cpp::bitset<256> scan_set;

  scan_set.set_range(']', '^');
  scan_set.flip();

  expected.scan_set = scan_set;

  ASSERT_SFORMAT_EQ(expected, format_arr[0]);
}

// This is not part of the standard, but the hyphen's effect is always
// implementation defined, and I have defined it such that it will capture the
// correct range regardless of the order of the characters.
TEST(LlvmLibcScanfParserTest, EvalBracketArgBackwardsRange) {
  __llvm_libc::scanf_core::FormatSection format_arr[10];
  const char *str = "%[9-0]";
  char arg1 = 'a';
  evaluate(format_arr, str, &arg1);

  __llvm_libc::scanf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = str;
  expected.conv_name = '[';
  expected.output_ptr = &arg1;

  __llvm_libc::cpp::bitset<256> scan_set;

  scan_set.set_range('0', '9');

  expected.scan_set = scan_set;

  ASSERT_SFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcScanfParserTest, EvalThreeArgs) {
  __llvm_libc::scanf_core::FormatSection format_arr[10];
  const char *str = "%d%f%s";
  int arg1 = 12345;
  double arg2 = 123.45;
  const char *arg3 = "12345";
  evaluate(format_arr, str, &arg1, &arg2, &arg3);

  __llvm_libc::scanf_core::FormatSection expected0, expected1, expected2;
  expected0.has_conv = true;

  expected0.raw_string = {str, 2};
  expected0.output_ptr = &arg1;
  expected0.conv_name = 'd';

  ASSERT_SFORMAT_EQ(expected0, format_arr[0]);

  expected1.has_conv = true;

  expected1.raw_string = {str + 2, 2};
  expected1.output_ptr = &arg2;
  expected1.conv_name = 'f';

  ASSERT_SFORMAT_EQ(expected1, format_arr[1]);

  expected2.has_conv = true;

  expected2.raw_string = {str + 4, 2};
  expected2.output_ptr = &arg3;
  expected2.conv_name = 's';

  ASSERT_SFORMAT_EQ(expected2, format_arr[2]);
}

#ifndef LLVM_LIBC_SCANF_DISABLE_INDEX_MODE

TEST(LlvmLibcScanfParserTest, IndexModeOneArg) {
  __llvm_libc::scanf_core::FormatSection format_arr[10];
  const char *str = "%1$d";
  int arg1 = 12345;
  evaluate(format_arr, str, &arg1);

  __llvm_libc::scanf_core::FormatSection expected;
  expected.has_conv = true;

  expected.raw_string = {str, 4};
  expected.output_ptr = &arg1;
  expected.conv_name = 'd';

  ASSERT_SFORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcScanfParserTest, IndexModeThreeArgsSequential) {
  __llvm_libc::scanf_core::FormatSection format_arr[10];
  const char *str = "%1$d%2$f%3$s";
  int arg1 = 12345;
  double arg2 = 123.45;
  const char *arg3 = "12345";
  evaluate(format_arr, str, &arg1, &arg2, &arg3);

  __llvm_libc::scanf_core::FormatSection expected0, expected1, expected2;
  expected0.has_conv = true;

  expected0.raw_string = {str, 4};
  expected0.output_ptr = &arg1;
  expected0.conv_name = 'd';

  ASSERT_SFORMAT_EQ(expected0, format_arr[0]);

  expected1.has_conv = true;

  expected1.raw_string = {str + 4, 4};
  expected1.output_ptr = &arg2;
  expected1.conv_name = 'f';

  ASSERT_SFORMAT_EQ(expected1, format_arr[1]);

  expected2.has_conv = true;

  expected2.raw_string = {str + 8, 4};
  expected2.output_ptr = &arg3;
  expected2.conv_name = 's';

  ASSERT_SFORMAT_EQ(expected2, format_arr[2]);
}

TEST(LlvmLibcScanfParserTest, IndexModeThreeArgsReverse) {
  __llvm_libc::scanf_core::FormatSection format_arr[10];
  const char *str = "%3$d%2$f%1$s";
  int arg1 = 12345;
  double arg2 = 123.45;
  const char *arg3 = "12345";
  evaluate(format_arr, str, &arg3, &arg2, &arg1);

  __llvm_libc::scanf_core::FormatSection expected0, expected1, expected2;
  expected0.has_conv = true;

  expected0.raw_string = {str, 4};
  expected0.output_ptr = &arg1;
  expected0.conv_name = 'd';

  ASSERT_SFORMAT_EQ(expected0, format_arr[0]);

  expected1.has_conv = true;

  expected1.raw_string = {str + 4, 4};
  expected1.output_ptr = &arg2;
  expected1.conv_name = 'f';

  ASSERT_SFORMAT_EQ(expected1, format_arr[1]);

  expected2.has_conv = true;

  expected2.raw_string = {str + 8, 4};
  expected2.output_ptr = &arg3;
  expected2.conv_name = 's';

  ASSERT_SFORMAT_EQ(expected2, format_arr[2]);
}

TEST(LlvmLibcScanfParserTest, IndexModeTenArgsRandom) {
  __llvm_libc::scanf_core::FormatSection format_arr[10];
  const char *str = "%6$d%3$d%7$d%2$d%8$d%1$d%4$d%9$d%5$d%10$d";
  uintptr_t args[10] = {6, 4, 2, 7, 9, 1, 3, 5, 8, 10};
  evaluate(format_arr, str, args[0], args[1], args[2], args[3], args[4],
           args[5], args[6], args[7], args[8], args[9]);

  for (size_t i = 0; i < 10; ++i) {
    __llvm_libc::scanf_core::FormatSection expected;
    expected.has_conv = true;

    expected.raw_string = {str + (4 * i),
                           static_cast<size_t>(4 + (i >= 9 ? 1 : 0))};
    expected.output_ptr = reinterpret_cast<void *>(i + 1);
    expected.conv_name = 'd';
    EXPECT_SFORMAT_EQ(expected, format_arr[i]);
  }
}

TEST(LlvmLibcScanfParserTest, IndexModeComplexParsing) {
  __llvm_libc::scanf_core::FormatSection format_arr[11];
  const char *str = "normal text %3$llu %% %2$*f %4$d %1$1c%5$[123]";
  char arg1 = '1';
  double arg2 = 123.45;
  unsigned long long arg3 = 12345;
  int arg4 = 10;
  char arg5 = 'A';
  evaluate(format_arr, str, &arg1, &arg2, &arg3, &arg4, &arg5);

  __llvm_libc::scanf_core::FormatSection expected0, expected1, expected2,
      expected3, expected4, expected5, expected6, expected7, expected8,
      expected9, expected10;

  expected0.has_conv = false;

  // "normal text "
  expected0.raw_string = {str, 12};

  EXPECT_SFORMAT_EQ(expected0, format_arr[0]);

  expected1.has_conv = true;

  // "%3$llu"
  expected1.raw_string = {str + 12, 6};
  expected1.length_modifier = __llvm_libc::scanf_core::LengthModifier::ll;
  expected1.output_ptr = &arg3;
  expected1.conv_name = 'u';

  EXPECT_SFORMAT_EQ(expected1, format_arr[1]);

  expected2.has_conv = false;

  // " "
  expected2.raw_string = {str + 18, 1};

  EXPECT_SFORMAT_EQ(expected2, format_arr[2]);

  expected3.has_conv = true;

  expected3.raw_string = {str + 19, 2};
  expected3.conv_name = '%';

  EXPECT_SFORMAT_EQ(expected3, format_arr[3]);

  expected4.has_conv = false;

  // " "
  expected4.raw_string = {str + 21, 1};

  EXPECT_SFORMAT_EQ(expected4, format_arr[4]);

  expected5.has_conv = true;

  // "%%"
  expected5.raw_string = {str + 22, 5};
  expected5.flags = __llvm_libc::scanf_core::FormatFlags::NO_WRITE;
  expected5.conv_name = 'f';

  EXPECT_SFORMAT_EQ(expected5, format_arr[5]);

  expected6.has_conv = false;

  // " "
  expected6.raw_string = {str + 27, 1};

  EXPECT_SFORMAT_EQ(expected6, format_arr[6]);

  expected7.has_conv = true;

  // "%2$*f"
  expected7.raw_string = {str + 28, 4};
  expected7.output_ptr = &arg4;
  expected7.conv_name = 'd';

  EXPECT_SFORMAT_EQ(expected7, format_arr[7]);

  expected8.has_conv = false;

  // " "
  expected8.raw_string = {str + 32, 1};

  EXPECT_SFORMAT_EQ(expected8, format_arr[8]);

  expected9.has_conv = true;

  // "%1$1c"
  expected9.raw_string = {str + 33, 5};
  expected9.max_width = 1;
  expected9.output_ptr = &arg1;
  expected9.conv_name = 'c';

  EXPECT_SFORMAT_EQ(expected9, format_arr[9]);

  expected9.has_conv = true;

  // "%5$[123]"
  expected10.raw_string = {str + 38, 8};
  expected10.output_ptr = &arg5;
  expected10.conv_name = '[';

  __llvm_libc::cpp::bitset<256> scan_set;

  scan_set.set_range('1', '3');

  expected10.scan_set = scan_set;

  EXPECT_SFORMAT_EQ(expected10, format_arr[10]);
}

#endif // LLVM_LIBC_SCANF_DISABLE_INDEX_MODE
