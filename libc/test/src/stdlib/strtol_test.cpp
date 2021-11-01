//===-- Unittests for strtol ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/strtol.h"

#include "utils/UnitTest/Test.h"

#include <errno.h>
#include <limits.h>
#include <stddef.h>

TEST(LlvmLibcStrToLTest, InvalidBase) {
  const char *ten = "10";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(ten, nullptr, -1), 0l);
  ASSERT_EQ(errno, EINVAL);
}

TEST(LlvmLibcStrToLTest, CleanBaseTenDecode) {
  char *str_end = nullptr;

  const char *ten = "10";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(ten, &str_end, 10), 10l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - ten, ptrdiff_t(2));

  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(ten, nullptr, 10), 10l);
  ASSERT_EQ(errno, 0);

  const char *hundred = "100";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(hundred, &str_end, 10), 100l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - hundred, ptrdiff_t(3));

  const char *negative = "-100";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(negative, &str_end, 10), -100l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - negative, ptrdiff_t(4));

  const char *big_number = "1234567890";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(big_number, &str_end, 10), 1234567890l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - big_number, ptrdiff_t(10));

  const char *big_negative_number = "-1234567890";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(big_negative_number, &str_end, 10),
            -1234567890l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - big_negative_number, ptrdiff_t(11));

  const char *too_big_number = "123456789012345678901";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(too_big_number, &str_end, 10), LONG_MAX);
  ASSERT_EQ(errno, ERANGE);
  EXPECT_EQ(str_end - too_big_number, ptrdiff_t(21));

  const char *too_big_negative_number = "-123456789012345678901";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(too_big_negative_number, &str_end, 10),
            LONG_MIN);
  ASSERT_EQ(errno, ERANGE);
  EXPECT_EQ(str_end - too_big_negative_number, ptrdiff_t(22));

  const char *long_number_range_test =
      "10000000000000000000000000000000000000000000000000";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(long_number_range_test, &str_end, 10),
            LONG_MAX);
  ASSERT_EQ(errno, ERANGE);
  EXPECT_EQ(str_end - long_number_range_test, ptrdiff_t(50));
}

TEST(LlvmLibcStrToLTest, MessyBaseTenDecode) {
  char *str_end = nullptr;

  const char *spaces_before = "     10";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(spaces_before, &str_end, 10), 10l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - spaces_before, ptrdiff_t(7));

  const char *spaces_after = "10      ";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(spaces_after, &str_end, 10), 10l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - spaces_after, ptrdiff_t(2));

  const char *word_before = "word10";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(word_before, &str_end, 10), 0l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - word_before, ptrdiff_t(0));

  const char *word_after = "10word";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(word_after, &str_end, 10), 10l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - word_after, ptrdiff_t(2));

  const char *two_numbers = "10 999";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(two_numbers, &str_end, 10), 10l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - two_numbers, ptrdiff_t(2));

  const char *two_signs = "--10 999";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(two_signs, &str_end, 10), 0l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - two_signs, ptrdiff_t(0));

  const char *sign_before = "+2=4";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(sign_before, &str_end, 10), 2l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - sign_before, ptrdiff_t(2));

  const char *sign_after = "2+2=4";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(sign_after, &str_end, 10), 2l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - sign_after, ptrdiff_t(1));

  const char *tab_before = "\t10";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(tab_before, &str_end, 10), 10l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - tab_before, ptrdiff_t(3));

  const char *all_together = "\t  -12345and+67890";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(all_together, &str_end, 10), -12345l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - all_together, ptrdiff_t(9));

  const char *just_spaces = "  ";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(just_spaces, &str_end, 10), 0l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - just_spaces, ptrdiff_t(0));

  const char *just_space_and_sign = " +";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(just_space_and_sign, &str_end, 10), 0l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - just_space_and_sign, ptrdiff_t(0));
}

static char int_to_b36_char(int input) {
  if (input < 0 || input > 36)
    return '0';
  if (input < 10)
    return static_cast<char>('0' + input);
  return static_cast<char>('A' + input - 10);
}

TEST(LlvmLibcStrToLTest, DecodeInOtherBases) {
  char small_string[4] = {'\0', '\0', '\0', '\0'};
  for (int base = 2; base <= 36; ++base) {
    for (int first_digit = 0; first_digit <= 36; ++first_digit) {
      small_string[0] = int_to_b36_char(first_digit);
      if (first_digit < base) {
        errno = 0;
        ASSERT_EQ(__llvm_libc::strtol(small_string, nullptr, base),
                  static_cast<long int>(first_digit));
        ASSERT_EQ(errno, 0);
      } else {
        errno = 0;
        ASSERT_EQ(__llvm_libc::strtol(small_string, nullptr, base), 0l);
        ASSERT_EQ(errno, 0);
      }
    }
  }

  for (int base = 2; base <= 36; ++base) {
    for (int first_digit = 0; first_digit <= 36; ++first_digit) {
      small_string[0] = int_to_b36_char(first_digit);
      for (int second_digit = 0; second_digit <= 36; ++second_digit) {
        small_string[1] = int_to_b36_char(second_digit);
        if (first_digit < base && second_digit < base) {
          errno = 0;
          ASSERT_EQ(__llvm_libc::strtol(small_string, nullptr, base),
                    static_cast<long int>(second_digit + (first_digit * base)));
          ASSERT_EQ(errno, 0);
        } else if (first_digit < base) {
          errno = 0;
          ASSERT_EQ(__llvm_libc::strtol(small_string, nullptr, base),
                    static_cast<long int>(first_digit));
          ASSERT_EQ(errno, 0);
        } else {
          errno = 0;
          ASSERT_EQ(__llvm_libc::strtol(small_string, nullptr, base), 0l);
          ASSERT_EQ(errno, 0);
        }
      }
    }
  }

  for (int base = 2; base <= 36; ++base) {
    for (int first_digit = 0; first_digit <= 36; ++first_digit) {
      small_string[0] = int_to_b36_char(first_digit);
      for (int second_digit = 0; second_digit <= 36; ++second_digit) {
        small_string[1] = int_to_b36_char(second_digit);
        for (int third_digit = 0; third_digit <= 36; ++third_digit) {
          small_string[2] = int_to_b36_char(third_digit);

          if (first_digit < base && second_digit < base && third_digit < base) {
            errno = 0;
            ASSERT_EQ(__llvm_libc::strtol(small_string, nullptr, base),
                      static_cast<long int>(third_digit +
                                            (second_digit * base) +
                                            (first_digit * base * base)));
            ASSERT_EQ(errno, 0);
          } else if (first_digit < base && second_digit < base) {
            errno = 0;
            ASSERT_EQ(
                __llvm_libc::strtol(small_string, nullptr, base),
                static_cast<long int>(second_digit + (first_digit * base)));
            ASSERT_EQ(errno, 0);
          } else if (first_digit < base) {
            // if the base is 16 there is a special case for the prefix 0X.
            // The number is treated as a one digit hexadecimal.
            if (base == 16 && first_digit == 0 && second_digit == 33) {
              if (third_digit < base) {
                errno = 0;
                ASSERT_EQ(__llvm_libc::strtol(small_string, nullptr, base),
                          static_cast<long int>(third_digit));
                ASSERT_EQ(errno, 0);
              } else {
                errno = 0;
                ASSERT_EQ(__llvm_libc::strtol(small_string, nullptr, base), 0l);
                ASSERT_EQ(errno, 0);
              }
            } else {
              errno = 0;
              ASSERT_EQ(__llvm_libc::strtol(small_string, nullptr, base),
                        static_cast<long int>(first_digit));
              ASSERT_EQ(errno, 0);
            }
          } else {
            errno = 0;
            ASSERT_EQ(__llvm_libc::strtol(small_string, nullptr, base), 0l);
            ASSERT_EQ(errno, 0);
          }
        }
      }
    }
  }
}

TEST(LlvmLibcStrToLTest, CleanBaseSixteenDecode) {
  char *str_end = nullptr;

  const char *no_prefix = "123abc";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(no_prefix, &str_end, 16), 0x123abcl);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - no_prefix, ptrdiff_t(6));

  const char *yes_prefix = "0x456def";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(yes_prefix, &str_end, 16), 0x456defl);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - yes_prefix, ptrdiff_t(8));

  const char *letter_after_prefix = "0xabc123";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(letter_after_prefix, &str_end, 16), 0xabc123l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - letter_after_prefix, ptrdiff_t(8));
}

TEST(LlvmLibcStrToLTest, MessyBaseSixteenDecode) {
  char *str_end = nullptr;

  const char *just_prefix = "0x";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(just_prefix, &str_end, 16), 0l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - just_prefix, ptrdiff_t(1));

  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(just_prefix, &str_end, 0), 0l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - just_prefix, ptrdiff_t(1));

  const char *prefix_with_x_after = "0xx";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(prefix_with_x_after, &str_end, 16), 0l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - prefix_with_x_after, ptrdiff_t(1));

  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(prefix_with_x_after, &str_end, 0), 0l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - prefix_with_x_after, ptrdiff_t(1));
}

TEST(LlvmLibcStrToLTest, AutomaticBaseSelection) {
  char *str_end = nullptr;

  const char *base_ten = "12345";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(base_ten, &str_end, 0), 12345l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - base_ten, ptrdiff_t(5));

  const char *base_sixteen_no_prefix = "123abc";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(base_sixteen_no_prefix, &str_end, 0), 123l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - base_sixteen_no_prefix, ptrdiff_t(3));

  const char *base_sixteen_with_prefix = "0x456def";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(base_sixteen_with_prefix, &str_end, 0),
            0x456defl);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - base_sixteen_with_prefix, ptrdiff_t(8));

  const char *base_eight_with_prefix = "012345";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(base_eight_with_prefix, &str_end, 0), 012345l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - base_eight_with_prefix, ptrdiff_t(6));

  const char *just_zero = "0";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(just_zero, &str_end, 0), 0l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - just_zero, ptrdiff_t(1));

  const char *just_zero_x = "0x";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(just_zero_x, &str_end, 0), 0l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - just_zero_x, ptrdiff_t(1));

  const char *just_zero_eight = "08";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtol(just_zero_eight, &str_end, 0), 0l);
  ASSERT_EQ(errno, 0);
  EXPECT_EQ(str_end - just_zero_eight, ptrdiff_t(1));
}
