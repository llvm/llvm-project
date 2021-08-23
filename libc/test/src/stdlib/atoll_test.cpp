//===-- Unittests for atoll -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/atoll.h"

#include "utils/UnitTest/Test.h"

#include <limits.h>

TEST(LlvmLibcAToLLTest, ValidNumbers) {
  const char *zero = "0";
  ASSERT_EQ(__llvm_libc::atoll(zero), 0ll);

  const char *ten = "10";
  ASSERT_EQ(__llvm_libc::atoll(ten), 10ll);

  const char *negative_hundred = "-100";
  ASSERT_EQ(__llvm_libc::atoll(negative_hundred), -100ll);

  const char *positive_thousand = "+1000";
  ASSERT_EQ(__llvm_libc::atoll(positive_thousand), 1000ll);

  const char *spaces_before = "     12345";
  ASSERT_EQ(__llvm_libc::atoll(spaces_before), 12345ll);

  const char *tabs_before = "\t\t\t\t67890";
  ASSERT_EQ(__llvm_libc::atoll(tabs_before), 67890ll);

  const char *letters_after = "123abc";
  ASSERT_EQ(__llvm_libc::atoll(letters_after), 123ll);

  const char *letters_between = "456def789";
  ASSERT_EQ(__llvm_libc::atoll(letters_between), 456ll);

  const char *all_together = "\t   110 times 5 = 550";
  ASSERT_EQ(__llvm_libc::atoll(all_together), 110ll);

  const char *biggest_long_long = "9223372036854775807";
  ASSERT_EQ(__llvm_libc::atoll(biggest_long_long), LLONG_MAX);

  const char *smallest_long_long = "-9223372036854775808";
  ASSERT_EQ(__llvm_libc::atoll(smallest_long_long), LLONG_MIN);
}

TEST(LlvmLibcAToLLTest, NonBaseTenWholeNumbers) {
  const char *hexadecimal = "0x10";
  ASSERT_EQ(__llvm_libc::atoll(hexadecimal), 0ll);

  const char *octal = "010";
  ASSERT_EQ(__llvm_libc::atoll(octal), 10ll);

  const char *decimal_point = "5.9";
  ASSERT_EQ(__llvm_libc::atoll(decimal_point), 5ll);
}

TEST(LlvmLibcAToLLTest, NotNumbers) {
  const char *ten_as_word = "ten";
  ASSERT_EQ(__llvm_libc::atoll(ten_as_word), 0ll);

  const char *lots_of_letters =
      "wtragsdhfgjykutjdyfhgnchgmjhkyurktfgjhlu;po7urtdjyfhgklyk";
  ASSERT_EQ(__llvm_libc::atoll(lots_of_letters), 0ll);
}
