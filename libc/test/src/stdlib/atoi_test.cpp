//===-- Unittests for atoi -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/atoi.h"

#include "utils/UnitTest/Test.h"

#include <limits.h>

TEST(LlvmLibcAToITest, ValidNumbers) {
  const char *zero = "0";
  ASSERT_EQ(__llvm_libc::atoi(zero), 0);

  const char *ten = "10";
  ASSERT_EQ(__llvm_libc::atoi(ten), 10);

  const char *negative_hundred = "-100";
  ASSERT_EQ(__llvm_libc::atoi(negative_hundred), -100);

  const char *positive_thousand = "+1000";
  ASSERT_EQ(__llvm_libc::atoi(positive_thousand), 1000);

  const char *spaces_before = "     12345";
  ASSERT_EQ(__llvm_libc::atoi(spaces_before), 12345);

  const char *tabs_before = "\t\t\t\t67890";
  ASSERT_EQ(__llvm_libc::atoi(tabs_before), 67890);

  const char *letters_after = "123abc";
  ASSERT_EQ(__llvm_libc::atoi(letters_after), 123);

  const char *letters_between = "456def789";
  ASSERT_EQ(__llvm_libc::atoi(letters_between), 456);

  const char *all_together = "\t   110 times 5 = 550";
  ASSERT_EQ(__llvm_libc::atoi(all_together), 110);

  const char *biggest_int = "2147483647";
  ASSERT_EQ(__llvm_libc::atoi(biggest_int), INT_MAX);

  const char *smallest_int = "-2147483648";
  ASSERT_EQ(__llvm_libc::atoi(smallest_int), INT_MIN);
}

TEST(LlvmLibcAToITest, NonBaseTenWholeNumbers) {
  const char *hexadecimal = "0x10";
  ASSERT_EQ(__llvm_libc::atoi(hexadecimal), 0);

  const char *octal = "010";
  ASSERT_EQ(__llvm_libc::atoi(octal), 10);

  const char *decimal_point = "5.9";
  ASSERT_EQ(__llvm_libc::atoi(decimal_point), 5);
}

TEST(LlvmLibcAToITest, NotNumbers) {
  const char *ten_as_word = "ten";
  ASSERT_EQ(__llvm_libc::atoi(ten_as_word), 0);

  const char *lots_of_letters =
      "wtragsdhfgjykutjdyfhgnchgmjhkyurktfgjhlu;po7urtdjyfhgklyk";
  ASSERT_EQ(__llvm_libc::atoi(lots_of_letters), 0);
}
