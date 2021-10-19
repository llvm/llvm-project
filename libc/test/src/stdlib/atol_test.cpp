//===-- Unittests for atol -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/atol.h"

#include "utils/UnitTest/Test.h"

#include <limits.h>

TEST(LlvmLibcAToLTest, ValidNumbers) {
  const char *zero = "0";
  ASSERT_EQ(__llvm_libc::atol(zero), 0l);

  const char *ten = "10";
  ASSERT_EQ(__llvm_libc::atol(ten), 10l);

  const char *negative_hundred = "-100";
  ASSERT_EQ(__llvm_libc::atol(negative_hundred), -100l);

  const char *positive_thousand = "+1000";
  ASSERT_EQ(__llvm_libc::atol(positive_thousand), 1000l);

  const char *spaces_before = "     12345";
  ASSERT_EQ(__llvm_libc::atol(spaces_before), 12345l);

  const char *tabs_before = "\t\t\t\t67890";
  ASSERT_EQ(__llvm_libc::atol(tabs_before), 67890l);

  const char *letters_after = "123abc";
  ASSERT_EQ(__llvm_libc::atol(letters_after), 123l);

  const char *letters_between = "456def789";
  ASSERT_EQ(__llvm_libc::atol(letters_between), 456l);

  const char *all_together = "\t   110 times 5 = 550";
  ASSERT_EQ(__llvm_libc::atol(all_together), 110l);
}

TEST(LlvmLibcAToLTest, NonBaseTenWholeNumbers) {
  const char *hexadecimal = "0x10";
  ASSERT_EQ(__llvm_libc::atol(hexadecimal), 0l);

  const char *octal = "010";
  ASSERT_EQ(__llvm_libc::atol(octal), 10l);

  const char *decimal_point = "5.9";
  ASSERT_EQ(__llvm_libc::atol(decimal_point), 5l);
}

TEST(LlvmLibcAToLTest, NotNumbers) {
  const char *ten_as_word = "ten";
  ASSERT_EQ(__llvm_libc::atol(ten_as_word), 0l);

  const char *lots_of_letters =
      "wtragsdhfgjykutjdyfhgnchgmjhkyurktfgjhlu;po7urtdjyfhgklyk";
  ASSERT_EQ(__llvm_libc::atol(lots_of_letters), 0l);
}
