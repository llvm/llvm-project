//===-- Unittests for ispunct----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ctype/ispunct.h"
#include "test/UnitTest/Test.h"

// Helper function to mark the sections of the ASCII table that are
// punctuation characters. These are listed below:
//  Decimal    |         Symbol
//  -----------------------------------------
//  33 -  47   |  ! " $ % & ' ( ) * + , - . /
//  58 -  64   |  : ; < = > ? @
//  91 -  96   |  [ \ ] ^ _ `
// 123 - 126   |  { | } ~
static inline int is_punctuation_character(int c) {
  return ('!' <= c && c <= '/') || (':' <= c && c <= '@') ||
         ('[' <= c && c <= '`') || ('{' <= c && c <= '~');
}

TEST(LlvmLibcIsPunct, DefaultLocale) {
  // Loops through all characters, verifying that punctuation characters
  // return a non-zero integer, and everything else returns zero.
  for (int ch = -255; ch < 255; ++ch) {
    if (is_punctuation_character(ch))
      EXPECT_NE(LIBC_NAMESPACE::ispunct(ch), 0);
    else
      EXPECT_EQ(LIBC_NAMESPACE::ispunct(ch), 0);
  }
}
