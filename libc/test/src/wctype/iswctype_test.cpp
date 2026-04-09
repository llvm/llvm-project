//===-- Unittests for iswctype --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wctype/iswctype.h"

#include "test/UnitTest/Test.h"

// Simple tests, already properly tested in
// libc/test/src/__support/wctype_utils_test.cpp

static constexpr wctype_t WCTYPE_INVALID = static_cast<wctype_t>(0);
static constexpr wctype_t WCTYPE_ALNUM = static_cast<wctype_t>(1);
static constexpr wctype_t WCTYPE_ALPHA = static_cast<wctype_t>(2);
static constexpr wctype_t WCTYPE_BLANK = static_cast<wctype_t>(3);
static constexpr wctype_t WCTYPE_CNTRL = static_cast<wctype_t>(4);
static constexpr wctype_t WCTYPE_DIGIT = static_cast<wctype_t>(5);
static constexpr wctype_t WCTYPE_GRAPH = static_cast<wctype_t>(6);
static constexpr wctype_t WCTYPE_LOWER = static_cast<wctype_t>(7);
static constexpr wctype_t WCTYPE_PRINT = static_cast<wctype_t>(8);
static constexpr wctype_t WCTYPE_PUNCT = static_cast<wctype_t>(9);
static constexpr wctype_t WCTYPE_SPACE = static_cast<wctype_t>(10);
static constexpr wctype_t WCTYPE_UPPER = static_cast<wctype_t>(11);
static constexpr wctype_t WCTYPE_XDIGIT = static_cast<wctype_t>(12);

TEST(LlvmLibciswctype, SimpleTest) {
  using LIBC_NAMESPACE::iswctype;

  // alnum
  EXPECT_NE(iswctype('a', WCTYPE_ALNUM), 0);
  EXPECT_NE(iswctype('Z', WCTYPE_ALNUM), 0);
  EXPECT_NE(iswctype('5', WCTYPE_ALNUM), 0);
  EXPECT_EQ(iswctype('!', WCTYPE_ALNUM), 0);

  // alpha
  EXPECT_NE(iswctype('a', WCTYPE_ALPHA), 0);
  EXPECT_NE(iswctype('Z', WCTYPE_ALPHA), 0);
  EXPECT_EQ(iswctype('1', WCTYPE_ALPHA), 0);
  EXPECT_EQ(iswctype(' ', WCTYPE_ALPHA), 0);

  // blank
  EXPECT_NE(iswctype(' ', WCTYPE_BLANK), 0);
  EXPECT_NE(iswctype('\t', WCTYPE_BLANK), 0);
  EXPECT_EQ(iswctype('\n', WCTYPE_BLANK), 0);
  EXPECT_EQ(iswctype('A', WCTYPE_BLANK), 0);

  // cntrl
  EXPECT_NE(iswctype('\0', WCTYPE_CNTRL), 0);
  EXPECT_NE(iswctype('\n', WCTYPE_CNTRL), 0);
  EXPECT_NE(iswctype(0x7f, WCTYPE_CNTRL), 0);
  EXPECT_EQ(iswctype('A', WCTYPE_CNTRL), 0);

  // digit
  EXPECT_NE(iswctype('0', WCTYPE_DIGIT), 0);
  EXPECT_NE(iswctype('9', WCTYPE_DIGIT), 0);
  EXPECT_EQ(iswctype('a', WCTYPE_DIGIT), 0);
  EXPECT_EQ(iswctype(' ', WCTYPE_DIGIT), 0);

  // graph
  EXPECT_NE(iswctype('A', WCTYPE_GRAPH), 0);
  EXPECT_NE(iswctype('1', WCTYPE_GRAPH), 0);
  EXPECT_NE(iswctype('!', WCTYPE_GRAPH), 0);
  EXPECT_EQ(iswctype(' ', WCTYPE_GRAPH), 0);

  // lower
  EXPECT_NE(iswctype('a', WCTYPE_LOWER), 0);
  EXPECT_NE(iswctype('z', WCTYPE_LOWER), 0);
  EXPECT_EQ(iswctype('A', WCTYPE_LOWER), 0);
  EXPECT_EQ(iswctype('1', WCTYPE_LOWER), 0);

  // print
  EXPECT_NE(iswctype(' ', WCTYPE_PRINT), 0);
  EXPECT_NE(iswctype('A', WCTYPE_PRINT), 0);
  EXPECT_NE(iswctype('~', WCTYPE_PRINT), 0);
  EXPECT_EQ(iswctype('\n', WCTYPE_PRINT), 0);

  // punct
  EXPECT_NE(iswctype('!', WCTYPE_PUNCT), 0);
  EXPECT_NE(iswctype('?', WCTYPE_PUNCT), 0);
  EXPECT_EQ(iswctype('a', WCTYPE_PUNCT), 0);
  EXPECT_EQ(iswctype('1', WCTYPE_PUNCT), 0);

  // space
  EXPECT_NE(iswctype(' ', WCTYPE_SPACE), 0);
  EXPECT_NE(iswctype('\t', WCTYPE_SPACE), 0);
  EXPECT_NE(iswctype('\n', WCTYPE_SPACE), 0);
  EXPECT_EQ(iswctype('A', WCTYPE_SPACE), 0);

  // upper
  EXPECT_NE(iswctype('A', WCTYPE_UPPER), 0);
  EXPECT_NE(iswctype('Z', WCTYPE_UPPER), 0);
  EXPECT_EQ(iswctype('a', WCTYPE_UPPER), 0);
  EXPECT_EQ(iswctype('1', WCTYPE_UPPER), 0);

  // xdigit
  EXPECT_NE(iswctype('0', WCTYPE_XDIGIT), 0);
  EXPECT_NE(iswctype('9', WCTYPE_XDIGIT), 0);
  EXPECT_NE(iswctype('a', WCTYPE_XDIGIT), 0);
  EXPECT_NE(iswctype('F', WCTYPE_XDIGIT), 0);
  EXPECT_EQ(iswctype('g', WCTYPE_XDIGIT), 0);
  EXPECT_EQ(iswctype('?', WCTYPE_XDIGIT), 0);

  // invalid descriptor
  EXPECT_EQ(iswctype('a', WCTYPE_INVALID), 0);
  EXPECT_EQ(iswctype('a', static_cast<wctype_t>(999)), 0);
}
