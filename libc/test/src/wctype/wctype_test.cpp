//===-- Unittests for wctype --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wctype/wctype.h"

#include "test/UnitTest/Test.h"

// wctype descriptors (must match implementation)
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

TEST(LlvmLibcwctype, SimpleTest) {
  using LIBC_NAMESPACE::wctype;

  auto alnum = wctype("alnum");
  auto alpha = wctype("alpha");
  auto blank = wctype("blank");
  auto cntrl = wctype("cntrl");
  auto digit = wctype("digit");
  auto graph = wctype("graph");
  auto lower = wctype("lower");
  auto print = wctype("print");
  auto punct = wctype("punct");
  auto space = wctype("space");
  auto upper = wctype("upper");
  auto xdigit = wctype("xdigit");

  // valid descriptors should be nonzero
  EXPECT_EQ(alnum, WCTYPE_ALNUM);
  EXPECT_EQ(alpha, WCTYPE_ALPHA);
  EXPECT_EQ(blank, WCTYPE_BLANK);
  EXPECT_EQ(cntrl, WCTYPE_CNTRL);
  EXPECT_EQ(digit, WCTYPE_DIGIT);
  EXPECT_EQ(graph, WCTYPE_GRAPH);
  EXPECT_EQ(lower, WCTYPE_LOWER);
  EXPECT_EQ(print, WCTYPE_PRINT);
  EXPECT_EQ(punct, WCTYPE_PUNCT);
  EXPECT_EQ(space, WCTYPE_SPACE);
  EXPECT_EQ(upper, WCTYPE_UPPER);
  EXPECT_EQ(xdigit, WCTYPE_XDIGIT);

  // invalid properties should return zero
  EXPECT_EQ(wctype(""), WCTYPE_INVALID);
  EXPECT_EQ(wctype("invalid"), WCTYPE_INVALID);
  EXPECT_EQ(wctype("Alpha"), WCTYPE_INVALID);
  EXPECT_EQ(wctype("unknown"), WCTYPE_INVALID);
}
