//===-- Unittests for stdbit ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/*
 * The intent of this test is validate that:
 * 1. We provide the definition of the various type generic macros of stdbit.h
 * (the macros are transitively included from stdbit-macros.h by stdbit.h).
 * 2. It dispatches to the correct underlying function.
 * Because unit tests build without public packaging, the object files produced
 * do not contain non-namespaced symbols.
 */

/*
 * Declare these BEFORE including stdbit-macros.h so that this test may still be
 * run even if a given target doesn't yet have these individual entrypoints
 * enabled.
 */
#include "stdbit_stub.h"

#include "include/llvm-libc-macros/stdbit-macros.h"
#include "test/UnitTest/LibcCTest.h"

// The trailing comma in a compound literal must not be treated as a macro
// argument separator. Also check that the initializer is evaluated only once.
#define CHECK_COMPOUND_LITERAL(FUNC_NAME, TYPE, VALUE, EXPECTED)               \
  do {                                                                         \
    unsigned evaluations = 0;                                                  \
    EXPECT_TRUE(FUNC_NAME((TYPE){                                              \
                    (evaluations++, VALUE),                                    \
                }) == EXPECTED);                                               \
    EXPECT_TRUE(evaluations == 1);                                             \
  } while (0)

#define CHECK_FUNCTION(FUNC_NAME, VAL)                                         \
  do {                                                                         \
    EXPECT_TRUE(FUNC_NAME((unsigned char)0U) == VAL##AU);                      \
    EXPECT_TRUE(FUNC_NAME((unsigned short)0U) == VAL##BU);                     \
    EXPECT_TRUE(FUNC_NAME(0U) == VAL##CU);                                     \
    EXPECT_TRUE(FUNC_NAME(0UL) == VAL##DU);                                    \
    EXPECT_TRUE(FUNC_NAME(0ULL) == VAL##EU);                                   \
    CHECK_COMPOUND_LITERAL(FUNC_NAME, unsigned char, 0, VAL##AU);              \
    CHECK_COMPOUND_LITERAL(FUNC_NAME, unsigned short, 0, VAL##BU);             \
    CHECK_COMPOUND_LITERAL(FUNC_NAME, unsigned int, 0, VAL##CU);               \
    CHECK_COMPOUND_LITERAL(FUNC_NAME, unsigned long, 0, VAL##DU);              \
    CHECK_COMPOUND_LITERAL(FUNC_NAME, unsigned long long, 0, VAL##EU);         \
  } while (0)

TEST(stdbit) {
  CHECK_FUNCTION(stdc_leading_zeros, 0xA);
  CHECK_FUNCTION(stdc_leading_ones, 0xB);
  CHECK_FUNCTION(stdc_trailing_zeros, 0xC);
  CHECK_FUNCTION(stdc_trailing_ones, 0xD);
  CHECK_FUNCTION(stdc_first_leading_zero, 0xE);
  CHECK_FUNCTION(stdc_first_leading_one, 0xF);
  CHECK_FUNCTION(stdc_first_trailing_zero, 0x0);
  CHECK_FUNCTION(stdc_first_trailing_one, 0x1);
  CHECK_FUNCTION(stdc_count_zeros, 0x2);
  CHECK_FUNCTION(stdc_count_ones, 0x3);

  EXPECT_FALSE(stdc_has_single_bit((unsigned char)1U));
  EXPECT_FALSE(stdc_has_single_bit((unsigned short)1U));
  EXPECT_FALSE(stdc_has_single_bit(1U));
  EXPECT_FALSE(stdc_has_single_bit(1UL));
  EXPECT_FALSE(stdc_has_single_bit(1ULL));
  CHECK_COMPOUND_LITERAL(stdc_has_single_bit, unsigned char, 0xAU, true);
  CHECK_COMPOUND_LITERAL(stdc_has_single_bit, unsigned short, 0xBU, true);
  CHECK_COMPOUND_LITERAL(stdc_has_single_bit, unsigned int, 0xCU, true);
  CHECK_COMPOUND_LITERAL(stdc_has_single_bit, unsigned long, 0xDU, true);
  CHECK_COMPOUND_LITERAL(stdc_has_single_bit, unsigned long long, 0xEU, true);

  CHECK_FUNCTION(stdc_bit_width, 0x4);
  CHECK_FUNCTION(stdc_bit_floor, 0x5);
  CHECK_FUNCTION(stdc_bit_ceil, 0x6);
}
