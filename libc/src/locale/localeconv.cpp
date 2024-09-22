//===-- Implementation of localeconv --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/locale/localeconv.h"

#include "src/__support/CPP/limits.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

static char DOT_STRING[] = ".";
static char EMPTY_STRING[] = "";

static struct lconv C_LCONV = {
    .decimal_point = DOT_STRING,
    .thousands_sep = EMPTY_STRING,
    .grouping = EMPTY_STRING,
    .mon_decimal_point = EMPTY_STRING,
    .mon_thousands_sep = EMPTY_STRING,
    .mon_grouping = EMPTY_STRING,
    .positive_sign = EMPTY_STRING,
    .negative_sign = EMPTY_STRING,
    .currency_symbol = EMPTY_STRING,
    .frac_digits = CHAR_MAX,
    .p_cs_precedes = CHAR_MAX,
    .n_cs_precedes = CHAR_MAX,
    .p_sep_by_space = CHAR_MAX,
    .n_sep_by_space = CHAR_MAX,
    .p_sign_posn = CHAR_MAX,
    .n_sign_posn = CHAR_MAX,
    .int_curr_symbol = EMPTY_STRING,
    .int_frac_digits = CHAR_MAX,
    .int_p_cs_precedes = CHAR_MAX,
    .int_n_cs_precedes = CHAR_MAX,
    .int_p_sep_by_space = CHAR_MAX,
    .int_n_sep_by_space = CHAR_MAX,
    .int_p_sign_posn = CHAR_MAX,
    .int_n_sign_posn = CHAR_MAX,
};

LLVM_LIBC_FUNCTION(struct lconv *, localeconv, ()) { return &C_LCONV; }

} // namespace LIBC_NAMESPACE_DECL
