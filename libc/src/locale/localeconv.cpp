//===-- Implementation of localeconv --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "localeconv.h"
#include "src/__support/common.h"
#include <limits.h>

namespace LIBC_NAMESPACE {

static const struct lconv posix_lconv = {
    .currency_symbol = const_cast<char *>(""),
    .decimal_point = const_cast<char *>("."),
    .grouping = const_cast<char *>(""),
    .int_curr_symbol = const_cast<char *>(""),
    .mon_decimal_point = const_cast<char *>(""),
    .mon_grouping = const_cast<char *>(""),
    .mon_thousands_sep = const_cast<char *>(""),
    .negative_sign = const_cast<char *>(""),
    .positive_sign = const_cast<char *>(""),
    .thousands_sep = const_cast<char *>(""),
    .frac_digits = CHAR_MAX,
    .int_frac_digits = CHAR_MAX,
    .int_p_cs_precedes = CHAR_MAX,
    .int_p_sep_by_space = CHAR_MAX,
    .int_n_cs_precedes = CHAR_MAX,
    .int_n_sep_by_space = CHAR_MAX,
    .int_n_sign_posn = CHAR_MAX,
    .int_p_sign_posn = CHAR_MAX,
    .n_cs_precedes = CHAR_MAX,
    .n_sep_by_space = CHAR_MAX,
    .n_sign_posn = CHAR_MAX,
    .p_cs_precedes = CHAR_MAX,
    .p_sep_by_space = CHAR_MAX,
    .p_sign_posn = CHAR_MAX,
};

LLVM_LIBC_FUNCTION(struct lconv *, localeconv, ()) {
  return const_cast<struct lconv *>(&posix_lconv);
}

} // namespace LIBC_NAMESPACE
