/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <locale.h>
#include "localeinfo.h"
#include <shlib-compat.h>

/* Return monetary and numeric information about the current locale.  */
struct lconv *
__localeconv (void)
{
  static struct lconv result;

  result.decimal_point = (char *) _NL_CURRENT (LC_NUMERIC, DECIMAL_POINT);
  result.thousands_sep = (char *) _NL_CURRENT (LC_NUMERIC, THOUSANDS_SEP);
  result.grouping = (char *) _NL_CURRENT (LC_NUMERIC, GROUPING);
  if (*result.grouping == '\177' || *result.grouping == '\377')
    result.grouping = (char *) "";

  result.int_curr_symbol = (char *) _NL_CURRENT (LC_MONETARY, INT_CURR_SYMBOL);
  result.currency_symbol = (char *) _NL_CURRENT (LC_MONETARY, CURRENCY_SYMBOL);
  result.mon_decimal_point = (char *) _NL_CURRENT (LC_MONETARY,
						   MON_DECIMAL_POINT);
  result.mon_thousands_sep = (char *) _NL_CURRENT (LC_MONETARY,
						   MON_THOUSANDS_SEP);
  result.mon_grouping = (char *) _NL_CURRENT (LC_MONETARY, MON_GROUPING);
  if (*result.mon_grouping == '\177' || *result.mon_grouping == '\377')
    result.mon_grouping = (char *) "";
  result.positive_sign = (char *) _NL_CURRENT (LC_MONETARY, POSITIVE_SIGN);
  result.negative_sign = (char *) _NL_CURRENT (LC_MONETARY, NEGATIVE_SIGN);

#define INT_ELEM(member, element) \
  result.member = *(char *) _NL_CURRENT (LC_MONETARY, element);		      \
  if (result.member == '\377') result.member = CHAR_MAX

  INT_ELEM (int_frac_digits, INT_FRAC_DIGITS);
  INT_ELEM (frac_digits, FRAC_DIGITS);
  INT_ELEM (p_cs_precedes, P_CS_PRECEDES);
  INT_ELEM (p_sep_by_space, P_SEP_BY_SPACE);
  INT_ELEM (n_cs_precedes, N_CS_PRECEDES);
  INT_ELEM (n_sep_by_space, N_SEP_BY_SPACE);
  INT_ELEM (p_sign_posn, P_SIGN_POSN);
  INT_ELEM (n_sign_posn, N_SIGN_POSN);
  INT_ELEM (int_p_cs_precedes, INT_P_CS_PRECEDES);
  INT_ELEM (int_p_sep_by_space, INT_P_SEP_BY_SPACE);
  INT_ELEM (int_n_cs_precedes, INT_N_CS_PRECEDES);
  INT_ELEM (int_n_sep_by_space, INT_N_SEP_BY_SPACE);
  INT_ELEM (int_p_sign_posn, INT_P_SIGN_POSN);
  INT_ELEM (int_n_sign_posn, INT_N_SIGN_POSN);

  return &result;
}

versioned_symbol (libc, __localeconv, localeconv, GLIBC_2_2);
#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_2)
strong_alias (__localeconv, __localeconv20)
compat_symbol (libc, __localeconv20, localeconv, GLIBC_2_0);
#endif
