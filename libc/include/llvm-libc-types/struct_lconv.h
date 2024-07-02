//===-- Definition of type struct lconv -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_STRUCT_LCONV_H
#define LLVM_LIBC_TYPES_STRUCT_LCONV_H

struct lconv {
  char *currency_symbol;
	char *decimal_point;
  char *grouping;
  char *int_curr_symbol;
  char *mon_decimal_point;
  char *mon_grouping;
	char *mon_thousands_sep;
  char *negative_sign;
  char *positive_sign;
	char *thousands_sep;
  char frac_digits;
  char int_frac_digits;
  char int_p_cs_precedes;
	char int_p_sep_by_space;
	char int_n_cs_precedes;
	char int_n_sep_by_space;
  char int_n_sign_posn;
	char int_p_sign_posn;
  char n_cs_precedes;
	char n_sep_by_space;
  char n_sign_posn;
	char p_cs_precedes;
	char p_sep_by_space;
	char p_sign_posn;	
};

#endif // LLVM_LIBC_TYPES_STRUCT_LCONV_H
