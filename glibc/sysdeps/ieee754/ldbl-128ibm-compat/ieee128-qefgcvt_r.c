/* IEEE binary128 versions of reentrant *cvt_r functions.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

/* When in IEEE long double mode, call ___ieee128_snprintf.  */
#include <stdio.h>
typeof (snprintf) ___ieee128_snprintf attribute_hidden;
#define SNPRINTF ___ieee128_snprintf

#define ECVT_R __qecvtieee128_r
#define FCVT_R __qfcvtieee128_r
#define __ECVT_R ___qecvtieee128_r
#define __FCVT_R ___qfcvtieee128_r
#include <efgcvt-ldbl-macros.h>
#include <efgcvt_r-template.c>

#define cvt_symbol(local, symbol) \
  strong_alias (local, symbol)
cvt_symbol (___qfcvtieee128_r, __qfcvtieee128_r);
cvt_symbol (___qecvtieee128_r, __qecvtieee128_r);
