/* IEEE binary128 versions of *cvt functions.
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

/* When in IEEE long double mode, call ___ieee128_sprintf.  */
#include <stdio.h>
typeof (sprintf) ___ieee128_sprintf attribute_hidden;
#define SPRINTF ___ieee128_sprintf

/* Declare internal functions: ___qecvtieee128_r and ___qfcvtieee128_r,
   built from a different compiling unit, and called from here.  */
#include <stdlib.h>
typeof (qecvt_r) ___qecvtieee128_r;
typeof (qfcvt_r) ___qfcvtieee128_r;

/* Rename the static buffers and pointer, otherwise the IEEE long double
   variants of qecvt and qfcvt would reuse the same buffers and pointer
   as their non-IEEE long double counterparts.  */
#define qecvt_buffer qecvtieee128_buffer
#define qfcvt_buffer qfcvtieee128_buffer
#define qfcvt_bufptr qfcvtieee128_bufptr

#define ECVT __qecvtieee128
#define FCVT __qfcvtieee128
#define GCVT __qgcvtieee128
#define __ECVT ___qecvtieee128
#define __FCVT ___qfcvtieee128
#define __GCVT ___qgcvtieee128
#define __ECVT_R ___qecvtieee128_r
#define __FCVT_R ___qfcvtieee128_r
#include <efgcvt-ldbl-macros.h>
#include <efgcvt-template.c>

#define cvt_symbol(local, symbol) \
  strong_alias (local, symbol)
cvt_symbol (___qfcvtieee128, __qfcvtieee128);
cvt_symbol (___qecvtieee128, __qecvtieee128);
cvt_symbol (___qgcvtieee128, __qgcvtieee128);
