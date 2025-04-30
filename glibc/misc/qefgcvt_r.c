/* Compatibility functions for floating point formatting, reentrant,
   long double versions.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

#define ECVT_R qecvt_r
#define FCVT_R qfcvt_r
#define __ECVT_R __qecvt_r
#define __FCVT_R __qfcvt_r
#include <efgcvt-ldbl-macros.h>
#include <efgcvt_r-template.c>

#if LONG_DOUBLE_COMPAT (libc, GLIBC_2_0)
# define cvt_symbol(local, symbol) \
  libc_hidden_def (local) \
  versioned_symbol (libc, local, symbol, GLIBC_2_4)
#else
# define cvt_symbol(local, symbol) \
  libc_hidden_def (local) \
  weak_alias (local, symbol)
#endif
cvt_symbol (__qfcvt_r, qfcvt_r);
cvt_symbol (__qecvt_r, qecvt_r);
