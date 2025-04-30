/* Double versions of *cvt_r functions.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#define ECVT ecvt
#define FCVT fcvt
#define GCVT gcvt
#define __ECVT __ecvt
#define __FCVT __fcvt
#define __GCVT __gcvt
#define __ECVT_R __ecvt_r
#define __FCVT_R __fcvt_r
#include <efgcvt-dbl-macros.h>
#include <efgcvt-template.c>

#if LONG_DOUBLE_COMPAT (libc, GLIBC_2_0)
# define cvt_symbol(local, symbol) \
  cvt_symbol_1 (libc, local, APPEND (q, symbol), GLIBC_2_0); \
  strong_alias (local, symbol)
# define cvt_symbol_1(lib, local, symbol, version) \
  compat_symbol (lib, local, symbol, version)
#else
# define cvt_symbol(local, symbol) \
  strong_alias (local, symbol)
#endif
cvt_symbol (__fcvt, fcvt);
cvt_symbol (__ecvt, ecvt);
cvt_symbol (__gcvt, gcvt);
