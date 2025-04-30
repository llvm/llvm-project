/* Tests for *cvt function, double version.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
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

#include <float.h>

#define ECVT ecvt
#define FCVT fcvt
#define ECVT_R ecvt_r
#define FCVT_R fcvt_r
#define FLOAT double
#define PRINTF_CONVERSION "%f"

#if DBL_MANT_DIG == 53
# define EXTRA_ECVT_TESTS \
  { 0x1p-1074, 3, -323, "494" }, \
  { -0x1p-1074, 3, -323, "494" },
#else
# define EXTRA_ECVT_TESTS
#endif

#include <tst-efgcvt-template.c>
