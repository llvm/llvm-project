/* Print floating point number in hexadecimal notation according to
   ISO C99.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#include <ldbl-128/printf_fphex_macros.h>
#define PRINT_FPHEX_LONG_DOUBLE \
  PRINT_FPHEX (long double, fpnum.ldbl, ieee854_long_double, \
	       IEEE854_LONG_DOUBLE_BIAS)

#include <stdio-common/printf_fphex.c>
