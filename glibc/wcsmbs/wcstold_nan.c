/* Convert string for NaN payload to corresponding NaN.  Wide strings,
   long double.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#include <math.h>

/* This function is unused if long double and double have the same
   representation.  */
#ifndef __NO_LONG_DOUBLE_MATH
# include "../stdlib/strtod_nan_wide.h"
# include <math-type-macros-ldouble.h>

# define STRTOD_NAN __wcstold_nan
# include "../stdlib/strtod_nan_main.c"
#endif
