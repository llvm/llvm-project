/* Multiple versions of __floorf.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#define NO_MATH_REDIRECT
#include <libm-alias-float.h>

#define floorf __redirect_floorf
#define __floorf __redirect___floorf
#include <math.h>
#undef floorf
#undef __floorf

#define SYMBOL_NAME floorf
#include "ifunc-sse4_1.h"

libc_ifunc_redirected (__redirect_floorf, __floorf, IFUNC_SELECTOR ());
libm_alias_float (__floor, floor)
