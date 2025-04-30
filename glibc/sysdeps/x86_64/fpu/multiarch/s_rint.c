/* Multiple versions of __rint.
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
#include <libm-alias-double.h>

#define rint __redirect_rint
#define __rint __redirect___rint
#include <math.h>
#undef rint
#undef __rint

#define SYMBOL_NAME rint
#include "ifunc-sse4_1.h"

libc_ifunc_redirected (__redirect_rint, __rint, IFUNC_SELECTOR ());
libm_alias_double (__rint, rint)
