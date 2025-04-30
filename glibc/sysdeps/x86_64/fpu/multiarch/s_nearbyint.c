/* Multiple versions of __nearbyint.
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

#include <libm-alias-double.h>

#define nearbyint __redirect_nearbyint
#define __nearbyint __redirect___nearbyint
#include <math.h>
#undef nearbyint
#undef __nearbyint

#define SYMBOL_NAME nearbyint
#include "ifunc-sse4_1.h"

libc_ifunc_redirected (__redirect_nearbyint, __nearbyint,
		       IFUNC_SELECTOR ());
libm_alias_double (__nearbyint, nearbyint)
