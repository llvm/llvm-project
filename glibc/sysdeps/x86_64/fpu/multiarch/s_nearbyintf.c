/* Multiple versions of __nearbyintf.
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

#include <libm-alias-float.h>

#define nearbyintf __redirect_nearbyintf
#define __nearbyintf __redirect___nearbyintf
#include <math.h>
#undef nearbyintf
#undef __nearbyintf

#define SYMBOL_NAME nearbyintf
#include "ifunc-sse4_1.h"

libc_ifunc_redirected (__redirect_nearbyintf, __nearbyintf,
		       IFUNC_SELECTOR ());
libm_alias_float (__nearbyint, nearbyint)
