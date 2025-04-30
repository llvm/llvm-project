/* Multiple versions of memchr.
   All versions must be listed in ifunc-impl-list.c.
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

/* For __ARM_NEON__ memchr_neon.S defines memchr directly and ifunc
   is not used.  */
#if IS_IN (libc) && !defined (__ARM_NEON__)
# define memchr __redirect_memchr
# include <string.h>
# undef memchr

# include <arm-ifunc.h>

# define SYMBOL_NAME memchr
# include "ifunc-memchr.h"

arm_libc_ifunc_redirected (__redirect_memchr, memchr, IFUNC_SELECTOR);

arm_libc_ifunc_hidden_def (__redirect_memchr, memchr);
#endif
