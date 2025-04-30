/* Multiple versions of memchr.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#if IS_IN (libc)
# undef memchr
/* Redefine memchr so that the compiler won't make the weak_alias point
   to internal hidden definition (__GI_memchr), since PPC32 does not
   support local IFUNC calls.  */
# define memchr __redirect_memchr
# include <string.h>
# include "init-arch.h"

extern __typeof (__redirect_memchr) __memchr_ppc attribute_hidden;
extern __typeof (__redirect_memchr) __memchr_power7 attribute_hidden;

extern __typeof (__redirect_memchr) __libc_memchr;

libc_ifunc (__libc_memchr,
	    (hwcap & PPC_FEATURE_HAS_VSX)
            ? __memchr_power7
            : __memchr_ppc);
#undef memchr
weak_alias (__libc_memchr, memchr)
#else
#include <string/memchr.c>
#endif
