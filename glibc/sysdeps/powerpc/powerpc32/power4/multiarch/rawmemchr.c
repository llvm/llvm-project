/* Multiple versions of rawmemchr.
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
# define __rawmemchr __redirect___rawmemchr
# include <string.h>
# include <shlib-compat.h>
# include "init-arch.h"

extern __typeof (__rawmemchr) __rawmemchr_ppc attribute_hidden;
extern __typeof (__rawmemchr) __rawmemchr_power7 attribute_hidden;
# undef __rawmemchr

/* Avoid DWARF definition DIE on ifunc symbol so that GDB can handle
   ifunc symbol properly.  */
libc_ifunc_redirected (__redirect___rawmemchr, __rawmemchr,
		       (hwcap & PPC_FEATURE_HAS_VSX)
		       ? __rawmemchr_power7
		       : __rawmemchr_ppc);
weak_alias (__rawmemchr, rawmemchr)
#else
#include <string/rawmemchr.c>
#endif
