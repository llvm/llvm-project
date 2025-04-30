/* Multiple versions of strncasecmp.
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
# include <string.h>
# define strncasecmp __strncasecmp_ppc
extern __typeof (__strncasecmp) __strncasecmp_ppc attribute_hidden;
extern __typeof (__strncasecmp) __strncasecmp_power7 attribute_hidden;
#endif

#include <string/strncase.c>
#undef strncasecmp

#if IS_IN (libc)
# include <shlib-compat.h>
# include "init-arch.h"

/* Avoid DWARF definition DIE on ifunc symbol so that GDB can handle
   ifunc symbol properly.  */
extern __typeof (__strncasecmp) __libc_strncasecmp;
libc_ifunc (__libc_strncasecmp,
	     (hwcap & PPC_FEATURE_HAS_VSX)
             ? __strncasecmp_power7
             : __strncasecmp_ppc);
weak_alias (__libc_strncasecmp, strncasecmp)
#endif
