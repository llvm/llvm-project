/* Multiple versions of strncmp.
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

/* Define multiple versions only for definition in libc.  */
#if defined SHARED && IS_IN (libc)
# define strncmp __redirect_strncmp
/* Omit the strncmp inline definitions because it would redefine strncmp.  */
# define __NO_STRING_INLINES
# include <string.h>
# include <shlib-compat.h>
# include "init-arch.h"

extern __typeof (strncmp) __strncmp_ppc attribute_hidden;
extern __typeof (strncmp) __strncmp_power4 attribute_hidden;
extern __typeof (strncmp) __strncmp_power7 attribute_hidden;
# undef strncmp

/* Avoid DWARF definition DIE on ifunc symbol so that GDB can handle
   ifunc symbol properly.  */
libc_ifunc_redirected (__redirect_strncmp, strncmp,
		       (hwcap & PPC_FEATURE_HAS_VSX)
		       ? __strncmp_power7
		       : __strncmp_ppc);
#endif
