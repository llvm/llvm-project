/* Multiple versions of strncpy.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/ >.  */

/* Define multiple versions only for definition in libc. */
#if IS_IN (libc)
# define strncpy __redirect_strncpy
/* Omit the strncpy inline definitions because it would redefine strncpy.  */
# define __NO_STRING_INLINES
# include <string.h>
# include <shlib-compat.h>
# include "init-arch.h"

extern __typeof (strncpy) __strncpy_ppc attribute_hidden;
extern __typeof (strncpy) __strncpy_power7 attribute_hidden;
extern __typeof (strncpy) __strncpy_power8 attribute_hidden;
# ifdef __LITTLE_ENDIAN__
extern __typeof (strncpy) __strncpy_power9 attribute_hidden;
# endif
# undef strncpy

/* Avoid DWARF definition DIE on ifunc symbol so that GDB can handle
 ifunc symbol properly. */
libc_ifunc_redirected (__redirect_strncpy, strncpy,
# ifdef __LITTLE_ENDIAN__
		       (hwcap2 & PPC_FEATURE2_ARCH_3_00) &&
		       (hwcap & PPC_FEATURE_HAS_VSX)
		       ? __strncpy_power9 :
# endif
		       (hwcap2 & PPC_FEATURE2_ARCH_2_07)
		       ? __strncpy_power8
		       : (hwcap & PPC_FEATURE_HAS_VSX)
			 ? __strncpy_power7
			 : __strncpy_ppc);

#endif
