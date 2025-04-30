/* Multiple versions of bzero. PowerPC64 version.
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
#if IS_IN (libc)
# include <string.h>
# include <strings.h>
# include "init-arch.h"

extern __typeof (bzero) __bzero_ppc attribute_hidden;
extern __typeof (bzero) __bzero_power4 attribute_hidden;
extern __typeof (bzero) __bzero_power6 attribute_hidden;
extern __typeof (bzero) __bzero_power7 attribute_hidden;
extern __typeof (bzero) __bzero_power8 attribute_hidden;
# ifdef __LITTLE_ENDIAN__
extern __typeof (bzero) __bzero_power10 attribute_hidden;
# endif

libc_ifunc (__bzero,
# ifdef __LITTLE_ENDIAN__
	    (hwcap2 & PPC_FEATURE2_ARCH_3_1
	     && hwcap2 & PPC_FEATURE2_HAS_ISEL
	     && hwcap & PPC_FEATURE_HAS_VSX)
	    ? __bzero_power10 :
# endif
            (hwcap2 & PPC_FEATURE2_ARCH_2_07)
            ? __bzero_power8 :
	      (hwcap & PPC_FEATURE_HAS_VSX)
	      ? __bzero_power7 :
		(hwcap & PPC_FEATURE_ARCH_2_05)
		? __bzero_power6 :
		  (hwcap & PPC_FEATURE_POWER4)
		  ? __bzero_power4
            : __bzero_ppc);

weak_alias (__bzero, bzero)
#endif
