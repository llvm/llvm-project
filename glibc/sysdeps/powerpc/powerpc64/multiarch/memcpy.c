/* Multiple versions of memcpy. PowerPC64 version.
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

/* Define multiple versions only for the definition in lib and for
   DSO.  In static binaries we need memcpy before the initialization
   happened.  */
#if defined SHARED && IS_IN (libc)
/* Redefine memcpy so that the compiler won't complain about the type
   mismatch with the IFUNC selector in strong_alias, below.  */
# undef memcpy
# define memcpy __redirect_memcpy
# include <string.h>
# include "init-arch.h"

extern __typeof (__redirect_memcpy) __libc_memcpy;

extern __typeof (__redirect_memcpy) __memcpy_ppc attribute_hidden;
extern __typeof (__redirect_memcpy) __memcpy_power4 attribute_hidden;
extern __typeof (__redirect_memcpy) __memcpy_cell attribute_hidden;
extern __typeof (__redirect_memcpy) __memcpy_power6 attribute_hidden;
extern __typeof (__redirect_memcpy) __memcpy_a2 attribute_hidden;
extern __typeof (__redirect_memcpy) __memcpy_power7 attribute_hidden;
extern __typeof (__redirect_memcpy) __memcpy_power8_cached attribute_hidden;
# if defined __LITTLE_ENDIAN__
extern __typeof (__redirect_memcpy) __memcpy_power10 attribute_hidden;
# endif

libc_ifunc (__libc_memcpy,
# if defined __LITTLE_ENDIAN__
	    (hwcap2 & PPC_FEATURE2_ARCH_3_1 && hwcap & PPC_FEATURE_HAS_VSX)
	    ? __memcpy_power10 :
# endif
	    ((hwcap2 & PPC_FEATURE2_ARCH_2_07) && use_cached_memopt)
	    ? __memcpy_power8_cached :
	      (hwcap & PPC_FEATURE_HAS_VSX)
	      ? __memcpy_power7 :
		(hwcap & PPC_FEATURE_ARCH_2_06)
		? __memcpy_a2 :
		  (hwcap & PPC_FEATURE_ARCH_2_05)
		  ? __memcpy_power6 :
		    (hwcap & PPC_FEATURE_CELL_BE)
		    ? __memcpy_cell :
		      (hwcap & PPC_FEATURE_POWER4)
		      ? __memcpy_power4
            : __memcpy_ppc);

#undef memcpy
strong_alias (__libc_memcpy, memcpy);
libc_hidden_ver (__libc_memcpy, memcpy);
#endif
