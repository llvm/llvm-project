/* Multiple versions of memset.
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
/* Redefine memset so that the compiler won't complain about the type
   mismatch with the IFUNC selector in strong_alias, below.  */
# undef memset
# define memset __redirect_memset
# include <string.h>
# include <shlib-compat.h>
# include "init-arch.h"

extern __typeof (__redirect_memset) __libc_memset;

extern __typeof (__redirect_memset) __memset_ppc attribute_hidden;
extern __typeof (__redirect_memset) __memset_power4 attribute_hidden;
extern __typeof (__redirect_memset) __memset_power6 attribute_hidden;
extern __typeof (__redirect_memset) __memset_power7 attribute_hidden;
extern __typeof (__redirect_memset) __memset_power8 attribute_hidden;
# ifdef __LITTLE_ENDIAN__
extern __typeof (__redirect_memset) __memset_power10 attribute_hidden;
# endif

/* Avoid DWARF definition DIE on ifunc symbol so that GDB can handle
   ifunc symbol properly.  */
libc_ifunc (__libc_memset,
# ifdef __LITTLE_ENDIAN__
	    (hwcap2 & PPC_FEATURE2_ARCH_3_1
	     && hwcap2 & PPC_FEATURE2_HAS_ISEL
	     && hwcap & PPC_FEATURE_HAS_VSX)
	    ? __memset_power10 :
# endif
            (hwcap2 & PPC_FEATURE2_ARCH_2_07)
            ? __memset_power8 :
	      (hwcap & PPC_FEATURE_HAS_VSX)
	      ? __memset_power7 :
		(hwcap & PPC_FEATURE_ARCH_2_05)
		? __memset_power6 :
		  (hwcap & PPC_FEATURE_POWER4)
		  ? __memset_power4
            : __memset_ppc);

#undef memset
strong_alias (__libc_memset, memset);
libc_hidden_ver (__libc_memset, memset);
#endif
