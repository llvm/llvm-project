/* Multiple versions of strlen. PowerPC64 version.
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

#if defined SHARED && IS_IN (libc)
/* Redefine strlen so that the compiler won't complain about the type
   mismatch with the IFUNC selector in strong_alias, below.  */
# undef strlen
# define strlen __redirect_strlen
# include <string.h>
# include <shlib-compat.h>
# include "init-arch.h"

extern __typeof (__redirect_strlen) __libc_strlen;

extern __typeof (__redirect_strlen) __strlen_ppc attribute_hidden;
extern __typeof (__redirect_strlen) __strlen_power7 attribute_hidden;
extern __typeof (__redirect_strlen) __strlen_power8 attribute_hidden;
extern __typeof (__redirect_strlen) __strlen_power9 attribute_hidden;
extern __typeof (__redirect_strlen) __strlen_power10 attribute_hidden;

libc_ifunc (__libc_strlen,
# ifdef __LITTLE_ENDIAN__
	(hwcap2 & PPC_FEATURE2_ARCH_3_1)
	? __strlen_power10 :
	  (hwcap2 & PPC_FEATURE2_ARCH_3_00)
	  ? __strlen_power9 :
# endif
	    (hwcap2 & PPC_FEATURE2_ARCH_2_07)
	    ? __strlen_power8 :
	      (hwcap & PPC_FEATURE_HAS_VSX)
	      ? __strlen_power7
	      : __strlen_ppc);

#undef strlen
strong_alias (__libc_strlen, strlen)
libc_hidden_ver (__libc_strlen, strlen)
#endif
