/* Multiple versions of bzero.
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
extern __typeof (bzero) __bzero_power6 attribute_hidden;
extern __typeof (bzero) __bzero_power7 attribute_hidden;

libc_ifunc (__bzero,
            (hwcap & PPC_FEATURE_HAS_VSX)
            ? __bzero_power7 :
	      (hwcap & PPC_FEATURE_ARCH_2_05)
		? __bzero_power6
            : __bzero_ppc);

weak_alias (__bzero, bzero)
#endif
