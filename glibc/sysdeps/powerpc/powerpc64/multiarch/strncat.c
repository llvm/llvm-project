/* Multiple versions of strncat. PowerPC64 version.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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
# include <shlib-compat.h>
# include "init-arch.h"

extern __typeof (strncat) __strncat_ppc attribute_hidden;
extern __typeof (strncat) __strncat_power7 attribute_hidden;
extern __typeof (strncat) __strncat_power8 attribute_hidden;

libc_ifunc (strncat,
	    (hwcap2 & PPC_FEATURE2_ARCH_2_07)
	    ? __strncat_power8
	    : (hwcap & PPC_FEATURE_HAS_VSX)
            ? __strncat_power7
            : __strncat_ppc);
#endif
