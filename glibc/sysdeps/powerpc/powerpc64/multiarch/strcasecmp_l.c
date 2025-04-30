/* Multiple versions of strcasecmp_l.
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
# define strcasecmp_l __strcasecmp_l_ppc
extern __typeof (__strcasecmp_l) __strcasecmp_l_ppc attribute_hidden;
extern __typeof (__strcasecmp_l) __strcasecmp_l_power7 attribute_hidden;
#endif

#include <string/strcasecmp_l.c>
#undef strcasecmp_l

#if IS_IN (libc)
# include <shlib-compat.h>
# include "init-arch.h"

extern __typeof (__strcasecmp_l) __libc_strcasecmp_l;
libc_ifunc (__libc_strcasecmp_l,
	    (hwcap & PPC_FEATURE_HAS_VSX)
            ? __strcasecmp_l_power7
            : __strcasecmp_l_ppc);

weak_alias (__libc_strcasecmp_l, strcasecmp_l)
#endif
