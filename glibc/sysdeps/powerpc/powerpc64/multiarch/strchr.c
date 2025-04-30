/* Multiple versions of strchr.
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
# define strchr __redirect_strchr
/* Omit the strchr inline definitions because it would redefine strchr.  */
# define __NO_STRING_INLINES
# include <string.h>
# include <shlib-compat.h>
# include "init-arch.h"

extern __typeof (strchr) __strchr_ppc attribute_hidden;
extern __typeof (strchr) __strchr_power7 attribute_hidden;
extern __typeof (strchr) __strchr_power8 attribute_hidden;
# undef strchr

/* Avoid DWARF definition DIE on ifunc symbol so that GDB can handle
   ifunc symbol properly.  */
libc_ifunc_redirected (__redirect_strchr, strchr,
		       (hwcap2 & PPC_FEATURE2_ARCH_2_07)
		       ? __strchr_power8 :
		       (hwcap & PPC_FEATURE_HAS_VSX)
		       ? __strchr_power7
		       : __strchr_ppc);
weak_alias (strchr, index)
#endif
