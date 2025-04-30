/* Multiple versions of strspn. PowerPC64 version.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

# include <string.h>
# include <shlib-compat.h>
# include "init-arch.h"

#undef strspn
extern __typeof (strspn) __libc_strspn;

extern __typeof (strspn) __strspn_ppc attribute_hidden;
extern __typeof (strspn) __strspn_power8 attribute_hidden;

libc_ifunc (__libc_strspn,
	    (hwcap2 & PPC_FEATURE2_ARCH_2_07)
	    ? __strspn_power8
	    : __strspn_ppc);

weak_alias (__libc_strspn, strspn)
libc_hidden_builtin_def (strspn)
