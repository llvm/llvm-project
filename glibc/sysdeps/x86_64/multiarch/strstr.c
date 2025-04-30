/* Multiple versions of strstr.
   All versions must be listed in ifunc-impl-list.c.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

/* Redefine strstr so that the compiler won't complain about the type
   mismatch with the IFUNC selector in strong_alias, below.  */
#undef  strstr
#define strstr __redirect_strstr
#include <string.h>
#undef  strstr

#define STRSTR __strstr_sse2
#ifdef SHARED
# undef libc_hidden_builtin_def
# define libc_hidden_builtin_def(name) \
  __hidden_ver1 (__strstr_sse2, __GI_strstr, __strstr_sse2);
#endif

#include "string/strstr.c"

extern __typeof (__redirect_strstr) __strstr_sse2_unaligned attribute_hidden;
extern __typeof (__redirect_strstr) __strstr_sse2 attribute_hidden;

#include "init-arch.h"

/* Avoid DWARF definition DIE on ifunc symbol so that GDB can handle
   ifunc symbol properly.  */
extern __typeof (__redirect_strstr) __libc_strstr;
libc_ifunc (__libc_strstr,
	    HAS_ARCH_FEATURE (Fast_Unaligned_Load)
	    ? __strstr_sse2_unaligned
	    : __strstr_sse2)

#undef strstr
strong_alias (__libc_strstr, strstr)
