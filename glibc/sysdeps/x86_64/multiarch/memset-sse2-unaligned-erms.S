/* memset with SSE2.
   All versions must be listed in ifunc-impl-list.c.
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

#include <sysdep.h>
#include <shlib-compat.h>

#if IS_IN (libc)
# define MEMSET_SYMBOL(p,s)	p##_sse2_##s
# define WMEMSET_SYMBOL(p,s)	p##_sse2_##s

# ifdef SHARED
#  undef libc_hidden_builtin_def
#  define libc_hidden_builtin_def(name)
# endif

# undef weak_alias
# define weak_alias(original, alias) \
	.weak bzero; bzero = __bzero

# undef strong_alias
# define strong_alias(ignored1, ignored2)
#endif

#include <sysdeps/x86_64/memset.S>
