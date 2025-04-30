/* A Generic Optimized memcpy implementation for AARCH64.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

/* The actual memcpy and memmove code is in ../memcpy.S.  If we are
   building libc this file defines __memcpy_generic and __memmove_generic.
   Otherwise the include of ../memcpy.S will define the normal __memcpy
   and__memmove entry points.  */

#include <sysdep.h>

#if IS_IN (libc)

# define MEMCPY __memcpy_generic
# define MEMMOVE __memmove_generic

/* Do not hide the generic versions of memcpy and memmove, we use them
   internally.  */
# undef libc_hidden_builtin_def
# define libc_hidden_builtin_def(name)

# ifdef SHARED
/* It doesn't make sense to send libc-internal memcpy calls through a PLT. */
	.globl __GI_memcpy; __GI_memcpy = __memcpy_generic
	.globl __GI_memmove; __GI_memmove = __memmove_generic
# endif

#endif

#include "../memcpy.S"
