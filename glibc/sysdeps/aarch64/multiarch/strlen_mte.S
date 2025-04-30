/* A Generic Optimized strlen implementation for AARCH64.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

/* The actual strlen code is in ../strlen.S.  If we are building libc this file
   defines __strlen_mte.  Otherwise the include of ../strlen.S will define
   the normal __strlen  entry points.  */

#include <sysdep.h>

#if IS_IN (libc)

# define STRLEN __strlen_mte

/* Do not hide the generic version of strlen, we use it internally.  */
# undef libc_hidden_builtin_def
# define libc_hidden_builtin_def(name)

# ifdef SHARED
/* It doesn't make sense to send libc-internal strlen calls through a PLT. */
	.globl __GI_strlen; __GI_strlen = __strlen_mte
# endif
#endif

#include "../strlen.S"
