/* Memchr for aarch64, default version for internal use.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#if IS_IN (libc)
# define MEMCHR __memchr_generic

/* Do not hide the generic version of memchr, we use it internally.  */
# undef libc_hidden_builtin_def
# define libc_hidden_builtin_def(name)

/* Add a hidden definition for use within libc.so.  */
# ifdef SHARED
	.globl __GI_memchr; __GI_memchr = __memchr_generic
# endif
#endif

# include "../memchr.S"
