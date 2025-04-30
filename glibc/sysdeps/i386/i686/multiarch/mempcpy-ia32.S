/* mempcpy optimized for i686.
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

#if defined SHARED && IS_IN (libc)
# define __mempcpy __mempcpy_ia32
# define __mempcpy_chk __mempcpy_chk_ia32

# undef libc_hidden_def
# undef libc_hidden_builtin_def
/* IFUNC doesn't work with the hidden functions in shared library since
   they will be called without setting up EBX needed for PLT which is
   used by IFUNC.  */
# define libc_hidden_def(name) \
	.globl __GI___mempcpy; __GI___mempcpy = __mempcpy
# define libc_hidden_builtin_def(name) \
	.globl __GI_mempcpy; __GI_mempcpy = __mempcpy

# undef weak_alias
# define weak_alias(original, alias)
#endif

#include <sysdeps/i386/i686/mempcpy.S>
