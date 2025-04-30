/* Optimized strncpy implementation for POWER9 LE.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#if defined __LITTLE_ENDIAN__ && IS_IN (libc)
# define STRNCPY __strncpy_power9

# undef libc_hidden_builtin_def
# define libc_hidden_builtin_def(name)

/* memset is used to pad the end of the string.  */
# define MEMSET __memset_power8
# ifdef SHARED
#  define MEMSET_is_local
# endif

# include <sysdeps/powerpc/powerpc64/le/power9/strncpy.S>
#endif
