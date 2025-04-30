/* Compat glob which does not use gl_lstat for GLOB_ALTDIRFUNC.
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

#include <shlib-compat.h>

#if SHLIB_COMPAT(libc, GLIBC_2_0, GLIBC_2_27)

# include <glob.h>

# define glob(pattern, flags, errfunc, pglob) \
  __glob64_lstat_compat (pattern, flags, errfunc, pglob)

# define GLOB_ATTRIBUTE attribute_compat_text_section

/* Avoid calling gl_lstat with GLOB_ALTDIRFUNC.  */
# define GLOB_LSTAT   gl_stat
# define GLOB_LSTAT64 __stat64

# include <posix/glob64.c>

compat_symbol (libc, __glob64_lstat_compat, glob64, GLIBC_2_0);
#endif
