/* Compat glob which does not use gl_lstat for GLOB_ALTDIRFUNC.
   Linux version which handles LFS when required.
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

#include <sys/stat.h>
#include <kernel_stat.h>

#if !XSTAT_IS_XSTAT64
# include <glob.h>
# include <dirent.h>
# include <sys/stat.h>
# include <shlib-compat.h>

# define dirent dirent64
# define __readdir(dirp) __readdir64 (dirp)

# define glob_t glob64_t
# define __glob __glob64_lstat_compat
# define globfree globfree64

# define GLOB_ATTRIBUTE attribute_compat_text_section

/* Avoid calling gl_lstat with GLOB_ALTDIRFUNC.  */
# define COMPILE_GLOB64	1
# define struct_stat    struct stat64
# define struct_stat64  struct stat64
# define GLOB_LSTAT     gl_stat
# define GLOB_STAT64    __stat64
# define GLOB_LSTAT64   __stat64

# include <posix/glob.c>

# if SHLIB_COMPAT(libc, GLIBC_2_2, GLIBC_2_27)
#  ifndef GLOB_NO_OLD_VERSION
#   define GLOB_LSTAT_START_VER GLIBC_2_2
#  else
#   define GLOB_LSTAT_START_VER GLIBC_2_1
#  endif
compat_symbol (libc, __glob64_lstat_compat, glob64, GLOB_LSTAT_START_VER);
# endif
#endif /* XSTAT_IS_XSTAT64  */
