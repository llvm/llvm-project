/* Find pathnames matching a pattern.  Linux version.
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

# define dirent dirent64
# define __readdir(dirp) __readdir64 (dirp)

# define glob_t glob64_t
# define __glob __glob64
# define globfree(pglob) globfree64 (pglob)

# define COMPILE_GLOB64	1
# define struct_stat    struct stat64
# define struct_stat64  struct stat64
# define GLOB_LSTAT     gl_lstat
# define GLOB_STAT64    __stat64
# define GLOB_LSTAT64   __lstat64

# include <posix/glob.c>

# include <shlib-compat.h>

# ifdef GLOB_NO_OLD_VERSION
strong_alias (__glob64, glob64)
libc_hidden_def (glob64)
# else
libc_hidden_def (__glob64)
versioned_symbol (libc, __glob64, glob64, GLIBC_2_27);
libc_hidden_ver (__glob64, glob64)
# endif
#endif /* XSTAT_IS_XSTAT64  */
