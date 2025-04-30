/* Find pathnames matching a pattern.  Linux version.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#if __TIMESIZE != 64
# include <glob.h>
# include <dirent.h>
# include <sys/stat.h>

# define dirent dirent64
# define __readdir(dirp) __readdir64 (dirp)

# define glob_t glob64_time64_t
# define __glob __glob64_time64

# define globfree(pglob) __globfree64_time64 (pglob)

# define COMPILE_GLOB64  1
# define struct_stat     struct __stat64_t64
# define struct_stat64   struct __stat64_t64
# define GLOB_LSTAT      gl_lstat
# define GLOB_STAT64     __stat64_time64
# define GLOB_LSTAT64    __lstat64_time64

# define COMPILE_GLOB64	1

# include <posix/glob.c>
libc_hidden_def (__glob64_time64)
#endif
