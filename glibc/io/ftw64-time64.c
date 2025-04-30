/* File tree walker functions.  LFS version.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.

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

#include <sys/types.h>

#if __TIMESIZE != 64
# define FTW_NAME       __ftw64_time64
# define NFTW_NAME      __nftw64_time64
# define INO_T          ino64_t
# define STRUCT_STAT    __stat64_t64
# define LSTAT          __lstat64_time64
# define STAT           __stat64_time64
# define FSTATAT        __fstatat64_time64
# define FTW_FUNC_T     __ftw64_time64_func_t
# define NFTW_FUNC_T    __nftw64_time64_func_t

# include "ftw.c"
#endif
