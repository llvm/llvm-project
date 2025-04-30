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
#include <shlib-compat.h>

#define glob64 __no_glob64_decl
#include <glob.h>
#undef glob64

#define __glob __glob_lstat_compat

#define GLOB_ATTRIBUTE attribute_compat_text_section

/* Avoid calling gl_lstat with GLOB_ALTDIRFUNC.  */
#define struct_stat    struct stat
#define struct_stat64  struct stat64
#define GLOB_LSTAT     gl_stat
#define GLOB_STAT64    __stat64
#define GLOB_LSTAT64   __stat64

#include <posix/glob.c>

#ifndef GLOB_LSTAT_VERSION
# define GLOB_LSTAT_VERSION GLIBC_2_0
#endif

#if SHLIB_COMPAT(libc, GLOB_LSTAT_VERSION, GLIBC_2_27)
compat_symbol (libc, __glob_lstat_compat, glob, GLOB_LSTAT_VERSION);
# if XSTAT_IS_XSTAT64
strong_alias (__glob_lstat_compat, __glob64_lstat_compat)
compat_symbol (libc, __glob64_lstat_compat, glob64, GLOB_LSTAT_VERSION);
# endif
#endif
