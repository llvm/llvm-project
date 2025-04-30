/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <stdlib.h>
#include <unistd.h>

#include <shlib-compat.h>

/* Some programs and especially the libc itself have to be careful
   what values to accept from the environment.  This special version
   checks for SUID or SGID first before doing any work.  */
char *
__libc_secure_getenv (const char *name)
{
  return __libc_enable_secure ? NULL : getenv (name);
}
weak_alias (__libc_secure_getenv, secure_getenv)
libc_hidden_weak (__libc_secure_getenv)

#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_17)
compat_symbol (libc, __libc_secure_getenv, __secure_getenv, GLIBC_2_0);
#endif
