/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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

#include <unistd.h>
#include <sysdep-cancel.h>
#include <shlib-compat.h>

#ifndef __OFF_T_MATCHES_OFF64_T

ssize_t
__libc_pwrite (int fd, const void *buf, size_t count, off_t offset)
{
  return SYSCALL_CANCEL (pwrite64, fd, buf, count, SYSCALL_LL_PRW (offset));
}

strong_alias (__libc_pwrite, __pwrite)
libc_hidden_weak (__pwrite)
weak_alias (__libc_pwrite, pwrite)

# if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_1, GLIBC_2_2)
compat_symbol (libc, __libc_pwrite, pwrite, GLIBC_2_2);
# endif

#endif
