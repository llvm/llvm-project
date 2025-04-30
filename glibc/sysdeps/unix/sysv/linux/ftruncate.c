/* Copyright (C) 2016-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <unistd.h>
#include <sysdep.h>
#include <errno.h>

#ifndef __OFF_T_MATCHES_OFF64_T
/* Truncate the file FD refers to LENGTH bytes.  */
int
__ftruncate (int fd, off_t length)
{
# ifndef __NR_ftruncate
  return INLINE_SYSCALL_CALL (ftruncate64, fd,
			      __ALIGNMENT_ARG SYSCALL_LL (length));
# else
  return INLINE_SYSCALL_CALL (ftruncate, fd, length);
# endif
}
weak_alias (__ftruncate, ftruncate)
#endif
