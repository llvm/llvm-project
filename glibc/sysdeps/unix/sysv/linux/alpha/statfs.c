/* Return information about the filesystem on which FILE resides.  Linux/alpha.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

#include <sys/statfs.h>
#include <sysdep.h>
#include <kernel_stat.h>

/* Return information about the filesystem on which FILE resides.  */
int
__statfs (const char *file, struct statfs *buf)
{
  return INLINE_SYSCALL_CALL (statfs, file, buf);
}
libc_hidden_def (__statfs)
weak_alias (__statfs, statfs)
