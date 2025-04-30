/* Linux lseek implementation, 32 bits off_t.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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
#include <stdint.h>
#include <sys/types.h>
#include <sysdep.h>
#include <errno.h>

#ifndef __OFF_T_MATCHES_OFF64_T

/* Test for overflows of structures where we ask the kernel to fill them
   in with standard 64-bit syscalls but return them through APIs that
   only expose the low 32 bits of some fields.  */

static inline off_t lseek_overflow (loff_t res)
{
  off_t retval = (off_t) res;
  if (retval == res)
    return retval;

  __set_errno (EOVERFLOW);
  return (off_t) -1;
}

off_t
__lseek (int fd, off_t offset, int whence)
{
# ifdef __NR__llseek
  loff_t res;
  int rc = INLINE_SYSCALL_CALL (_llseek, fd,
				(long) (((uint64_t) (offset)) >> 32),
				(long) offset, &res, whence);
  return rc ?: lseek_overflow (res);
# else
  return INLINE_SYSCALL_CALL (lseek, fd, offset, whence);
# endif
}
libc_hidden_def (__lseek)
weak_alias (__lseek, lseek)
strong_alias (__lseek, __libc_lseek)
#endif /* __OFF_T_MATCHES_OFF64_T  */
