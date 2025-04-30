/* Wrapper for the mlock2 system call with fallback to mlock.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.

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

#include <sys/mman.h>
#include <errno.h>
#include <sysdep.h>

int
mlock2 (const void *addr, size_t length, unsigned int flags)
{
#ifdef __ASSUME_MLOCK2
  return INLINE_SYSCALL_CALL (mlock2, addr, length, flags);
#else
  if (flags == 0)
    return INLINE_SYSCALL_CALL (mlock, addr, length);
  int ret = INLINE_SYSCALL_CALL (mlock2, addr, length, flags);
  if (ret == 0 || errno != ENOSYS)
    return ret;
  /* Treat the missing system call as an invalid (non-zero) flag
     argument.  */
  __set_errno (EINVAL);
  return -1;
#endif /* __ASSUME_MLOCK2 */
}
