/* Copyright (C) 2008-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <sys/times.h>
#include <sysdep.h>


clock_t
__times (struct tms *buf)
{
  clock_t ret = INTERNAL_SYSCALL_CALL (times, buf);
  if (INTERNAL_SYSCALL_ERROR_P (ret)
      && __glibc_unlikely (INTERNAL_SYSCALL_ERRNO (ret) == EFAULT)
      && buf)
    {
      /* This might be an error or not.  For architectures which have no
	 separate return value and error indicators we cannot
	 distinguish a return value of e.g. (clock_t) -14 from -EFAULT.
	 Therefore the only course of action is to dereference the user
	 -supplied structure on a return of (clock_t) -14.  This will crash
	 applications which pass in an invalid non-NULL BUF pointer.
	 Note that Linux allows BUF to be NULL in which case we skip this.  */
#define touch(v) \
      do {								      \
	clock_t temp = v;						      \
	asm volatile ("" : "+r" (temp));				      \
	v = temp;							      \
      } while (0)
      touch (buf->tms_utime);
      touch (buf->tms_stime);
      touch (buf->tms_cutime);
      touch (buf->tms_cstime);

      /* If we come here the memory is valid and the kernel did not
	 return an EFAULT error, but rather e.g. (clock_t) -14.
	 Return the value given by the kernel.  */
    }

  /* On Linux this function never fails except with EFAULT.
     POSIX says that returning a value (clock_t) -1 indicates an error,
     but on Linux this is simply one of the valid clock values after
     clock_t wraps.  Therefore when we would return (clock_t) -1, we
     instead return (clock_t) 0, and loose a tick of accuracy (having
     returned 0 for two consecutive calls even though the clock
     advanced).  */
  if (ret == (clock_t) -1)
    return (clock_t) 0;

  return ret;
}
weak_alias (__times, times)
