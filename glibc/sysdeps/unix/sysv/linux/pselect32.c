/* Synchronous I/O multiplexing.  Linux 32-bit time fallback.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <sys/select.h>
#include <sysdep-cancel.h>

#ifndef __ASSUME_TIME64_SYSCALLS

int
__pselect32 (int nfds, fd_set *readfds, fd_set *writefds,
	     fd_set *exceptfds, const struct __timespec64 *timeout,
	     const sigset_t *sigmask)
{
  struct timespec ts32, *pts32 = NULL;
  if (timeout != NULL)
    {
      ts32 = valid_timespec64_to_timespec (*timeout);
      pts32 = &ts32;
    }

  return SYSCALL_CANCEL (pselect6, nfds, readfds, writefds, exceptfds,
			 pts32,
			 ((__syscall_ulong_t[]){ (uintptr_t) sigmask,
						 __NSIG_BYTES }));
}
#endif
