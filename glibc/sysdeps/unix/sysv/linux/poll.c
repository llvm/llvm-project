/* Linux poll implementation.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <errno.h>
#include <sys/poll.h>

#include <sysdep-cancel.h>
#include <sys/syscall.h>

int
__poll (struct pollfd *fds, nfds_t nfds, int timeout)
{
#ifdef __NR_poll
  return SYSCALL_CANCEL (poll, fds, nfds, timeout);
#else
  struct timespec timeout_ts;
  struct timespec *timeout_ts_p = NULL;

  if (timeout >= 0)
    {
      timeout_ts.tv_sec = timeout / 1000;
      timeout_ts.tv_nsec = (timeout % 1000) * 1000000;
      timeout_ts_p = &timeout_ts;
    }

  return SYSCALL_CANCEL (ppoll, fds, nfds, timeout_ts_p, NULL, 0);
#endif
}
libc_hidden_def (__poll)
weak_alias (__poll, poll)
strong_alias (__poll, __libc_poll)
