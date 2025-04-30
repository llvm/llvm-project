/* Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

#include <sys/poll.h>


int
__ppoll_chk (struct pollfd *fds, nfds_t nfds, const struct timespec *timeout,
	     const __sigset_t *ss, __SIZE_TYPE__ fdslen)
{
  if (fdslen / sizeof (*fds) < nfds)
    __chk_fail ();

  return ppoll (fds, nfds, timeout, ss);
}
