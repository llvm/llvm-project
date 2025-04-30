/* poll file descriptors.  Hurd version.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
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
#include <sys/time.h>
#include <hurd.h>
#include <hurd/fd.h>

int
ppoll (struct pollfd *fds, nfds_t nfds,
       const struct timespec *timeout, const sigset_t *sigmask)
{
  return _hurd_select (nfds, fds, NULL, NULL, NULL, timeout, sigmask);
}
libc_hidden_def (ppoll)
