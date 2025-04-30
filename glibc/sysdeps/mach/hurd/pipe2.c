/* Copyright (C) 1992-2021 Free Software Foundation, Inc.

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
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <fcntl-internal.h>
#include <hurd.h>

/* Create a one-way communication channel (pipe).
   Actually the channel is two-way on the Hurd.
   If successful, two file descriptors are stored in FDS;
   bytes written on FDS[1] can be read from FDS[0].
   Apply FLAGS to the new file descriptors.
   Returns 0 if successful, -1 if not.  */
int
__pipe2 (int fds[2], int flags)
{
  int save_errno = errno;
  int result;

  if (flags & ~(O_CLOEXEC | O_NONBLOCK))
    return __hurd_fail (EINVAL);

  flags = o_to_sock_flags (flags);

  /* The magic S_IFIFO protocol tells the pflocal server to create
     sockets which report themselves as FIFOs, as POSIX requires for
     pipes.  */
  result = __socketpair (PF_LOCAL, SOCK_STREAM | flags, S_IFIFO, fds);
  if (result == -1 && errno == EPROTONOSUPPORT)
    {
      /* We contacted an "old" pflocal server that doesn't support the
         magic S_IFIFO protocol.
	 FIXME: Remove this junk somewhere in the future.  */
      __set_errno (save_errno);
      return __socketpair (PF_LOCAL, SOCK_STREAM | flags, 0, fds);
    }

  return result;
}
weak_alias (__pipe2, pipe2)
