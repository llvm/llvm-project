/* Copyright (C) 1994-2021 Free Software Foundation, Inc.
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

#include <hurd.h>
#include <hurd/fd.h>

/* Allocate a new file descriptor and install PORT in it.  FLAGS are as for
   `open'; only O_IGNORE_CTTY and O_CLOEXEC are meaningful.

   If the descriptor table is full, set errno, and return -1.
   If DEALLOC is nonzero, deallocate PORT first.  */
int
_hurd_intern_fd (io_t port, int flags, int dealloc)
{
  int fd;
  struct hurd_fd *d;

  HURD_CRITICAL_BEGIN;
  d = _hurd_alloc_fd (&fd, 0);
  if (d != NULL)
    {
      _hurd_port2fd (d, port, flags);
      __spin_unlock (&d->port.lock);
    }
  HURD_CRITICAL_END;

  if (d == NULL)
    {
      if (dealloc)
	__mach_port_deallocate (__mach_task_self (), port);
      return -1;
    }

  return fd;
}
libc_hidden_def (_hurd_intern_fd)
