/* Copyright (C) 1991-2021 Free Software Foundation, Inc.

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

/* This is initialized in dtable.c when that gets linked in.
   If dtable.c is not linked in, it will be zero.  */
static file_t (*_default_hurd_getdport_fn) (int fd) = 0;
weak_alias (_default_hurd_getdport_fn, _hurd_getdport_fn)

file_t
__getdport (int fd)
{
  if (_hurd_getdport_fn)
    /* dtable.c has defined the function to fetch a port from the real file
       descriptor table.  */
    return (*_hurd_getdport_fn) (fd);

  /* getdport is the only use of file descriptors,
     so we don't bother allocating a real table.  */

  if (_hurd_init_dtable == NULL)
    {
      /* Never had a descriptor table.  */
      errno = EBADF;
      return MACH_PORT_NULL;
    }

  if (fd < 0 || (unsigned int) fd > _hurd_init_dtablesize
      || _hurd_init_dtable[fd] == MACH_PORT_NULL)
    {
      errno = EBADF;
      return MACH_PORT_NULL;
    }
  else
    {
      __mach_port_mod_refs (__mach_task_self (), _hurd_init_dtable[fd],
			    MACH_PORT_RIGHT_SEND, 1);
      return _hurd_init_dtable[fd];
    }
}

weak_alias (__getdport, getdport)
