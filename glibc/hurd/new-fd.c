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

#include <hurd/fd.h>
#include <stdlib.h>
#include "hurdmalloc.h"		/* XXX */

/* Allocate a new file descriptor structure
   and initialize it with PORT and CTTY.  */

struct hurd_fd *
_hurd_new_fd (io_t port, io_t ctty)
{
  struct hurd_fd *d = malloc (sizeof (struct hurd_fd));

  if (d != NULL)
    {
      /* Initialize the port cells.  */
      _hurd_port_init (&d->port, port);
      _hurd_port_init (&d->ctty, ctty);

      /* And the fcntl flags.  */
      d->flags = 0;
    }

  return d;
}
