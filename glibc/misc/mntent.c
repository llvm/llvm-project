/* Utilities for reading/writing fstab, mtab, etc.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#include <mntent.h>
#include <stdlib.h>
#include <allocate_once.h>

struct mntent_buffer
{
  struct mntent m;
  char buffer[4096];
};

/* We don't want to allocate the static buffer all the time since it
   is not always used (in fact, rather infrequently).  */
libc_freeres_ptr (static void *mntent_buffer);

static void *
allocate (void *closure)
{
  return malloc (sizeof (struct mntent_buffer));
}

static void
deallocate (void *closure, void *ptr)
{
  free (ptr);
}

struct mntent *
getmntent (FILE *stream)
{
  struct mntent_buffer *buffer = allocate_once (&mntent_buffer,
						allocate, deallocate, NULL);
  if (buffer == NULL)
    /* If no core is available we don't have a chance to run the
       program successfully and so returning NULL is an acceptable
       result.  */
    return NULL;

  return __getmntent_r (stream, &buffer->m,
			buffer->buffer, sizeof (buffer->buffer));
}
