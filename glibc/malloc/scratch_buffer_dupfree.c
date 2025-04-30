/* Variable-sized buffer with on-stack default allocation.
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

#ifndef _LIBC
# include <libc-config.h>
#endif

#include <scratch_buffer.h>
#include <string.h>

void *
__libc_scratch_buffer_dupfree (struct scratch_buffer *buffer, size_t size)
{
  void *data = buffer->data;
  if (data == buffer->__space.__c)
    {
      void *copy = malloc (size);
      return copy != NULL ? memcpy (copy, data, size) : NULL;
    }
  else
    {
      void *copy = realloc (data, size);
      return copy != NULL ? copy : data;
    }
}
libc_hidden_def (__libc_scratch_buffer_dupfree)
