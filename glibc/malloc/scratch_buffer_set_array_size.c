/* Variable-sized buffer with on-stack default allocation.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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
#include <errno.h>
#include <limits.h>

bool
__libc_scratch_buffer_set_array_size (struct scratch_buffer *buffer,
				      size_t nelem, size_t size)
{
  size_t new_length = nelem * size;

  /* Avoid overflow check if both values are small. */
  if ((nelem | size) >> (sizeof (size_t) * CHAR_BIT / 2) != 0
      && nelem != 0 && size != new_length / nelem)
    {
      /* Overflow.  Discard the old buffer, but it must remain valid
	 to free.  */
      scratch_buffer_free (buffer);
      scratch_buffer_init (buffer);
      __set_errno (ENOMEM);
      return false;
    }

  if (new_length <= buffer->length)
    return true;

  /* Discard old buffer.  */
  scratch_buffer_free (buffer);

  char *new_ptr = malloc (new_length);
  if (new_ptr == NULL)
    {
      /* Buffer must remain valid to free.  */
      scratch_buffer_init (buffer);
      return false;
    }

  /* Install new heap-based buffer.  */
  buffer->data = new_ptr;
  buffer->length = new_length;
  return true;
}
libc_hidden_def (__libc_scratch_buffer_set_array_size)
