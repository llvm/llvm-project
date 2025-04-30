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
#include <string.h>

bool
__libc_scratch_buffer_grow_preserve (struct scratch_buffer *buffer)
{
  size_t new_length = 2 * buffer->length;
  void *new_ptr;

  if (buffer->data == buffer->__space.__c)
    {
      /* Move buffer to the heap.  No overflow is possible because
	 buffer->length describes a small buffer on the stack.  */
      new_ptr = malloc (new_length);
      if (new_ptr == NULL)
	return false;
      memcpy (new_ptr, buffer->__space.__c, buffer->length);
    }
  else
    {
      /* Buffer was already on the heap.  Check for overflow.  */
      if (__glibc_likely (new_length >= buffer->length))
	new_ptr = realloc (buffer->data, new_length);
      else
	{
	  __set_errno (ENOMEM);
	  new_ptr = NULL;
	}

      if (__glibc_unlikely (new_ptr == NULL))
	{
	  /* Deallocate, but buffer must remain valid to free.  */
	  free (buffer->data);
	  scratch_buffer_init (buffer);
	  return false;
	}
    }

  /* Install new heap-based buffer.  */
  buffer->data = new_ptr;
  buffer->length = new_length;
  return true;
}
libc_hidden_def (__libc_scratch_buffer_grow_preserve)
