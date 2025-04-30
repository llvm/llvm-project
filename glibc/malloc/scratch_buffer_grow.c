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

bool
__libc_scratch_buffer_grow (struct scratch_buffer *buffer)
{
  void *new_ptr;
  size_t new_length = buffer->length * 2;

  /* Discard old buffer.  */
  scratch_buffer_free (buffer);

  /* Check for overflow.  */
  if (__glibc_likely (new_length >= buffer->length))
    new_ptr = malloc (new_length);
  else
    {
      __set_errno (ENOMEM);
      new_ptr = NULL;
    }

  if (__glibc_unlikely (new_ptr == NULL))
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
libc_hidden_def (__libc_scratch_buffer_grow)
