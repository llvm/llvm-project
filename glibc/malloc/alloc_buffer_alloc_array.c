/* Array allocation from a fixed-size buffer.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <alloc_buffer.h>
#include <libc-pointer-arith.h>

void *
__libc_alloc_buffer_alloc_array (struct alloc_buffer *buf, size_t element_size,
                                 size_t align, size_t count)
{
  size_t current = buf->__alloc_buffer_current;
  /* The caller asserts that align is a power of two.  */
  size_t aligned = ALIGN_UP (current, align);
  size_t size;
  bool overflow = __builtin_mul_overflow (element_size, count, &size);
  size_t new_current = aligned + size;
  if (!overflow                /* Multiplication did not overflow.  */
      && aligned >= current    /* No overflow in align step.  */
      && new_current >= size   /* No overflow in size computation.  */
      && new_current <= buf->__alloc_buffer_end) /* Room in buffer.  */
    {
      buf->__alloc_buffer_current = new_current;
      return (void *) aligned;
    }
  else
    {
      alloc_buffer_mark_failed (buf);
      return NULL;
    }
}
libc_hidden_def (__libc_alloc_buffer_alloc_array)
