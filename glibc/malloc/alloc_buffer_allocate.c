/* Allocate a fixed-size allocation buffer using malloc.
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

#include <stdlib.h>

struct alloc_buffer
__libc_alloc_buffer_allocate (size_t size, void **pptr)
{
  *pptr = malloc (size);
  if (*pptr == NULL)
    return (struct alloc_buffer)
      {
        .__alloc_buffer_current = __ALLOC_BUFFER_INVALID_POINTER,
        .__alloc_buffer_end = __ALLOC_BUFFER_INVALID_POINTER
      };
  else
    return alloc_buffer_create (*pptr, size);
}
libc_hidden_def (__libc_alloc_buffer_allocate)
