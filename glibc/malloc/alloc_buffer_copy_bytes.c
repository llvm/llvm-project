/* Copy an array of bytes into the buffer.
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

#include <string.h>

/* This function works on a copy of the buffer object, so that it can
   remain non-addressable in the caller.  */
struct alloc_buffer
__libc_alloc_buffer_copy_bytes (struct alloc_buffer buf,
                                const void *src, size_t len)
{
  void *ptr = alloc_buffer_alloc_bytes (&buf, len);
  if (ptr != NULL)
    memcpy (ptr, src, len);
  return buf;
}
libc_hidden_def (__libc_alloc_buffer_copy_bytes)
