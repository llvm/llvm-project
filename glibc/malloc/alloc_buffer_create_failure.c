/* Terminate the process as the result of an invalid allocation buffer.
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
#include <stdio.h>

void
__libc_alloc_buffer_create_failure (void *start, size_t size)
{
  char buf[200];
  __snprintf (buf, sizeof (buf), "Fatal glibc error: "
              "invalid allocation buffer of size %zu\n",
              size);
  __libc_fatal (buf);
}
libc_hidden_def (__libc_alloc_buffer_create_failure)
