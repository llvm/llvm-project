/* Increase the size of a dynamic array and clear the new part.
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

#include <dynarray.h>
#include <string.h>

bool
__libc_dynarray_resize_clear (struct dynarray_header *list, size_t size,
                              void *scratch, size_t element_size)
{
  size_t old_size = list->used;
  if (!__libc_dynarray_resize (list, size, scratch, element_size))
    return false;
  /* __libc_dynarray_resize already checked for overflow.  */
  char *array = list->array;
  memset (array + (old_size * element_size), 0,
          (size - old_size) * element_size);
  return true;
}
libc_hidden_def (__libc_dynarray_resize_clear)
