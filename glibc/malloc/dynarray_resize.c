/* Increase the size of a dynamic array.
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
#include <errno.h>
#include <intprops.h>
#include <stdlib.h>
#include <string.h>

bool
__libc_dynarray_resize (struct dynarray_header *list, size_t size,
                        void *scratch, size_t element_size)
{
  /* The existing allocation provides sufficient room.  */
  if (size <= list->allocated)
    {
      list->used = size;
      return true;
    }

  /* Otherwise, use size as the new allocation size.  The caller is
     expected to provide the final size of the array, so there is no
     over-allocation here.  */

  size_t new_size_bytes;
  if (INT_MULTIPLY_WRAPV (size, element_size, &new_size_bytes))
    {
      /* Overflow.  */
      __set_errno (ENOMEM);
      return false;
    }
  void *new_array;
  if (list->array == scratch)
    {
      /* The previous array was not heap-allocated.  */
      new_array = malloc (new_size_bytes);
      if (new_array != NULL && list->array != NULL)
        memcpy (new_array, list->array, list->used * element_size);
    }
  else
    new_array = realloc (list->array, new_size_bytes);
  if (new_array == NULL)
    return false;
  list->array = new_array;
  list->allocated = size;
  list->used = size;
  return true;
}
libc_hidden_def (__libc_dynarray_resize)
