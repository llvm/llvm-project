/* Increase the size of a dynamic array in preparation of an emplace operation.
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
__libc_dynarray_emplace_enlarge (struct dynarray_header *list,
                                 void *scratch, size_t element_size)
{
  size_t new_allocated;
  if (list->allocated == 0)
    {
      /* No scratch buffer provided.  Choose a reasonable default
         size.  */
      if (element_size < 4)
        new_allocated = 16;
      else if (element_size < 8)
        new_allocated = 8;
      else
        new_allocated = 4;
    }
  else
    /* Increase the allocated size, using an exponential growth
       policy.  */
    {
      new_allocated = list->allocated + list->allocated / 2 + 1;
      if (new_allocated <= list->allocated)
        {
          /* Overflow.  */
          __set_errno (ENOMEM);
          return false;
        }
    }

  size_t new_size;
  if (INT_MULTIPLY_WRAPV (new_allocated, element_size, &new_size))
    return false;
  void *new_array;
  if (list->array == scratch)
    {
      /* The previous array was not heap-allocated.  */
      new_array = malloc (new_size);
      if (new_array != NULL && list->array != NULL)
        memcpy (new_array, list->array, list->used * element_size);
    }
  else
    new_array = realloc (list->array, new_size);
  if (new_array == NULL)
    return false;
  list->array = new_array;
  list->allocated = new_allocated;
  return true;
}
libc_hidden_def (__libc_dynarray_emplace_enlarge)
