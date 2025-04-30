/* Copy the dynamically-allocated area to an explicitly-sized heap allocation.
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
#include <stdlib.h>
#include <string.h>

bool
__libc_dynarray_finalize (struct dynarray_header *list,
                          void *scratch, size_t element_size,
                          struct dynarray_finalize_result *result)
{
  if (__dynarray_error (list))
    /* The caller will reported the deferred error.  */
    return false;

  size_t used = list->used;

  /* Empty list.  */
  if (used == 0)
    {
      /* An empty list could still be backed by a heap-allocated
         array.  Free it if necessary.  */
      if (list->array != scratch)
        free (list->array);
      *result = (struct dynarray_finalize_result) { NULL, 0 };
      return true;
    }

  size_t allocation_size = used * element_size;
  void *heap_array = malloc (allocation_size);
  if (heap_array != NULL)
    {
      /* The new array takes ownership of the strings.  */
      if (list->array != NULL)
        memcpy (heap_array, list->array, allocation_size);
      if (list->array != scratch)
        free (list->array);
      *result = (struct dynarray_finalize_result)
        { .array = heap_array, .length = used };
      return true;
    }
  else
    /* The caller will perform the freeing operation.  */
    return false;
}
libc_hidden_def (__libc_dynarray_finalize)
