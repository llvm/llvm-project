/* Type-safe arrays which grow dynamically.  Shared definitions.
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

/* To use the dynarray facility, you need to include
   <malloc/dynarray-skeleton.c> and define the parameter macros
   documented in that file.

   A minimal example which provides a growing list of integers can be
   defined like this:

     struct int_array
     {
       // Pointer to result array followed by its length,
       // as required by DYNARRAY_FINAL_TYPE.
       int *array;
       size_t length;
     };

     #define DYNARRAY_STRUCT dynarray_int
     #define DYNARRAY_ELEMENT int
     #define DYNARRAY_PREFIX dynarray_int_
     #define DYNARRAY_FINAL_TYPE struct int_array
     #include <malloc/dynarray-skeleton.c>

   To create a three-element array with elements 1, 2, 3, use this
   code:

     struct dynarray_int dyn;
     dynarray_int_init (&dyn);
     for (int i = 1; i <= 3; ++i)
       {
         int *place = dynarray_int_emplace (&dyn);
         assert (place != NULL);
         *place = i;
       }
     struct int_array result;
     bool ok = dynarray_int_finalize (&dyn, &result);
     assert (ok);
     assert (result.length == 3);
     assert (result.array[0] == 1);
     assert (result.array[1] == 2);
     assert (result.array[2] == 3);
     free (result.array);

   If the elements contain resources which must be freed, define
   DYNARRAY_ELEMENT_FREE appropriately, like this:

     struct str_array
     {
       char **array;
       size_t length;
     };

     #define DYNARRAY_STRUCT dynarray_str
     #define DYNARRAY_ELEMENT char *
     #define DYNARRAY_ELEMENT_FREE(ptr) free (*ptr)
     #define DYNARRAY_PREFIX dynarray_str_
     #define DYNARRAY_FINAL_TYPE struct str_array
     #include <malloc/dynarray-skeleton.c>

   Compared to scratch buffers, dynamic arrays have the following
   features:

   - They have an element type, and are not just an untyped buffer of
     bytes.

   - When growing, previously stored elements are preserved.  (It is
     expected that scratch_buffer_grow_preserve and
     scratch_buffer_set_array_size eventually go away because all
     current users are moved to dynamic arrays.)

   - Scratch buffers have a more aggressive growth policy because
     growing them typically means a retry of an operation (across an
     NSS service module boundary), which is expensive.

   - For the same reason, scratch buffers have a much larger initial
     stack allocation.  */

#ifndef _DYNARRAY_H
#define _DYNARRAY_H

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

struct dynarray_header
{
  size_t used;
  size_t allocated;
  void *array;
};

/* Marker used in the allocated member to indicate that an error was
   encountered.  */
static inline size_t
__dynarray_error_marker (void)
{
  return -1;
}

/* Internal function.  See the has_failed function in
   dynarray-skeleton.c.  */
static inline bool
__dynarray_error (struct dynarray_header *list)
{
  return list->allocated == __dynarray_error_marker ();
}

/* Internal function.  Enlarge the dynamically allocated area of the
   array to make room for one more element.  SCRATCH is a pointer to
   the scratch area (which is not heap-allocated and must not be
   freed).  ELEMENT_SIZE is the size, in bytes, of one element.
   Return false on failure, true on success.  */
bool __libc_dynarray_emplace_enlarge (struct dynarray_header *,
                                      void *scratch, size_t element_size);

/* Internal function.  Enlarge the dynamically allocated area of the
   array to make room for at least SIZE elements (which must be larger
   than the existing used part of the dynamic array).  SCRATCH is a
   pointer to the scratch area (which is not heap-allocated and must
   not be freed).  ELEMENT_SIZE is the size, in bytes, of one element.
   Return false on failure, true on success.  */
bool __libc_dynarray_resize (struct dynarray_header *, size_t size,
                             void *scratch, size_t element_size);

/* Internal function.  Like __libc_dynarray_resize, but clear the new
   part of the dynamic array.  */
bool __libc_dynarray_resize_clear (struct dynarray_header *, size_t size,
                                   void *scratch, size_t element_size);

/* Internal type.  */
struct dynarray_finalize_result
{
  void *array;
  size_t length;
};

/* Internal function.  Copy the dynamically-allocated area to an
   explicitly-sized heap allocation.  SCRATCH is a pointer to the
   embedded scratch space.  ELEMENT_SIZE is the size, in bytes, of the
   element type.  On success, true is returned, and pointer and length
   are written to *RESULT.  On failure, false is returned.  The caller
   has to take care of some of the memory management; this function is
   expected to be called from dynarray-skeleton.c.  */
bool __libc_dynarray_finalize (struct dynarray_header *list, void *scratch,
                               size_t element_size,
                               struct dynarray_finalize_result *result);


/* Internal function.  Terminate the process after an index error.
   SIZE is the number of elements of the dynamic array.  INDEX is the
   lookup index which triggered the failure.  */
_Noreturn void __libc_dynarray_at_failure (size_t size, size_t index);

#ifndef _ISOMAC
libc_hidden_proto (__libc_dynarray_emplace_enlarge)
libc_hidden_proto (__libc_dynarray_resize)
libc_hidden_proto (__libc_dynarray_resize_clear)
libc_hidden_proto (__libc_dynarray_finalize)
libc_hidden_proto (__libc_dynarray_at_failure)
#endif

#endif /* _DYNARRAY_H */
