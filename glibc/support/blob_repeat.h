/* Repeating a memory blob, with alias mapping optimization.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#ifndef SUPPORT_BLOB_REPEAT_H
#define SUPPORT_BLOB_REPEAT_H

#include <stdbool.h>
#include <stddef.h>

struct support_blob_repeat
{
  void *start;
  size_t size;
  bool use_malloc;
};

/* Return an allocation of COUNT elements, each of ELEMENT_SIZE bytes,
   initialized with the bytes starting at ELEMENT.  The memory is
   writable (and thus counts towards the commit charge).  In case of
   on error, all members of the return struct are zero-initialized,
   and errno is set accordingly.  */
struct support_blob_repeat support_blob_repeat_allocate (const void *element,
                                                         size_t element_size,
                                                         size_t count);

/* Like support_blob_repeat_allocate, except that copy-on-write
   semantics are disabled.  This means writing to one part of the blob
   can affect other parts.  It is possible to map non-shared memory
   over parts of the resulting blob using MAP_ANONYMOUS | MAP_FIXED
   | MAP_PRIVATE, so that writes to these parts do not affect
   others.  */
struct support_blob_repeat support_blob_repeat_allocate_shared
  (const void *element, size_t element_size, size_t count);

/* Deallocate the blob created by support_blob_repeat_allocate or
   support_blob_repeat_allocate_shared.  */
void support_blob_repeat_free (struct support_blob_repeat *);

#endif /* SUPPORT_BLOB_REPEAT_H */
