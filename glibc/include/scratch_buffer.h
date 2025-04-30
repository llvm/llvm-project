/* Variable-sized buffer with on-stack default allocation.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#ifndef _SCRATCH_BUFFER_H
#define _SCRATCH_BUFFER_H

/* Scratch buffers with a default stack allocation and fallback to
   heap allocation.  It is expected that this function is used in this
   way:

     struct scratch_buffer tmpbuf;
     scratch_buffer_init (&tmpbuf);

     while (!function_that_uses_buffer (tmpbuf.data, tmpbuf.length))
       if (!scratch_buffer_grow (&tmpbuf))
	 return -1;

     scratch_buffer_free (&tmpbuf);
     return 0;

   The allocation functions (scratch_buffer_grow,
   scratch_buffer_grow_preserve, scratch_buffer_set_array_size) make
   sure that the heap allocation, if any, is freed, so that the code
   above does not have a memory leak.  The buffer still remains in a
   state that can be deallocated using scratch_buffer_free, so a loop
   like this is valid as well:

     struct scratch_buffer tmpbuf;
     scratch_buffer_init (&tmpbuf);

     while (!function_that_uses_buffer (tmpbuf.data, tmpbuf.length))
       if (!scratch_buffer_grow (&tmpbuf))
	 break;

     scratch_buffer_free (&tmpbuf);

   scratch_buffer_grow and scratch_buffer_grow_preserve are guaranteed
   to grow the buffer by at least 512 bytes.  This means that when
   using the scratch buffer as a backing store for a non-character
   array whose element size, in bytes, is 512 or smaller, the scratch
   buffer only has to grow once to make room for at least one more
   element.
*/

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

/* Scratch buffer.  Must be initialized with scratch_buffer_init
   before its use.  */
struct scratch_buffer {
  void *data;    /* Pointer to the beginning of the scratch area.  */
  size_t length; /* Allocated space at the data pointer, in bytes.  */
  union { max_align_t __align; char __c[1024]; } __space;
};

/* Initializes *BUFFER so that BUFFER->data points to BUFFER->__space
   and BUFFER->length reflects the available space.  */
static inline void
scratch_buffer_init (struct scratch_buffer *buffer)
{
  buffer->data = buffer->__space.__c;
  buffer->length = sizeof (buffer->__space);
}

/* Deallocates *BUFFER (if it was heap-allocated).  */
static inline void
scratch_buffer_free (struct scratch_buffer *buffer)
{
  if (buffer->data != buffer->__space.__c)
    free (buffer->data);
}

/* Grow *BUFFER by some arbitrary amount.  The buffer contents is NOT
   preserved.  Return true on success, false on allocation failure (in
   which case the old buffer is freed).  On success, the new buffer is
   larger than the previous size.  On failure, *BUFFER is deallocated,
   but remains in a free-able state, and errno is set.  */
bool __libc_scratch_buffer_grow (struct scratch_buffer *buffer);
libc_hidden_proto (__libc_scratch_buffer_grow)

/* Alias for __libc_scratch_buffer_grow.  */
static __always_inline bool
scratch_buffer_grow (struct scratch_buffer *buffer)
{
  return __glibc_likely (__libc_scratch_buffer_grow (buffer));
}

/* Like __libc_scratch_buffer_grow, but preserve the old buffer
   contents on success, as a prefix of the new buffer.  */
bool __libc_scratch_buffer_grow_preserve (struct scratch_buffer *buffer);
libc_hidden_proto (__libc_scratch_buffer_grow_preserve)

/* Alias for __libc_scratch_buffer_grow_preserve.  */
static __always_inline bool
scratch_buffer_grow_preserve (struct scratch_buffer *buffer)
{
  return __glibc_likely (__libc_scratch_buffer_grow_preserve (buffer));
}

/* Grow *BUFFER so that it can store at least NELEM elements of SIZE
   bytes.  The buffer contents are NOT preserved.  Both NELEM and SIZE
   can be zero.  Return true on success, false on allocation failure
   (in which case the old buffer is freed, but *BUFFER remains in a
   free-able state, and errno is set).  It is unspecified whether this
   function can reduce the array size.  */
bool __libc_scratch_buffer_set_array_size (struct scratch_buffer *buffer,
					   size_t nelem, size_t size);
libc_hidden_proto (__libc_scratch_buffer_set_array_size)

/* Alias for __libc_scratch_set_array_size.  */
static __always_inline bool
scratch_buffer_set_array_size (struct scratch_buffer *buffer,
			       size_t nelem, size_t size)
{
  return __glibc_likely (__libc_scratch_buffer_set_array_size
			 (buffer, nelem, size));
}

/* Return a copy of *BUFFER's first SIZE bytes as a heap-allocated block,
   deallocating *BUFFER if it was heap-allocated.  SIZE must be at
   most *BUFFER's size.  Return NULL (setting errno) on memory
   exhaustion.  */
void *__libc_scratch_buffer_dupfree (struct scratch_buffer *buffer,
                                     size_t size);
libc_hidden_proto (__libc_scratch_buffer_dupfree)

/* Alias for __libc_scratch_dupfree.  */
static __always_inline void *
scratch_buffer_dupfree (struct scratch_buffer *buffer, size_t size)
{
  void *r = __libc_scratch_buffer_dupfree (buffer, size);
  return __glibc_likely (r != NULL) ? r : NULL;
}

#endif /* _SCRATCH_BUFFER_H */
