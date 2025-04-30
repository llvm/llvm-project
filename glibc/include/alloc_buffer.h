/* Allocation from a fixed-size buffer.
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

/* Allocation buffers are used to carve out sub-allocations from a
   larger allocation.  Their primary application is in writing NSS
   modules, which receive a caller-allocated buffer in which they are
   expected to store variable-length results:

     void *buffer = ...;
     size_t buffer_size = ...;

     struct alloc_buffer buf = alloc_buffer_create (buffer, buffer_size);
     result->gr_name = alloc_buffer_copy_string (&buf, name);

     // Allocate a list of group_count groups and copy strings into it.
     char **group_list = alloc_buffer_alloc_array
       (&buf, char *, group_count  + 1);
     if (group_list == NULL)
       return ...; // Request a larger buffer.
     for (int i = 0; i < group_count; ++i)
       group_list[i] = alloc_buffer_copy_string (&buf, group_list_src[i]);
     group_list[group_count] = NULL;
     ...

     if (alloc_buffer_has_failed (&buf))
       return ...; // Request a larger buffer.
     result->gr_mem = group_list;
     ...

   Note that it is not necessary to check the results of individual
   allocation operations if the returned pointer is not dereferenced.
   Allocation failure is sticky, so one check using
   alloc_buffer_has_failed at the end covers all previous failures.

   A different use case involves combining multiple heap allocations
   into a single, large one.  In the following example, an array of
   doubles and an array of ints is allocated:

     size_t double_array_size = ...;
     size_t int_array_size = ...;

     void *heap_ptr;
     struct alloc_buffer buf = alloc_buffer_allocate
       (double_array_size * sizeof (double) + int_array_size * sizeof (int),
        &heap_ptr);
     _Static_assert (__alignof__ (double) >= __alignof__ (int),
                     "no padding after double array");
     double *double_array = alloc_buffer_alloc_array
       (&buf, double, double_array_size);
     int *int_array = alloc_buffer_alloc_array (&buf, int, int_array_size);
     if (alloc_buffer_has_failed (&buf))
       return ...; // Report error.
     ...
     free (heap_ptr);

   The advantage over manual coding is that the computation of the
   allocation size does not need an overflow check.  In case of an
   overflow, one of the subsequent allocations from the buffer will
   fail.  The initial size computation is checked for consistency at
   run time, too.  */

#ifndef _ALLOC_BUFFER_H
#define _ALLOC_BUFFER_H

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <sys/param.h>

/* struct alloc_buffer objects refer to a region of bytes in memory of a
   fixed size.  The functions below can be used to allocate single
   objects and arrays from this memory region, or write to its end.
   On allocation failure (or if an attempt to write beyond the end of
   the buffer with one of the copy functions), the buffer enters a
   failed state.

   struct alloc_buffer objects can be copied.  The backing buffer will
   be shared, but the current write position will be independent.

   Conceptually, the memory region consists of a current write pointer
   and a limit, beyond which the write pointer cannot move.  */
struct alloc_buffer
{
  /* uintptr_t is used here to simplify the alignment code, and to
     avoid issues undefined subtractions if the buffer covers more
     than half of the address space (which would result in differences
     which could not be represented as a ptrdiff_t value).  */
  uintptr_t __alloc_buffer_current;
  uintptr_t __alloc_buffer_end;
};

enum
  {
    /* The value for the __alloc_buffer_current member which marks the
       buffer as invalid (together with a zero-length buffer).  */
    __ALLOC_BUFFER_INVALID_POINTER = 0,
  };

/* Internal function.  Terminate the process using __libc_fatal.  */
void __libc_alloc_buffer_create_failure (void *start, size_t size);
/* clang wants a hidden proto ahead of use. */
#ifndef _ISOMAC
libc_hidden_proto (__libc_alloc_buffer_create_failure)
#endif

/* Create a new allocation buffer.  The byte range from START to START
   + SIZE - 1 must be valid, and the allocation buffer allocates
   objects from that range.  If START is NULL (so that SIZE must be
   0), the buffer is marked as failed immediately.  */
static inline struct alloc_buffer
alloc_buffer_create (void *start, size_t size)
{
  uintptr_t current = (uintptr_t) start;
  uintptr_t end = (uintptr_t) start + size;
  if (end < current)
    __libc_alloc_buffer_create_failure (start, size);
  return (struct alloc_buffer) { current, end };
}

/* Internal function.  See alloc_buffer_allocate below.  */
struct alloc_buffer __libc_alloc_buffer_allocate (size_t size, void **pptr)
  __attribute__ ((nonnull (2)));
/* clang wants a hidden proto ahead of use. */
#ifndef _ISOMAC
libc_hidden_proto (__libc_alloc_buffer_allocate)
#endif

/* Allocate a buffer of SIZE bytes using malloc.  The returned buffer
   is in a failed state if malloc fails.  *PPTR points to the start of
   the buffer and can be used to free it later, after the returned
   buffer has been freed.  */
static __always_inline __attribute__ ((nonnull (2)))
struct alloc_buffer alloc_buffer_allocate (size_t size, void **pptr)
{
  return __libc_alloc_buffer_allocate (size, pptr);
}

/* Mark the buffer as failed.  */
static inline void __attribute__ ((nonnull (1)))
alloc_buffer_mark_failed (struct alloc_buffer *buf)
{
  buf->__alloc_buffer_current = __ALLOC_BUFFER_INVALID_POINTER;
  buf->__alloc_buffer_end = __ALLOC_BUFFER_INVALID_POINTER;
}

/* Return the remaining number of bytes in the buffer.  */
static __always_inline __attribute__ ((nonnull (1))) size_t
alloc_buffer_size (const struct alloc_buffer *buf)
{
  return buf->__alloc_buffer_end - buf->__alloc_buffer_current;
}

/* Return true if the buffer has been marked as failed.  */
static inline bool __attribute__ ((nonnull (1)))
alloc_buffer_has_failed (const struct alloc_buffer *buf)
{
  return buf->__alloc_buffer_current == __ALLOC_BUFFER_INVALID_POINTER;
}

/* Add a single byte to the buffer (consuming the space for this
   byte).  Mark the buffer as failed if there is not enough room.  */
static inline void __attribute__ ((nonnull (1)))
alloc_buffer_add_byte (struct alloc_buffer *buf, unsigned char b)
{
  if (__glibc_likely (buf->__alloc_buffer_current < buf->__alloc_buffer_end))
    {
      *(unsigned char *) buf->__alloc_buffer_current = b;
      ++buf->__alloc_buffer_current;
    }
  else
    alloc_buffer_mark_failed (buf);
}

/* Obtain a pointer to LENGTH bytes in BUF, and consume these bytes.
   NULL is returned if there is not enough room, and the buffer is
   marked as failed, or if the buffer has already failed.
   (Zero-length allocations from an empty buffer which has not yet
   failed succeed.)  The buffer contents is not modified.  */
static inline __attribute__ ((nonnull (1))) void *
alloc_buffer_alloc_bytes (struct alloc_buffer *buf, size_t length)
{
  if (length <= alloc_buffer_size (buf))
    {
      void *result = (void *) buf->__alloc_buffer_current;
      buf->__alloc_buffer_current += length;
      return result;
    }
  else
    {
      alloc_buffer_mark_failed (buf);
      return NULL;
    }
}

/* Internal function.  Statically assert that the type size is
   constant and valid.  */
static __always_inline size_t
__alloc_buffer_assert_size (size_t size)
{
  if (!__builtin_constant_p (size))
    {
      __errordecl (error, "type size is not constant");
      error ();
    }
  else if (size == 0)
    {
      __errordecl (error, "type size is zero");
      error ();
    }
  return size;
}

/* Internal function.  Statically assert that the type alignment is
   constant and valid.  */
static __always_inline size_t
__alloc_buffer_assert_align (size_t align)
{
  if (!__builtin_constant_p (align))
    {
      __errordecl (error, "type alignment is not constant");
      error ();
    }
  else if (align == 0)
    {
      __errordecl (error, "type alignment is zero");
      error ();
    }
  else if (!powerof2 (align))
    {
      __errordecl (error, "type alignment is not a power of two");
      error ();
    }
  return align;
}

/* Internal function.  Obtain a pointer to an object.  */
static inline __attribute__ ((nonnull (1))) void *
__alloc_buffer_alloc (struct alloc_buffer *buf, size_t size, size_t align)
{
  if (size == 1 && align == 1)
    return alloc_buffer_alloc_bytes (buf, size);

  size_t current = buf->__alloc_buffer_current;
  size_t aligned = roundup (current, align);
  size_t new_current = aligned + size;
  if (aligned >= current        /* No overflow in align step.  */
      && new_current >= size    /* No overflow in size computation.  */
      && new_current <= buf->__alloc_buffer_end) /* Room in buffer.  */
    {
      buf->__alloc_buffer_current = new_current;
      return (void *) aligned;
    }
  else
    {
      alloc_buffer_mark_failed (buf);
      return NULL;
    }
}

/* Obtain a TYPE * pointer to an object in BUF of TYPE.  Consume these
   bytes from the buffer.  Return NULL and mark the buffer as failed
   if there is not enough room in the buffer, or if the buffer has
   failed before.  */
#define alloc_buffer_alloc(buf, type)				\
  ((type *) __alloc_buffer_alloc				\
   (buf, __alloc_buffer_assert_size (sizeof (type)),		\
    __alloc_buffer_assert_align (__alignof__ (type))))

/* Internal function.  Obtain a pointer to an object which is
   subsequently added.  */
static inline const __attribute__ ((nonnull (1))) void *
__alloc_buffer_next (struct alloc_buffer *buf, size_t align)
{
  if (align == 1)
    return (const void *) buf->__alloc_buffer_current;

  size_t current = buf->__alloc_buffer_current;
  size_t aligned = roundup (current, align);
  if (aligned >= current        /* No overflow in align step.  */
      && aligned <= buf->__alloc_buffer_end) /* Room in buffer.  */
    {
      buf->__alloc_buffer_current = aligned;
      return (const void *) aligned;
    }
  else
    {
      alloc_buffer_mark_failed (buf);
      return NULL;
    }
}

/* Like alloc_buffer_alloc, but do not advance the pointer beyond the
   object (so a subseqent call to alloc_buffer_next or
   alloc_buffer_alloc returns the same pointer).  Note that the buffer
   is still aligned according to the requirements of TYPE, potentially
   consuming buffer space.  The effect of this function is similar to
   allocating a zero-length array from the buffer.

   It is possible to use the return pointer to write to the buffer and
   consume the written bytes using alloc_buffer_alloc_bytes (which
   does not change the buffer contents), but the calling code needs to
   perform manual length checks using alloc_buffer_size.  For example,
   to read as many int32_t values that are available in the input file
   and can fit into the remaining buffer space, you can use this:

     int32_t array = alloc_buffer_next (buf, int32_t);
     size_t ret = fread (array, sizeof (int32_t),
                         alloc_buffer_size (buf) / sizeof (int32_t), fp);
     if (ferror (fp))
       handle_error ();
     alloc_buffer_alloc_array (buf, int32_t, ret);

   The alloc_buffer_alloc_array call makes the actually-used part of
   the buffer permanent.  The remaining part of the buffer (not filled
   with data from the file) can be used for something else.

   This manual length checking can easily introduce errors, so this
   coding style is not recommended.  */
#define alloc_buffer_next(buf, type)				\
  ((type *) __alloc_buffer_next					\
   (buf, __alloc_buffer_assert_align (__alignof__ (type))))

/* Internal function.  Allocate an array.  */
void * __libc_alloc_buffer_alloc_array (struct alloc_buffer *buf,
					size_t size, size_t align,
					size_t count)
  __attribute__ ((nonnull (1)));

/* Obtain a TYPE * pointer to an array of COUNT objects in BUF of
   TYPE.  Consume these bytes from the buffer.  Return NULL and mark
   the buffer as failed if there is not enough room in the buffer,
   or if the buffer has failed before.  (Zero-length allocations from
   an empty buffer which has not yet failed succeed.)  */
#define alloc_buffer_alloc_array(buf, type, count)       \
  ((type *) __libc_alloc_buffer_alloc_array		 \
   (buf, __alloc_buffer_assert_size (sizeof (type)),	 \
    __alloc_buffer_assert_align (__alignof__ (type)),	 \
    count))

/* Internal function.  See alloc_buffer_copy_bytes below.  */
struct alloc_buffer __libc_alloc_buffer_copy_bytes (struct alloc_buffer,
						    const void *, size_t)
  __attribute__ ((nonnull (2)));

/* Copy SIZE bytes starting at SRC into the buffer.  If there is not
   enough room in the buffer, the buffer is marked as failed.  No
   alignment of the buffer is performed.  */
static inline __attribute__ ((nonnull (1, 2))) void
alloc_buffer_copy_bytes (struct alloc_buffer *buf, const void *src, size_t size)
{
  *buf = __libc_alloc_buffer_copy_bytes (*buf, src, size);
}

/* Internal function.  See alloc_buffer_copy_string below.  */
struct alloc_buffer __libc_alloc_buffer_copy_string (struct alloc_buffer,
						     const char *)
  __attribute__ ((nonnull (2)));
/* clang wants a hidden proto ahead of use. */
#ifndef _ISOMAC
libc_hidden_proto (__libc_alloc_buffer_copy_string)
#endif

/* Copy the string at SRC into the buffer, including its null
   terminator.  If there is not enough room in the buffer, the buffer
   is marked as failed.  Return a pointer to the string.  */
static inline __attribute__ ((nonnull (1, 2))) char *
alloc_buffer_copy_string (struct alloc_buffer *buf, const char *src)
{
  char *result = (char *) buf->__alloc_buffer_current;
  *buf = __libc_alloc_buffer_copy_string (*buf, src);
  if (alloc_buffer_has_failed (buf))
    result = NULL;
  return result;
}

#ifndef _ISOMAC
libc_hidden_proto (__libc_alloc_buffer_alloc_array)
libc_hidden_proto (__libc_alloc_buffer_allocate)
libc_hidden_proto (__libc_alloc_buffer_copy_bytes)
libc_hidden_proto (__libc_alloc_buffer_copy_string)
libc_hidden_proto (__libc_alloc_buffer_create_failure)
#endif

#endif /* _ALLOC_BUFFER_H */
