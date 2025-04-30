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

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <support/blob_repeat.h>
#include <support/check.h>
#include <support/test-driver.h>
#include <support/support.h>
#include <support/xunistd.h>
#include <sys/mman.h>
#include <unistd.h>
#include <wchar.h>

/* Small allocations should use malloc directly instead of the mmap
   optimization because mappings carry a lot of overhead.  */
static const size_t maximum_small_size = 4 * 1024 * 1024;

/* Internal helper for fill.  */
static void
fill0 (char *target, const char *element, size_t element_size,
       size_t count)
{
  while (count > 0)
    {
      memcpy (target, element, element_size);
      target += element_size;
      --count;
    }
}

/* Fill the buffer at TARGET with COUNT copies of the ELEMENT_SIZE
   bytes starting at ELEMENT.  */
static void
fill (char *target, const char *element, size_t element_size,
      size_t count)
{
  if (element_size == 0 || count == 0)
    return;
  else if (element_size == 1)
    memset (target, element[0], count);
  else if (element_size == sizeof (wchar_t))
    {
      wchar_t wc;
      memcpy (&wc, element, sizeof (wc));
      wmemset ((wchar_t *) target, wc, count);
    }
  else if (element_size < 1024 && count > 4096)
    {
      /* Use larger copies for really small element sizes.  */
      char buffer[8192];
      size_t buffer_count = sizeof (buffer) / element_size;
      fill0 (buffer, element, element_size, buffer_count);
      while (count > 0)
        {
          size_t copy_count = buffer_count;
          if (copy_count > count)
            copy_count = count;
          size_t copy_bytes = copy_count * element_size;
          memcpy (target, buffer, copy_bytes);
          target += copy_bytes;
          count -= copy_count;
        }
    }
  else
    fill0 (target, element, element_size, count);
}

/* Use malloc instead of mmap for small allocations and unusual size
   combinations.  */
static struct support_blob_repeat
allocate_malloc (size_t total_size, const void *element, size_t element_size,
                 size_t count)
{
  void *buffer = malloc (total_size);
  if (buffer == NULL)
    return (struct support_blob_repeat) { 0 };
  fill (buffer, element, element_size, count);
  return (struct support_blob_repeat)
    {
      .start = buffer,
      .size = total_size,
      .use_malloc = true
    };
}

/* Return the least common multiple of PAGE_SIZE and ELEMENT_SIZE,
   avoiding overflow.  This assumes that PAGE_SIZE is a power of
   two.  */
static size_t
minimum_stride_size (size_t page_size, size_t element_size)
{
  TEST_VERIFY_EXIT (page_size > 0);
  TEST_VERIFY_EXIT (element_size > 0);

  /* Compute the number of trailing zeros common to both sizes.  */
  unsigned int common_zeros = __builtin_ctzll (page_size | element_size);

  /* In the product, this power of two appears twice, but in the least
     common multiple, it appears only once.  Therefore, shift one
     factor.  */
  size_t multiple;
  if (__builtin_mul_overflow (page_size >> common_zeros, element_size,
			      &multiple))
    return 0;
  return multiple;
}

/* Allocations larger than maximum_small_size potentially use mmap
   with alias mappings.  If SHARED, the alias mappings are created
   using MAP_SHARED instead of MAP_PRIVATE.  */
static struct support_blob_repeat
allocate_big (size_t total_size, const void *element, size_t element_size,
              size_t count, bool shared)
{
  unsigned long page_size = xsysconf (_SC_PAGESIZE);
  size_t stride_size = minimum_stride_size (page_size, element_size);
  if (stride_size == 0)
    {
      errno = EOVERFLOW;
      return (struct support_blob_repeat) { 0 };
    }

  /* Ensure that the stride size is at least maximum_small_size.  This
     is necessary to reduce the number of distinct mappings.  */
  if (stride_size < maximum_small_size)
    stride_size
      = ((maximum_small_size + stride_size - 1) / stride_size) * stride_size;

  if (stride_size > total_size)
    /* The mmap optimization would not save anything.  */
    return allocate_malloc (total_size, element, element_size, count);

  /* Reserve the memory region.  If we cannot create the mapping,
     there is no reason to set up the backing file.  */
  void *target = mmap (NULL, total_size, PROT_NONE,
                       MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  if (target == MAP_FAILED)
    return (struct support_blob_repeat) { 0 };

  /* Create the backing file for the repeated mapping.  Call mkstemp
     directly to remove the resources backing the temporary file
     immediately, once support_blob_repeat_free is called.  Using
     create_temp_file would result in a warning during post-test
     cleanup.  */
  int fd;
  {
    char *temppath = xasprintf ("%s/support_blob_repeat-XXXXXX", test_dir);
    fd = mkstemp (temppath);
    if (fd < 0)
      FAIL_EXIT1 ("mkstemp (\"%s\"): %m", temppath);
    xunlink (temppath);
    free (temppath);
  }

  /* Make sure that there is backing storage, so that the fill
     operation will not fault.  */
  if (posix_fallocate (fd, 0, stride_size) != 0)
    FAIL_EXIT1 ("posix_fallocate (%zu): %m", stride_size);

  /* The stride size must still be a multiple of the page size and
     element size.  */
  TEST_VERIFY_EXIT ((stride_size % page_size) == 0);
  TEST_VERIFY_EXIT ((stride_size % element_size) == 0);

  /* Fill the backing store.  */
  {
    void *ptr = mmap (target, stride_size, PROT_READ | PROT_WRITE,
                      MAP_FIXED | MAP_FILE | MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED)
      {
        int saved_errno = errno;
        xmunmap (target, total_size);
        xclose (fd);
        errno = saved_errno;
        return (struct support_blob_repeat) { 0 };
      }
    if (ptr != target)
      FAIL_EXIT1 ("mapping of %zu bytes moved from %p to %p",
                  stride_size, target, ptr);

    /* Write the repeating data.  */
    fill (target, element, element_size, stride_size / element_size);

    /* Return to a PROT_NONE mapping, just to be on the safe side.  */
    ptr = mmap (target, stride_size, PROT_NONE,
                MAP_FIXED | MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (ptr == MAP_FAILED)
      FAIL_EXIT1 ("Failed to reinstate PROT_NONE mapping: %m");
    if (ptr != target)
      FAIL_EXIT1 ("PROT_NONE mapping of %zu bytes moved from %p to %p",
                  stride_size, target, ptr);
  }

  /* Create the alias mappings.  */
  {
    size_t remaining_size = total_size;
    char *current = target;
    int flags = MAP_FIXED | MAP_FILE;
    if (shared)
      flags |= MAP_SHARED;
    else
      flags |= MAP_PRIVATE;
#ifdef MAP_NORESERVE
    flags |= MAP_NORESERVE;
#endif
    while (remaining_size > 0)
      {
        size_t to_map = stride_size;
        if (to_map > remaining_size)
          to_map = remaining_size;
        void *ptr = mmap (current, to_map, PROT_READ | PROT_WRITE,
                          flags, fd, 0);
        if (ptr == MAP_FAILED)
          {
            int saved_errno = errno;
            xmunmap (target, total_size);
            xclose (fd);
            errno = saved_errno;
            return (struct support_blob_repeat) { 0 };
          }
        if (ptr != current)
          FAIL_EXIT1 ("MAP_PRIVATE mapping of %zu bytes moved from %p to %p",
                      to_map, target, ptr);
        remaining_size -= to_map;
        current += to_map;
      }
  }

  xclose (fd);

  return (struct support_blob_repeat)
    {
      .start = target,
      .size = total_size,
      .use_malloc = false
    };
}

struct support_blob_repeat
repeat_allocate (const void *element, size_t element_size,
		 size_t count, bool shared)
{
  size_t total_size;
  if (__builtin_mul_overflow (element_size, count, &total_size))
    {
      errno = EOVERFLOW;
      return (struct support_blob_repeat) { 0 };
    }
  if (total_size <= maximum_small_size)
    return allocate_malloc (total_size, element, element_size, count);
  else
    return allocate_big (total_size, element, element_size, count, shared);
}

struct support_blob_repeat
support_blob_repeat_allocate (const void *element, size_t element_size,
                              size_t count)
{
  return repeat_allocate (element, element_size, count, false);
}

struct support_blob_repeat
support_blob_repeat_allocate_shared (const void *element, size_t element_size,
				     size_t count)
{
  return repeat_allocate (element, element_size, count, true);
}

void
support_blob_repeat_free (struct support_blob_repeat *blob)
{
  if (blob->size > 0)
    {
      int saved_errno = errno;
      if (blob->use_malloc)
        free (blob->start);
      else
        xmunmap (blob->start, blob->size);
      errno = saved_errno;
    }
  *blob = (struct support_blob_repeat) { 0 };
}
