/* Allocate a memory region shared across processes.
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

#include <errno.h>
#include <stddef.h>
#include <support/support.h>
#include <support/xunistd.h>
#include <sys/mman.h>

/* Header for the allocation.  It contains the size of the allocation
   for subsequent unmapping.  */
struct header
{
  size_t total_size;
  char data[] __attribute__ ((aligned (__alignof__ (max_align_t))));
};

void *
support_shared_allocate (size_t size)
{
  size_t total_size = size + offsetof (struct header, data);
  if (total_size < size)
    {
      errno = ENOMEM;
      oom_error (__func__, size);
      return NULL;
    }
  else
    {
      struct header *result = xmmap (NULL, total_size, PROT_READ | PROT_WRITE,
                                     MAP_ANONYMOUS | MAP_SHARED, -1);
      result->total_size = total_size;
      return &result->data;
    }
}

void
support_shared_free (void *data)
{
  struct header *header = data - offsetof (struct header, data);
  xmunmap (header, header->total_size);
}
