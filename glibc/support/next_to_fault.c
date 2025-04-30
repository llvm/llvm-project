/* Memory allocation next to an unmapped page.
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

#include <support/check.h>
#include <support/next_to_fault.h>
#include <support/xunistd.h>
#include <sys/mman.h>
#include <sys/param.h>

struct support_next_to_fault
support_next_to_fault_allocate (size_t size)
{
  long page_size = sysconf (_SC_PAGE_SIZE);
  TEST_VERIFY_EXIT (page_size > 0);
  struct support_next_to_fault result;
  result.region_size = roundup (size, page_size) + page_size;
  if (size + page_size <= size || result.region_size <= size)
    FAIL_EXIT1 ("support_next_to_fault_allocate (%zu): overflow", size);
  result.region_start
    = xmmap (NULL, result.region_size, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1);
  /* Unmap the page after the allocation.  */
  xmprotect (result.region_start + (result.region_size - page_size),
             page_size, PROT_NONE);
  /* Align the allocation within the region so that it ends just
     before the PROT_NONE page.  */
  result.buffer = result.region_start + result.region_size - page_size - size;
  result.length = size;
  return result;
}

void
support_next_to_fault_free (struct support_next_to_fault *ntf)
{
  xmunmap (ntf->region_start, ntf->region_size);
  *ntf = (struct support_next_to_fault) { NULL, };
}
