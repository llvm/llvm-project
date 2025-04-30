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

#ifndef SUPPORT_NEXT_TO_FAULT_H
#define SUPPORT_NEXT_TO_FAULT_H

#include <sys/cdefs.h>
#include <sys/types.h>

__BEGIN_DECLS

/* The memory region created by next_to_fault_allocate.  */
struct support_next_to_fault
{
  /* The user data.  */
  char *buffer;
  size_t length;

  /* The entire allocated region.  */
  void *region_start;
  size_t region_size;
};

/* Allocate a buffer of SIZE bytes just before a page which is mapped
   with PROT_NONE (so that overrunning the buffer will cause a
   fault).  */
struct support_next_to_fault support_next_to_fault_allocate (size_t size);

/* Deallocate the memory region allocated by
   next_to_fault_allocate.  */
void support_next_to_fault_free (struct support_next_to_fault *);

#endif /* SUPPORT_NEXT_TO_FAULT_H */
