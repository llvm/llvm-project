/* Test if a memory region is wholly unwritable.  Mach version.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
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

#include <stdlib.h>
#include <stdint.h>
#include <mach.h>

/* Return 1 if the whole area PTR .. PTR+SIZE is not writable.
   Return -1 if it is writable.  */

int
__readonly_area (const char *ptr, size_t size)
{
  vm_address_t region_address = (uintptr_t) ptr;
  vm_size_t region_length = size;
  vm_prot_t protection;
  vm_prot_t max_protection;
  vm_inherit_t inheritance;
  boolean_t is_shared;
  mach_port_t object_name;
  vm_offset_t offset;

  while (__vm_region (__mach_task_self (),
		      &region_address, &region_length,
		      &protection, &max_protection, &inheritance, &is_shared,
		      &object_name, &offset) == KERN_SUCCESS
	 && region_address <= (uintptr_t) ptr)
    {
      region_address += region_length;
      if (region_address < (uintptr_t) ptr)
	continue;

      if (protection & VM_PROT_WRITE)
	return -1;

      if (region_address - (uintptr_t) ptr >= size)
	break;
    }

  return 1;
}
