/* msync -- Synchronize mapped memory to external storage.  Mach version.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#include <sys/types.h>
#include <sys/mman.h>
#include <errno.h>
#include <sysdep-cancel.h>

#include <hurd/hurd.h>

/* Synchronize the region starting at ADDR and extending LEN bytes with the
   file it maps.  Filesystem operations on a file being mapped are
   unpredictable before this is done.  */

int
msync (void *addr, size_t length, int flags)
{
  boolean_t should_flush = flags & MS_INVALIDATE ? 1 : 0;
  boolean_t should_iosync = flags & MS_ASYNC ? 0 : 1;

  vm_address_t cur = (vm_address_t) addr;
  vm_address_t target = cur + length;

  vm_size_t len;
  vm_prot_t prot;
  vm_prot_t max_prot;
  vm_inherit_t inherit;
  boolean_t shared;
  memory_object_name_t obj;
  vm_offset_t offset;

  kern_return_t err;
  int cancel_oldtype;

  if (flags & (MS_SYNC | MS_ASYNC) == (MS_SYNC | MS_ASYNC))
    return __hurd_fail (EINVAL);

  while (cur < target)
    {
      vm_address_t begin = cur;

      err = __vm_region (__mach_task_self (),
			 &begin, &len, &prot, &max_prot, &inherit,
			 &shared, &obj, &offset);

      if (err != KERN_SUCCESS)
	return __hurd_fail (err);

      if (begin > cur)
	/* We were given an address before the first region,
	   or we found a hole.  */
	cur = begin;

      if (cur >= target)
	/* We were given an ending address within a hole. */
	break;

      if (MACH_PORT_VALID (obj))
	{
	  vm_size_t sync_len;

	  if (begin + len > target)
	    sync_len = target - begin;
	  else
	    sync_len = len;

	  cancel_oldtype = LIBC_CANCEL_ASYNC();
	  err = __vm_object_sync (obj, cur - begin + offset, sync_len,
				  should_flush, 1, should_iosync);
	  LIBC_CANCEL_RESET (cancel_oldtype);
	  __mach_port_deallocate (__mach_task_self (), obj);

	  if (err)
	    return __hurd_fail (err);

	}

      cur = begin + len;
    }

  return 0;
}
