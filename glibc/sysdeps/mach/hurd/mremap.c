/* Copyright (C) 2020-2021 Free Software Foundation, Inc.
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
#include <stdarg.h>
#include <hurd.h>

#include <stdio.h>

/* Remap pages mapped by the range [ADDR,ADDR+OLD_LEN) to new length
   NEW_LEN.  If MREMAP_MAYMOVE is set in FLAGS the returned address
   may differ from ADDR.  If MREMAP_FIXED is set in FLAGS the function
   takes another parameter which is a fixed address at which the block
   resides after a successful call.  */

void *
__mremap (void *addr, size_t old_len, size_t new_len, int flags, ...)
{
  error_t err;
  vm_address_t vm_addr = (vm_address_t) addr;
  vm_offset_t new_vm_addr = 0;

  vm_address_t begin = vm_addr;
  vm_address_t end;
  vm_size_t len;
  vm_prot_t prot;
  vm_prot_t max_prot;
  vm_inherit_t inherit;
  boolean_t shared;
  memory_object_name_t obj;
  vm_offset_t offset;

  if ((flags & ~(MREMAP_MAYMOVE | MREMAP_FIXED)) ||
      ((flags & MREMAP_FIXED) && !(flags & MREMAP_MAYMOVE)) ||
      (old_len == 0 && !(flags & MREMAP_MAYMOVE)))
    return (void *) (long int) __hurd_fail (EINVAL);

  if (flags & MREMAP_FIXED)
    {
      va_list arg;
      va_start (arg, flags);
      new_vm_addr = (vm_offset_t) va_arg (arg, void *);
      va_end (arg);
    }

  err = __vm_region (__mach_task_self (),
		     &begin, &len, &prot, &max_prot, &inherit,
		     &shared, &obj, &offset);
  if (err)
    return (void *) (uintptr_t) __hurd_fail (err);

  if (begin > vm_addr)
    {
      err = EFAULT;
      goto out;
    }

  if (begin < vm_addr || (old_len != 0 && old_len != len))
    {
      err = EINVAL;
      goto out;
    }

  end = begin + len;

  if ((flags & MREMAP_FIXED) &&
      ((new_vm_addr + new_len > vm_addr && new_vm_addr < end)))
    {
    /* Overlapping is not supported, like in Linux.  */
      err = EINVAL;
      goto out;
    }

  /* FIXME: locked memory.  */

  if (old_len != 0 && !(flags & MREMAP_FIXED))
    {
      /* A mere change of the existing map.  */

      if (new_len == len)
	{
	  new_vm_addr = vm_addr;
	  goto out;
	}

      if (new_len < len)
	{
	  /* Shrink.  */
	  __mach_port_deallocate (__mach_task_self (), obj);
	  err = __vm_deallocate (__mach_task_self (),
				 begin + new_len, len - new_len);
	  new_vm_addr = vm_addr;
	  goto out;
	}

      /* Try to expand.  */
      err = __vm_map (__mach_task_self (),
		      &end, new_len - len, 0, 0,
		      obj, offset + len, 0, prot, max_prot, inherit);
      if (!err)
	{
	  /* Ok, that worked.  Now coalesce them.  */
	  new_vm_addr = vm_addr;

	  /* XXX this is not atomic as it is in unix! */
	  err = __vm_deallocate (__mach_task_self (), begin, new_len);
	  if (err)
	    {
	      __vm_deallocate (__mach_task_self (), end, new_len - len);
	      goto out;
	    }

	  err = __vm_map (__mach_task_self (),
			  &begin, new_len, 0, 0,
			  obj, offset, 0, prot, max_prot, inherit);
	  if (err)
	    {
	      /* Oops, try to remap before reporting.  */
	      __vm_map (__mach_task_self (),
			&begin, len, 0, 0,
			obj, offset, 0, prot, max_prot, inherit);
	    }

	  goto out;
	}
    }

  if (!(flags & MREMAP_MAYMOVE))
    {
      /* Can not map here */
      err = ENOMEM;
      goto out;
    }

  err = __vm_map (__mach_task_self (),
		  &new_vm_addr, new_len, 0,
		  new_vm_addr == 0, obj, offset,
		  old_len == 0, prot, max_prot, inherit);

  if (err == KERN_NO_SPACE && (flags & MREMAP_FIXED))
    {
      /* XXX this is not atomic as it is in unix! */
      /* The region is already allocated; deallocate it first.  */
      err = __vm_deallocate (__mach_task_self (), new_vm_addr, new_len);
      if (! err)
	err = __vm_map (__mach_task_self (),
			&new_vm_addr, new_len, 0,
			0, obj, offset,
			old_len == 0, prot, max_prot, inherit);
    }

  if (!err)
    /* Alright, can remove old mapping.  */
    __vm_deallocate (__mach_task_self (), begin, len);

out:
  __mach_port_deallocate (__mach_task_self (), obj);
  if (err)
    return (void *) (uintptr_t) __hurd_fail (err);
  return (void *) new_vm_addr;
}

libc_hidden_def (__mremap)
weak_alias (__mremap, mremap)
