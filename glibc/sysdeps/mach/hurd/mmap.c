/* Copyright (C) 1994-2021 Free Software Foundation, Inc.
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
#include <hurd.h>
#include <hurd/fd.h>

/* Map addresses starting near ADDR and extending for LEN bytes.  from
   OFFSET into the file FD describes according to PROT and FLAGS.  If ADDR
   is nonzero, it is the desired mapping address.  If the MAP_FIXED bit is
   set in FLAGS, the mapping will be at ADDR exactly (which must be
   page-aligned); otherwise the system chooses a convenient nearby address.
   The return value is the actual mapping address chosen or (void *) -1
   for errors (in which case `errno' is set).  A successful `mmap' call
   deallocates any previous mapping for the affected region.  */

void *
__mmap (void *addr, size_t len, int prot, int flags, int fd, off_t offset)
{
  error_t err;
  vm_prot_t vmprot;
  memory_object_t memobj;
  vm_address_t mapaddr;

  mapaddr = (vm_address_t) addr;

  /* ADDR and OFFSET must be page-aligned.  */
  if ((mapaddr & (__vm_page_size - 1)) || (offset & (__vm_page_size - 1)))
    return (void *) (long int) __hurd_fail (EINVAL);

  vmprot = VM_PROT_NONE;
  if (prot & PROT_READ)
    vmprot |= VM_PROT_READ;
  if (prot & PROT_WRITE)
    vmprot |= VM_PROT_WRITE;
  if (prot & PROT_EXEC)
    vmprot |= VM_PROT_EXECUTE;

  switch (flags & MAP_TYPE)
    {
    default:
      return (void *) (long int) __hurd_fail (EINVAL);

    case MAP_ANON:
      memobj = MACH_PORT_NULL;
      break;

    case MAP_FILE:
    case 0:			/* Allow, e.g., just MAP_SHARED.  */
      {
	mach_port_t robj, wobj;
	if (err = HURD_DPORT_USE (fd, __io_map (port, &robj, &wobj)))
	  {
	    if (err == MIG_BAD_ID || err == EOPNOTSUPP || err == ENOSYS)
	      err = ENODEV;	/* File descriptor doesn't support mmap.  */
	    return (void *) (long int) __hurd_dfail (fd, err);
	  }
	switch (prot & (PROT_READ|PROT_WRITE))
	  {
	  /* Although it apparently doesn't make sense to map a file with
	     protection set to PROT_NONE, it is actually sometimes done.
	     In particular, that's how localedef reserves some space for
	     the locale archive file, the rationale being that some
	     implementations take into account whether the mapping is
	     anonymous or not when selecting addresses.  */
	  case PROT_NONE:
	  case PROT_READ:
	    memobj = robj;
	    if (wobj != MACH_PORT_NULL)
	      __mach_port_deallocate (__mach_task_self (), wobj);
	    break;
	  case PROT_WRITE:
	    memobj = wobj;
	    if (robj != MACH_PORT_NULL)
	      __mach_port_deallocate (__mach_task_self (), robj);
	    break;
	  case PROT_READ|PROT_WRITE:
	    if (robj == wobj)
	      {
		memobj = wobj;
		/* Remove extra reference.  */
		__mach_port_deallocate (__mach_task_self (), memobj);
	      }
	    else if (wobj == MACH_PORT_NULL /* Not writable by mapping.  */
		     && !(flags & MAP_SHARED))
	      /* The file can only be mapped for reading.  Since we are
		 making a private mapping, we will never try to write the
		 object anyway, so we don't care.  */
	      memobj = robj;
	    else
	      {
		__mach_port_deallocate (__mach_task_self (), wobj);
		return (void *) (long int) __hurd_fail (EACCES);
	      }
	    break;
	  default:
	    __builtin_unreachable ();
	  }
	break;
	/* XXX handle MAP_NOEXTEND */
      }
    }

  /* XXX handle MAP_INHERIT */

  err = __vm_map (__mach_task_self (),
		  &mapaddr, (vm_size_t) len, (vm_address_t) 0,
		  mapaddr == 0,
		  memobj, (vm_offset_t) offset,
		  ! (flags & MAP_SHARED),
		  vmprot, VM_PROT_ALL,
		  (flags & MAP_SHARED) ? VM_INHERIT_SHARE : VM_INHERIT_COPY);

  if (flags & MAP_FIXED)
    {
      if (err == KERN_NO_SPACE)
	{
	  /* XXX this is not atomic as it is in unix! */
	  /* The region is already allocated; deallocate it first.  */
	  err = __vm_deallocate (__mach_task_self (), mapaddr, len);
	  if (! err)
	    err = __vm_map (__mach_task_self (),
			    &mapaddr, (vm_size_t) len, (vm_address_t) 0,
			    0, memobj, (vm_offset_t) offset,
			    ! (flags & MAP_SHARED),
			    vmprot, VM_PROT_ALL,
			    (flags & MAP_SHARED) ? VM_INHERIT_SHARE
			    : VM_INHERIT_COPY);
	}
    }
  else
    {
      if (mapaddr != 0 && (err == KERN_NO_SPACE || err == KERN_INVALID_ADDRESS))
	err = __vm_map (__mach_task_self (),
			&mapaddr, (vm_size_t) len, (vm_address_t) 0,
			1, memobj, (vm_offset_t) offset,
			! (flags & MAP_SHARED),
			vmprot, VM_PROT_ALL,
			(flags & MAP_SHARED) ? VM_INHERIT_SHARE
			: VM_INHERIT_COPY);
    }

  if (memobj != MACH_PORT_NULL)
    __mach_port_deallocate (__mach_task_self (), memobj);

  if (err)
    return (void *) (long int) __hurd_fail (err);

  return (void *) mapaddr;
}

libc_hidden_def (__mmap)
weak_alias (__mmap, mmap)
