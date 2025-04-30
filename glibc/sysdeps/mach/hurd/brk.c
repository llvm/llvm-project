/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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
#include <hurd.h>
#include <hurd/resource.h>
#include <lock-intern.h>	/* For `struct mutex'.  */
#include <vm_param.h>


/* Initial maximum size of the data segment (this is arbitrary).  */
#define	DATA_SIZE	(128 * 1024 * 1024)

/* Up to the page including this address is allocated from the kernel.
   This address is the data resource limit.  */
vm_address_t _hurd_data_end;

/* Up to this address is actually available to the user.
   Pages beyond the one containing this address allow no access.  */
vm_address_t _hurd_brk = 0;

/* This name is used by the Linux crtbeginS.o for reasons you don't even
   want to think about it.  It's just easier to provide some definition for
   it than even to explain the braindamage involved.  */
weak_alias (_hurd_brk, ___brk_addr)

struct mutex _hurd_brk_lock;

static vm_address_t brk_start;


/* Set the end of the process's data space to INADDR.
   Return 0 if successful, -1 if not.  */
int
__brk (void *inaddr)
{
  int ret;
  HURD_CRITICAL_BEGIN;
  __mutex_lock (&_hurd_brk_lock);
  ret = _hurd_set_brk ((vm_address_t) inaddr);
  __mutex_unlock (&_hurd_brk_lock);
  HURD_CRITICAL_END;
  return ret;
}
weak_alias (__brk, brk)


int
_hurd_set_brk (vm_address_t addr)
{
  error_t err = 0;
  vm_address_t pagend = round_page (addr);
  vm_address_t pagebrk = round_page (_hurd_brk);
  long int rlimit;

  if (pagend <= pagebrk)
    {
      if (pagend < pagebrk)
	{
	  /* XXX wish this were atomic... */
	  /* First deallocate the memory to release its backing space.  */
	  __vm_deallocate (__mach_task_self (), pagend, pagebrk - pagend);
	  /* Now reallocate it with no access allowed.  */
	  err = __vm_map (__mach_task_self (),
			  &pagend, pagebrk - pagend,
			  0, 0, MACH_PORT_NULL, 0, 0,
			  0, VM_PROT_READ|VM_PROT_WRITE|VM_PROT_EXECUTE,
			  VM_INHERIT_COPY);
	  /* XXX what if error? */
	}
      _hurd_brk = addr;
      return 0;
    }

  __mutex_lock (&_hurd_rlimit_lock);
  rlimit = _hurd_rlimits[RLIMIT_DATA].rlim_cur;
  __mutex_unlock (&_hurd_rlimit_lock);

  if (addr - brk_start > rlimit)
    {
      /* Need to increase the resource limit.  */
      errno = ENOMEM;
      return -1;
    }

  if (pagend > _hurd_data_end)
    {
      vm_address_t alloc_start = _hurd_data_end;

      /* We didn't allocate enough space!  Hopefully we can get some more!  */

      if (_hurd_data_end > pagebrk)
	/* First finish allocation.  */
	err = __vm_protect (__mach_task_self (), pagebrk,
			    alloc_start - pagebrk, 0,
			    VM_PROT_READ|VM_PROT_WRITE|VM_PROT_EXECUTE);
      if (! err)
	_hurd_brk = alloc_start;

      if (! err)
	err = __vm_allocate (__mach_task_self (), &alloc_start,
			     pagend - alloc_start, 0);

      if (! err)
	_hurd_data_end = pagend;
    }
  else
    /* Make the memory accessible.  */
    err = __vm_protect (__mach_task_self (), pagebrk, pagend - pagebrk,
			0, VM_PROT_READ|VM_PROT_WRITE|VM_PROT_EXECUTE);

  if (err)
    return __hurd_fail (err);

  _hurd_brk = addr;
  return 0;
}

static void
init_brk (void)
{
  vm_address_t pagend;

  __mutex_init (&_hurd_brk_lock);

  brk_start = (vm_address_t) BRK_START;

  /* If _hurd_brk is already set, don't change it.  The assumption is that
     it was set in a previous run before something like Emacs's unexec was
     called and dumped all the data up to the break at that point.  */
  if (_hurd_brk == 0) {
    _hurd_brk = (vm_address_t) BRK_START;
  }

  pagend = round_page (_hurd_brk);

  _hurd_data_end = round_page (brk_start + DATA_SIZE);

  if (pagend < _hurd_data_end)
    {
      /* We use vm_map to allocate and change permissions atomically.  */
      if (__vm_map (__mach_task_self (), &pagend, _hurd_data_end - pagend,
		    0, 0, MACH_PORT_NULL, 0, 0,
		    0, VM_PROT_READ|VM_PROT_WRITE|VM_PROT_EXECUTE,
		    VM_INHERIT_COPY))
	/* Couldn't allocate the memory.  The break will be very short.  */
	_hurd_data_end = pagend;
    }

  (void) &init_brk;		/* Avoid ``defined but not used'' warning.  */
}
text_set_element (_hurd_preinit_hook, init_brk);
