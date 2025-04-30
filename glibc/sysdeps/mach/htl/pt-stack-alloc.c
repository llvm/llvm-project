/* Allocate a new stack.  Mach version.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library;  if not, see
   <https://www.gnu.org/licenses/>.  */

#include <errno.h>

#include <mach.h>
#include <mach/machine/vm_param.h>

#include <pt-internal.h>

/* The next address to use for stack allocation.  */
static vm_address_t next_stack_base = VM_MIN_ADDRESS;


/* Allocate a new stack of size STACKSIZE.  If successful, store the
   address of the newly allocated stack in *STACKADDR and return 0.
   Otherwise return an error code (EINVAL for an invalid stack size,
   EAGAIN if the system lacked the necessary resources to allocate a
   new stack).  */
int
__pthread_stack_alloc (void **stackaddr, size_t stacksize)
{
  vm_offset_t base;
  int i = 0;

get_stack:
  i++;
  for (base = next_stack_base;
       base < VM_MAX_ADDRESS
       && __vm_allocate (__mach_task_self (), &base,
			 stacksize, FALSE) != KERN_SUCCESS; base += stacksize)
    ;

  if (base >= VM_MAX_ADDRESS)
    {
      if (i == 1)
	{
	  next_stack_base = VM_MIN_ADDRESS;
	  goto get_stack;
	}
      else
	return EAGAIN;
    }

  if (base >= VM_MAX_ADDRESS)
    return EAGAIN;

  next_stack_base = base + stacksize;

  (*stackaddr) = (void *) base;
  return 0;
}
