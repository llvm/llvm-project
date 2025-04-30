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
#include <mach.h>

/* Deallocate any mapping for the region starting at ADDR and extending LEN
   bytes.  Returns 0 if successful, -1 for errors (and sets errno).  */

int
__munmap (void *addr, size_t len)
{
  kern_return_t err;

  if (addr == 0)
    {
      errno = EINVAL;
      return -1;
    }

  if (err = __vm_deallocate (__mach_task_self (),
			     (vm_address_t) addr, (vm_size_t) len))
    {
      errno = err;
      return -1;
    }
  return 0;
}

libc_hidden_def (__munmap)
weak_alias (__munmap, munmap)
