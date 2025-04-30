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

/* Change the memory protection of the region starting at ADDR and
   extending LEN bytes to PROT.  Returns 0 if successful, -1 for errors
   (and sets errno).  */

int
__mprotect (void *addr, size_t len, int prot)
{
  kern_return_t err;
  vm_prot_t vmprot;

  vmprot = VM_PROT_NONE;
  if (prot & PROT_READ)
    vmprot |= VM_PROT_READ;
  if (prot & PROT_WRITE)
    vmprot |= VM_PROT_WRITE;
  if (prot & PROT_EXEC)
    vmprot |= VM_PROT_EXECUTE;

  if (err = __vm_protect (__mach_task_self (),
			  (vm_address_t) addr, (vm_size_t) len,
			  0, vmprot))
    {
      errno = err;
      return -1;
    }
  return 0;
}
libc_hidden_def (__mprotect)
weak_alias (__mprotect, mprotect)
