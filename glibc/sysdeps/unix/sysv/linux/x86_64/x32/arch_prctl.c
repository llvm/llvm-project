/* arch_prctl call for Linux/x32.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <sysdep.h>
#include <stdint.h>

/* Since x32 arch_prctl stores 32-bit base address of segment registers
   %fs and %gs as unsigned 64-bit value via ARCH_GET_FS and ARCH_GET_GS,
   we use an unsigned 64-bit variable to hold the base address and copy
   it to ADDR after the system call returns.  */

int
__arch_prctl (int code, uintptr_t *addr)
{
  int res;
  uint64_t addr64;
  void *prctl_arg = addr;

  switch (code)
    {
    case ARCH_GET_FS:
    case ARCH_GET_GS:
      prctl_arg = &addr64;
      break;
    }

  res = INLINE_SYSCALL (arch_prctl, 2, code, prctl_arg);
  if (res == 0)
    switch (code)
      {
      case ARCH_GET_FS:
      case ARCH_GET_GS:
	 /* Check for a large value that overflows.  */
	if ((uintptr_t) addr64 != addr64)
	  {
	    __set_errno (EOVERFLOW);
	    return -1;
	  }
	*addr = (uintptr_t) addr64;
	break;
      }

  return res;
}
weak_alias (__arch_prctl, arch_prctl)
