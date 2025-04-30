/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
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
#include <sys/mman.h>
#include <sys/types.h>
#include <mach/port.h>

/* Map addresses starting near ADDR and extending for LEN bytes.  From
   OFFSET into the file FD describes according to PROT and FLAGS.  If ADDR
   is nonzero, it is the desired mapping address.  If the MAP_FIXED bit is
   set in FLAGS, the mapping will be at ADDR exactly (which must be
   page-aligned); otherwise the system chooses a convenient nearby address.
   The return value is the actual mapping address chosen or MAP_FAILED
   for errors (in which case `errno' is set).  A successful `mmap' call
   deallocates any previous mapping for the affected region.  */

void *
__mmap64 (void *addr, size_t len, int prot, int flags, int fd,
	  __off64_t offset)
{
  vm_offset_t small_offset = (vm_offset_t) offset;

  if (small_offset != offset)
    {
      /* We cannot do this since the offset is too large.  */
      __set_errno (EOVERFLOW);
      return MAP_FAILED;
    }

  return __mmap (addr, len, prot, flags, fd, small_offset);
}

libc_hidden_def (__mmap64)
weak_alias (__mmap64, mmap64)
