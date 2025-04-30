/* Copyright (C) 2007-2021 Free Software Foundation, Inc.
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

#include <sysdep.h>
#include <sys/mman.h>


int
posix_madvise (void *addr, size_t len, int advice)
{
  /* We have one problem: the kernel's MADV_DONTNEED does not
     correspond to POSIX's POSIX_MADV_DONTNEED.  The former simply
     discards changes made to the memory without writing it back to
     disk, if this would be necessary.  The POSIX behavior does not
     allow this.  There is no functionality mapping the POSIX behavior
     so far so we ignore that advice for now.  */
  if (advice == POSIX_MADV_DONTNEED)
    return 0;

  int result = INTERNAL_SYSCALL_CALL (madvise, addr, len, advice);
  return INTERNAL_SYSCALL_ERRNO (result);
}
