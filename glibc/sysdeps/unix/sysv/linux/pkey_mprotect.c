/* mprotect with a memory protection key.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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
#include <sys/syscall.h>
#include <sysdep.h>

int
pkey_mprotect (void *addr, size_t len, int prot, int pkey)
{
  if (pkey == -1)
    /* If the key is -1, the system call is precisely equivalent to
       mprotect.  */
    return __mprotect (addr, len, prot);
  return INLINE_SYSCALL_CALL (pkey_mprotect, addr, len, prot, pkey);
}
