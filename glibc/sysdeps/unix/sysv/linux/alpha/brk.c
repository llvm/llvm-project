/* Change data segment size.  Linux/Alpha.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <errno.h>
#include <unistd.h>
#include <sysdep.h>

void *__curbrk = 0;

int
__brk (void *addr)
{
  /* Alpha brk returns -ENOMEM in case of failure.  */
  __curbrk = (void *) INTERNAL_SYSCALL_CALL (brk, addr);
  if ((unsigned long) __curbrk == -ENOMEM)
    {
      __set_errno (ENOMEM);
      return -1;
    }

  return 0;
}
weak_alias (__brk, brk)
