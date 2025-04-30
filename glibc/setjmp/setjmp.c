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
#include <setjmp.h>


/* Save the current program position in ENV and return 0.  */
int
__libc_sigsetjmp (jmp_buf env, int savemask)
{
  /* Save the signal mask if requested.  */
  __sigjmp_save (env, savemask);

  __set_errno (ENOSYS);
  /* No way to signal failure.	*/
  return 0;
}

weak_alias (__libc_sigsetjmp, __sigsetjmp)
stub_warning (__sigsetjmp)
