/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
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
#include <stdlib.h>
#include <unistd.h>
#include <sysdep.h>
#include <abort-instr.h>


void
_exit (int status)
{
  while (1)
    {
      INLINE_SYSCALL (exit_group, 1, status);
      INLINE_SYSCALL (exit, 1, status);

#ifdef ABORT_INSTRUCTION
      ABORT_INSTRUCTION;
#endif
    }
}
libc_hidden_def (_exit)
rtld_hidden_def (_exit)
weak_alias (_exit, _Exit)
