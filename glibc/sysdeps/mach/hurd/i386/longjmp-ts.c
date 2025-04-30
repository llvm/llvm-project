/* Perform a `longjmp' on a Mach thread_state.  i386 version.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <hurd/signal.h>
#include <setjmp.h>
#include <jmpbuf-offsets.h>
#include <mach/thread_status.h>


/* Set up STATE to do the equivalent of `longjmp (ENV, VAL);'.  */

void
_hurd_longjmp_thread_state (void *state, jmp_buf env, int val)
{
  struct i386_thread_state *ts = state;

  ts->ebx = env[0].__jmpbuf[JB_BX];
  ts->esi = env[0].__jmpbuf[JB_SI];
  ts->edi = env[0].__jmpbuf[JB_DI];
  ts->ebp = env[0].__jmpbuf[JB_BP];
  ts->uesp = env[0].__jmpbuf[JB_SP];
  ts->eip = env[0].__jmpbuf[JB_PC];
  ts->eax = val ?: 1;
}
