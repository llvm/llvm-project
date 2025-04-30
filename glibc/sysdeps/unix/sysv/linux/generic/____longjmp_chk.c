/* Copyright (C) 2011-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Chris Metcalf <cmetcalf@tilera.com>, 2011.

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

#include <jmpbuf-offsets.h>
#include <sysdep.h>
#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stackinfo.h>

#ifdef _STACK_GROWS_DOWN
#define called_from(this, saved) ((this) < (saved))
#else
#define called_from(this, saved) ((this) > (saved))
#endif

extern void ____longjmp_chk (__jmp_buf __env, int __val)
  __attribute__ ((__noreturn__));

void ____longjmp_chk (__jmp_buf env, int val)
{
  void *this_frame = __builtin_frame_address (0);
  void *saved_frame = JB_FRAME_ADDRESS (env);
  stack_t ss;

  /* If "env" is from a frame that called us, we're all set.  */
  if (called_from(this_frame, saved_frame))
    __longjmp (env, val);

  /* If we can't get the current stack state, give up and do the longjmp. */
  if (INTERNAL_SYSCALL_CALL (sigaltstack, NULL, &ss) != 0)
    __longjmp (env, val);

  /* If we we are executing on the alternate stack and within the
     bounds, do the longjmp.  */
  if (ss.ss_flags == SS_ONSTACK
      && (this_frame >= ss.ss_sp && this_frame < (ss.ss_sp + ss.ss_size)))
    __longjmp (env, val);

  __fortify_fail ("longjmp causes uninitialized stack frame");
}
