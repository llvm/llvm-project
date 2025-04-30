/* __libc_siglongjmp for x86.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#define __libc_longjmp __redirect___libc_longjmp
#include <setjmp/longjmp.c>
#undef __libc_longjmp

extern void __longjmp_cancel (__jmp_buf __env, int __val)
     __attribute__ ((__noreturn__)) attribute_hidden;

/* Since __libc_longjmp is a private interface for cancellation
   implementation in libpthread, there is no need to restore shadow
   stack register.  */

void
__libc_longjmp (sigjmp_buf env, int val)
{
  /* Perform any cleanups needed by the frames being unwound.  */
  _longjmp_unwind (env, val);

  if (env[0].__mask_was_saved)
    /* Restore the saved signal mask.  */
    (void) __sigprocmask (SIG_SETMASK,
			  (sigset_t *) &env[0].__saved_mask,
			  (sigset_t *) NULL);

  /* Call the machine-dependent function to restore machine state
     without shadow stack.  */
  __longjmp_cancel (env[0].__jmpbuf, val ?: 1);
}
