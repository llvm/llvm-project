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

/* Versioned copy of sysdeps/generic/longjmp.c modified for AltiVec support. */

#include  <shlib-compat.h>
#include <stddef.h>
#include <setjmp.h>
#include <signal.h>

extern void __vmx__longjmp (__jmp_buf __env, int __val)
     __attribute__ ((noreturn));
extern void __vmx__libc_longjmp (sigjmp_buf env, int val)
     __attribute__ ((noreturn));
libc_hidden_proto (__vmx__libc_longjmp)

/* Set the signal mask to the one specified in ENV, and jump
   to the position specified in ENV, causing the setjmp
   call there to return VAL, or 1 if VAL is 0.  */
void
__vmx__libc_siglongjmp (sigjmp_buf env, int val)
{
  /* Perform any cleanups needed by the frames being unwound.  */
  _longjmp_unwind (env, val);

  if (env[0].__mask_was_saved)
    /* Restore the saved signal mask.  */
    (void) __sigprocmask (SIG_SETMASK, &env[0].__saved_mask,
			  (sigset_t *) NULL);

  /* Call the machine-dependent function to restore machine state.  */
  __vmx__longjmp (env[0].__jmpbuf, val ?: 1);
}

strong_alias (__vmx__libc_siglongjmp, __vmx__libc_longjmp)
libc_hidden_def (__vmx__libc_longjmp)

strong_alias (__vmx__libc_longjmp, __libc_longjmp)
strong_alias (__vmx__libc_siglongjmp, __libc_siglongjmp)
versioned_symbol (libc, __vmx__libc_siglongjmp, _longjmp, GLIBC_2_3_4);
versioned_symbol (libc, __vmx__libc_siglongjmp, longjmp, GLIBC_2_3_4);
versioned_symbol (libc, __vmx__libc_siglongjmp, siglongjmp, GLIBC_2_3_4);
