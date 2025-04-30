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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <setjmp.h>

/* Save the current program position in ENV and return 0.  */
int
inhibit_stack_protector
#if defined BSD_SETJMP
# undef setjmp
# define savemask 1
setjmp (jmp_buf env)
#elif defined BSD__SETJMP
# undef _setjmp
# define savemask 0
_setjmp (jmp_buf env)
#else
__sigsetjmp (jmp_buf env, int savemask)
#endif
{
  /* Save data registers D1 through D7.  */
  asm volatile ("movem%.l %/d1-%/d7, %0"
		: : "m" (env[0].__jmpbuf[0].__dregs[0]));

  /* Save return address in place of register A0.  */
  env[0].__jmpbuf[0].__aregs[0] = __builtin_return_address (0);

  /* Save address registers A1 through A5.  */
  asm volatile ("movem%.l %/a1-%/a5, %0"
		: : "m" (env[0].__jmpbuf[0].__aregs[1]));

  /* Save caller's FP, not our own.  */
  env[0].__jmpbuf[0].__fp = *(int **) __builtin_frame_address (0);

  /* Save caller's SP, not our own.  */
  env[0].__jmpbuf[0].__sp = (int *) __builtin_frame_address (0) + 2;

#if defined __HAVE_68881__ || defined __HAVE_FPU__
  /* Save floating-point (68881) registers FP0 through FP7.  */
  asm volatile ("fmovem%.x %/fp0-%/fp7, %0"
		: : "m" (env[0].__jmpbuf[0].__fpregs[0]));
#elif defined (__mcffpu__)
  asm volatile ("fmovem %/fp0-%/fp7, %0"
		: : "m" (env[0].__jmpbuf[0].__fpregs[0]));
#endif

#if IS_IN (rtld)
  /* In ld.so we never save the signal mask.  */
  return 0;
#else
  /* Save the signal mask if requested.  */
  return __sigjmp_save (env, savemask);
#endif
}
#if !defined BSD_SETJMP && !defined BSD__SETJMP
libc_hidden_def (__sigsetjmp)
#endif
