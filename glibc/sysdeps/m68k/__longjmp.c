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
#include <stdlib.h>

/* Jump to the position specified by ENV, causing the
   setjmp call there to return VAL, or 1 if VAL is 0.  */
void
__longjmp (__jmp_buf env, int val)
{
  /* This restores the FP and SP that setjmp's caller had,
     and puts the return address into A0 and VAL into D0. */

#ifdef CHECK_SP
  CHECK_SP (env[0].__sp);
#endif

#if	defined(__HAVE_68881__) || defined(__HAVE_FPU__)
  /* Restore the floating-point registers.  */
  asm volatile("fmovem%.x %0, %/fp0-%/fp7" :
	       /* No outputs.  */ : "g" (env[0].__fpregs[0]));
#elif defined (__mcffpu__)
  asm volatile("fmovem %0, %/fp0-%/fp7" :
	       /* No outputs.  */ : "m" (env[0].__fpregs[0]));
#endif

  /* Put VAL in D0.  */
  asm volatile("move%.l %0, %/d0" : /* No outputs.  */ :
	       "g" (val == 0 ? 1 : val) : "d0");

  asm volatile(/* Restore the data and address registers.  */
	       "movem%.l %0, %/d1-%/d7/%/a0-%/a7\n"
	       /* Return to setjmp's caller.  */
#ifdef __motorola__
	       "jmp (%/a0)"
#else
	       "jmp %/a0@"
#endif
	       : /* No outputs.  */ : "g" (env[0].__dregs[0])
	       /* We don't bother with the clobbers,
		  because this code always jumps out anyway.  */
	       );

  /* Avoid `volatile function does return' warnings.  */
  for (;;);
}
