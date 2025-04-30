/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Martin Schwidefsky (schwidefsky@de.ibm.com).

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
#include <sysdep.h>
#include <setjmp.h>
#include <bits/setjmp.h>
#include <stdlib.h>
#include <unistd.h>
#include <stap-probe.h>

/* Jump to the position specified by ENV, causing the
   setjmp call there to return VAL, or 1 if VAL is 0.  */
void
__longjmp (__jmp_buf env, int val)
{
#ifdef PTR_DEMANGLE
  uintptr_t guard = THREAD_GET_POINTER_GUARD ();
# ifdef CHECK_SP
  CHECK_SP (env, guard);
# endif
#elif defined CHECK_SP
  CHECK_SP (env, 0);
#endif
  register int r2 __asm__ ("%r2") = val == 0 ? 1 : val;
#ifdef PTR_DEMANGLE
  register uintptr_t r3 __asm__ ("%r3") = guard;
  register void *r1 __asm__ ("%r1") = (void *) env;
#endif
  /* Restore registers and jump back.  */
  __asm__ __volatile__ (
		/* longjmp probe expects longjmp first argument, second
		   argument and target address.  */
#ifdef PTR_DEMANGLE
			"lm   %%r4,%%r5,32(%1)\n\t"
			"xr   %%r4,%2\n\t"
			"xr   %%r5,%2\n\t"
			LIBC_PROBE_ASM (longjmp, 4@%1 -4@%0 4@%%r4)
#else
			LIBC_PROBE_ASM (longjmp, 4@%1 -4@%0 4@%%r14)
#endif

			/* restore fpregs  */
			"ld   %%f6,48(%1)\n\t"
			"ld   %%f4,40(%1)\n\t"

			/* restore gregs and return to jmp_buf target  */
#ifdef PTR_DEMANGLE
			"lm   %%r6,%%r13,0(%1)\n\t"
			"lr   %%r15,%%r5\n\t"
			LIBC_PROBE_ASM (longjmp_target, 4@%1 -4@%0 4@%%r4)
			"br   %%r4"
#else
			"lm   %%r6,%%r15,0(%1)\n\t"
			LIBC_PROBE_ASM (longjmp_target, 4@%1 -4@%0 4@%%r14)
			"br   %%r14"
#endif
			: : "r" (r2),
#ifdef PTR_DEMANGLE
			  "r" (r1), "r" (r3)
#else
			  "a" (env)
#endif
			);

  /* Avoid `volatile function does return' warnings.  */
  for (;;);
}
