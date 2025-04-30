/* longjmp for PA-RISC.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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
#ifdef CHECK_SP
  CHECK_SP (env[0].__jmp_buf.__sp);
#endif

  /* We must use one of the non-callee saves registers
     for env.  */
  register unsigned long r26 asm ("r26") = (unsigned long)&env[0];
  register unsigned long r25 asm ("r25") = (unsigned long)(val == 0 ? 1 : val);

  asm volatile(
	/* Set return value.  */
	"copy	%0, %%r28\n\t"
	/* Load callee saves from r3 to r18.  */
	"ldw	0(%1), %%r3\n\t"
	"ldw	8(%1), %%r4\n\t"
	"ldw	12(%1), %%r5\n\t"
	"ldw	16(%1), %%r6\n\t"
	"ldw	20(%1), %%r7\n\t"
	"ldw	24(%1), %%r8\n\t"
	"ldw	28(%1), %%r9\n\t"
	"ldw	32(%1), %%r10\n\t"
	"ldw	36(%1), %%r11\n\t"
	"ldw	40(%1), %%r12\n\t"
	"ldw	44(%1), %%r13\n\t"
	"ldw	48(%1), %%r14\n\t"
	"ldw	52(%1), %%r15\n\t"
	"ldw	56(%1), %%r16\n\t"
	"ldw	60(%1), %%r17\n\t"
	"ldw	64(%1), %%r18\n\t"
	/* Load PIC register.  */
	"ldw	68(%1), %%r19\n\t"
	/* Load static link register.  */
	"ldw	72(%1), %%r27\n\t"
	/* Load stack pointer.  */
	"ldw	76(%1), %%r30\n\t"
	/* Load return pointer. */
	"ldw	80(%1), %%rp\n\t"
	/* Ues a spare caller saves register.  */
	"ldo	88(%1),%%r25\n\t"
	/* Load callee saves from fr12 to fr21.  */
	"fldds,ma 8(%%r25), %%fr12\n\t"
	"fldds,ma 8(%%r25), %%fr13\n\t"
	"fldds,ma 8(%%r25), %%fr14\n\t"
	"fldds,ma 8(%%r25), %%fr15\n\t"
	"fldds,ma 8(%%r25), %%fr16\n\t"
	"fldds,ma 8(%%r25), %%fr17\n\t"
	"fldds,ma 8(%%r25), %%fr18\n\t"
	"fldds,ma 8(%%r25), %%fr19\n\t"
	"fldds,ma 8(%%r25), %%fr20\n\t"
	"fldds	 0(%%r25), %%fr21\n\t"
	/* Jump back to stored return address.  */
	"bv,n	%%r0(%%r2)\n\t"
	: /* No outputs.  */
	: "r" (r25), "r" (r26)
	: /* No point in clobbers.  */ );

  /* Avoid `volatile function does return' warnings.  */
  for (;;);
}
