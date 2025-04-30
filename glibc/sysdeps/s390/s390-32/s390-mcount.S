/* S/390-specific implementation of profiling support.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Martin Schwidefsky (schwidefsky@de.ibm.com)

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

#include <sysdep.h>

/*
 * How profiling works on S/390:
 * On the start of each function _mcount is called with the address of a
 * data word in %r1 (initialized to 0, used for counting). The compiler
 * with the option -p generates code of the form:
 *
 *         STM    6,15,24(15)
 *         BRAS   13,.LTN0_0
 * .LT0_0:
 * .LC12:  .long  _mcount
 * .LC13:  .long  .LP0
 *         .data
 *         .align 4
 * .LP0:   .long  0
 *         .text
 * # function profiler
 *         st     14,4(15)
 *         l      14,.LC12-.LT0_0(13)
 *         l      1,.LC13-.LT0_0(13)
 *         basr   14,14
 *         l      14,4(15)
 *
 * The _mcount implementation now has to call __mcount_internal with the
 * address of .LP0 as first parameter and the return address as second
 * parameter. &.LP0 was loaded to %r1 and the return address is in %r14.
 * _mcount may not modify any register.
 */

	.globl C_SYMBOL_NAME(_mcount)
	.type C_SYMBOL_NAME(_mcount), @function
	cfi_startproc
	.align ALIGNARG(4)
C_LABEL(_mcount)
	/* Save the caller-clobbered registers.  */
	ahi   %r15,-128
	cfi_adjust_cfa_offset (128)
	/* binutils 2.28+: .cfi_val_offset r15, -96 */
	.cfi_escape \
		/* DW_CFA_val_offset */ 0x14, \
		/* r15 */               0x0f, \
		/* scaled offset */     0x18
	stm   %r14,%r5,96(%r15)
	cfi_offset (r14, -128)
	l     %r2,132(%r15)       # callers address  = first parameter
	la    %r2,0(%r2)          # clear bit 0
	la    %r3,0(%r14)         # callees address  = second parameter

#ifdef PIC
	bras  %r14,0f
	.long _GLOBAL_OFFSET_TABLE_-.
0:	al    %r14,0(%r14)
	l     %r14,__mcount_internal@GOT(%r14)
#else
	bras  %r14,0f
	.long __mcount_internal
0:      l     %r14,0(%r14)
#endif
	basr  %r14,%r14

	/*
	 * Pop the saved registers.  Please note that `mcount' has no
	 * return value.
	 */
	lm    %r14,%r5,96(%r15)
	ahi   %r15,128
	cfi_adjust_cfa_offset (-128)
	br    %r14
	cfi_endproc
	ASM_SIZE_DIRECTIVE(C_SYMBOL_NAME(_mcount))

#undef mcount
weak_alias (_mcount, mcount)
