/* 64 bit S/390-specific implementation of profiling support.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   Contributed by Martin Schwidefsky (schwidefsky@de.ibm.com)
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

#include <sysdep.h>

/* How profiling works on 64 bit S/390:
   On the start of each function _mcount is called with the address of a
   data word in %r1 (initialized to 0, used for counting). The compiler
   with the option -p generates code of the form:

           STM    6,15,24(15)
           BRAS   13,.LTN0_0
   .LT0_0:
   .LC13:  .long  .LP0
           .data
           .align 4
   .LP0:   .long  0
           .text
   # function profiler
           stg    14,8(15)
           lg     1,.LC13-.LT0_0(13)
           brasl  14,_mcount
           lg     14,8(15)

   The _mcount implementation now has to call __mcount_internal with the
   address of .LP0 as first parameter and the return address as second
   parameter. &.LP0 was loaded to %r1 and the return address is in %r14.
   _mcount may not modify any register.

   Alternatively, at the start of each function __fentry__ is called using a
   single

   # function profiler
           brasl  0,__fentry__

   instruction.  In this case %r0 points to the callee, and %r14 points to the
   caller.  These values need to be passed to __mcount_internal using the same
   sequence as for _mcount, so the code below is shared between both functions.
   The only major difference is that __fentry__ cannot return through %r0, in
   which the return address is located, because br instruction is a no-op with
   this register.  Therefore %r1, which is clobbered by the PLT anyway, is
   used.  */

#define xglue(x, y) x ## y
#define glue(x, y) xglue(x, y)

	.globl C_SYMBOL_NAME(MCOUNT_SYMBOL)
	.type C_SYMBOL_NAME(MCOUNT_SYMBOL), @function
	cfi_startproc
	.align ALIGNARG(4)
C_LABEL(MCOUNT_SYMBOL)
	cfi_return_column (glue(r, MCOUNT_CALLEE_REG))
	/* Save the caller-clobbered registers.  */
	aghi  %r15,-224
	cfi_adjust_cfa_offset (224)
	/* binutils 2.28+: .cfi_val_offset r15, -160 */
	.cfi_escape \
		/* DW_CFA_val_offset */ 0x14, \
		/* r15 */               0x0f, \
		/* scaled offset */     0x14
	stmg  %r14,%r5,160(%r15)
	cfi_offset (r14, -224)
	cfi_offset (r0, -224+16)
	lg    %r2,MCOUNT_CALLER_OFF(%r15)	# callers address  = 1st param
	lgr   %r3,glue(%r, MCOUNT_CALLEE_REG)	# callees address  = 2nd param

#ifdef PIC
	brasl %r14,__mcount_internal@PLT
#else
	brasl %r14,__mcount_internal
#endif

	/* Pop the saved registers.  Please note that `mcount' has no
	   return value.  */
	lmg   %r14,%r5,160(%r15)
	aghi  %r15,224
	cfi_adjust_cfa_offset (-224)
#if MCOUNT_RETURN_REG != MCOUNT_CALLEE_REG
	lgr   glue(%r, MCOUNT_RETURN_REG),glue(%r, MCOUNT_CALLEE_REG)
#endif
	br    glue(%r, MCOUNT_RETURN_REG)
	cfi_endproc
	ASM_SIZE_DIRECTIVE(C_SYMBOL_NAME(MCOUNT_SYMBOL))
