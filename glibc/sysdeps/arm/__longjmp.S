/* longjmp for ARM.
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

#include <sysdep.h>
#include <stap-probe.h>
#include <bits/setjmp.h>
#include <rtld-global-offsets.h>
#include <arm-features.h>

/* __longjmp(jmpbuf, val) */

ENTRY (__longjmp)
	mov	ip, r0

#ifdef CHECK_SP
	ldr	r4, [ip]	/* jmpbuf's sp */
	cfi_undefined (r4)
#ifdef PTR_DEMANGLE
	PTR_DEMANGLE (r4, r4, a3, a4)
#endif
	CHECK_SP (r4)
#endif

#ifdef PTR_DEMANGLE
	ldr	a4, [ip], #4
	PTR_DEMANGLE (a4, a4, a3, r4)
	cfi_undefined (r4)
	ldr	r4, [ip], #4
	PTR_DEMANGLE2 (r4, r4, a3)
#else
	ldr	a4, [ip], #4
	ldr	r4, [ip], #4
	cfi_undefined (r4)
#endif
	/* longjmp probe expects longjmp first argument (4@r0), second
	   argument (-4@r1), and target address (4@r4), respectively.  */
	LIBC_PROBE (longjmp, 3, 4@r0, -4@r1, 4@r4)
	mov	sp, a4
	mov	lr, r4
	ldmia	ip!, JMP_BUF_REGLIST
	cfi_restore (v1)
	cfi_restore (v2)
	cfi_restore (v3)
	cfi_restore (v4)
	cfi_restore (v5)
	cfi_restore (v6)
	cfi_restore (sl)
	cfi_restore (fp)
	cfi_restore (sp)
	cfi_restore (lr)

#if !defined ARM_ASSUME_NO_IWMMXT || defined __SOFTFP__
# define NEED_HWCAP 1
#endif

#ifdef NEED_HWCAP
# if IS_IN (rtld)
	LDST_PCREL (ldr, a4, a3, \
		    C_SYMBOL_NAME(_rtld_local_ro) \
		    + RTLD_GLOBAL_RO_DL_HWCAP_OFFSET)
# else
#  ifdef SHARED
	LDR_GLOBAL (a4, a3, C_SYMBOL_NAME(_rtld_global_ro), \
		    RTLD_GLOBAL_RO_DL_HWCAP_OFFSET)
#  else
	LDR_GLOBAL (a4, a3, C_SYMBOL_NAME(_dl_hwcap), 0)
#  endif
# endif
#endif

#ifdef __SOFTFP__
	tst	a4, #HWCAP_ARM_VFP
	beq	.Lno_vfp
#endif

	/* Restore the VFP registers.  */
	/* Following instruction is vldmia ip!, {d8-d15}.  */
	ldc	p11, cr8, [r12], #64
.Lno_vfp:

#ifndef ARM_ASSUME_NO_IWMMXT
	tst	a4, #HWCAP_ARM_IWMMXT
	beq	.Lno_iwmmxt

	/* Restore the call-preserved iWMMXt registers.  */
	/* Following instructions are wldrd wr10, [ip], #8 (etc.)  */
	ldcl	p1, cr10, [r12], #8
	ldcl	p1, cr11, [r12], #8
	ldcl	p1, cr12, [r12], #8
	ldcl	p1, cr13, [r12], #8
	ldcl	p1, cr14, [r12], #8
	ldcl	p1, cr15, [r12], #8
.Lno_iwmmxt:
#endif

	/* longjmp_target probe expects longjmp first argument (4@r0), second
	   argument (-4@r1), and target address (4@r14), respectively.  */
	LIBC_PROBE (longjmp_target, 3, 4@r0, -4@r1, 4@r14)

	movs	r0, r1		/* get the return value in place */
	it	eq
	moveq	r0, #1		/* can't let setjmp() return zero! */

	DO_RET(lr)

END (__longjmp)
