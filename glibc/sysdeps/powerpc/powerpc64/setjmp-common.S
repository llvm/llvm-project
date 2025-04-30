/* setjmp for PowerPC64.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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
#include <stap-probe.h>
#define _ASM
#ifdef __NO_VMX__
#include <novmxsetjmp.h>
#else
#include <jmpbuf-offsets.h>
#endif

#ifndef __NO_VMX__
	.section	".toc","aw"
.LC__dl_hwcap:
# ifdef SHARED
#  if IS_IN (rtld)
	/* Inside ld.so we use the local alias to avoid runtime GOT
	   relocations.  */
	.tc _rtld_local_ro[TC],_rtld_local_ro
#  else
	.tc _rtld_global_ro[TC],_rtld_global_ro
#  endif
# else
	.tc _dl_hwcap[TC],_dl_hwcap
# endif
	.section ".text"
#endif

	.machine	"altivec"
ENTRY (setjmp_symbol)
	CALL_MCOUNT 1
	li r4,1			/* Set second argument to 1.  */
	b JUMPTARGET (GLUE(__sigsetjmp_symbol,_ent))
END (setjmp_symbol)

#if defined SHARED && !IS_IN (rtld) && !defined __NO_VMX__
/* When called from within libc we need a special version of _setjmp
   that saves r2 since the call won't go via a plt call stub.  See
   bugz #269.  __GI__setjmp is used in csu/libc-start.c when
   HAVE_CLEANUP_JMP_BUF is defined.  */
ENTRY (__GI__setjmp)
	std r2,FRAME_TOC_SAVE(r1)		/* Save the callers TOC in the save area.  */
	CALL_MCOUNT 1
	li r4,0			/* Set second argument to 0.  */
	b JUMPTARGET (GLUE(__sigsetjmp_symbol,_ent))
END (__GI__setjmp)
#endif

ENTRY (_setjmp_symbol)
	CALL_MCOUNT 1
	li r4,0			/* Set second argument to 0.  */
	b JUMPTARGET (GLUE(__sigsetjmp_symbol,_ent))
END (_setjmp_symbol)
libc_hidden_def (_setjmp_symbol)

ENTRY (__sigsetjmp_symbol)
	CALL_MCOUNT 2
JUMPTARGET(GLUE(__sigsetjmp_symbol,_ent)):
#ifdef PTR_MANGLE
	mr   r5, r1
	PTR_MANGLE (r5, r6)
	std  r5,(JB_GPR1*8)(3)
#else
	std  r1,(JB_GPR1*8)(3)
#endif
	mflr r0
#if defined SHARED && !IS_IN (rtld)
	ld   r5,FRAME_TOC_SAVE(r1)	/* Retrieve the callers TOC.  */
	std  r5,(JB_GPR2*8)(3)
#else
	std  r2,(JB_GPR2*8)(3)
#endif
	/* setjmp probe expects longjmp first argument (8@3), second argument
	   (-4@4), and target address (8@0), respectively.  */
	LIBC_PROBE (setjmp, 3, 8@3, -4@4, 8@0)
	std  r14,((JB_GPRS+0)*8)(3)
	stfd fp14,((JB_FPRS+0)*8)(3)
#ifdef PTR_MANGLE
	PTR_MANGLE2 (r0, r6)
#endif
	std  r0,(JB_LR*8)(3)
	std  r15,((JB_GPRS+1)*8)(3)
	stfd fp15,((JB_FPRS+1)*8)(3)
	mfcr r0
	std  r16,((JB_GPRS+2)*8)(3)
	stfd fp16,((JB_FPRS+2)*8)(3)
	stw  r0,((JB_CR*8)+4)(3)	/* 32-bit CR.  */
	std  r17,((JB_GPRS+3)*8)(3)
	stfd fp17,((JB_FPRS+3)*8)(3)
	std  r18,((JB_GPRS+4)*8)(3)
	stfd fp18,((JB_FPRS+4)*8)(3)
	std  r19,((JB_GPRS+5)*8)(3)
	stfd fp19,((JB_FPRS+5)*8)(3)
	std  r20,((JB_GPRS+6)*8)(3)
	stfd fp20,((JB_FPRS+6)*8)(3)
	std  r21,((JB_GPRS+7)*8)(3)
	stfd fp21,((JB_FPRS+7)*8)(3)
	std  r22,((JB_GPRS+8)*8)(3)
	stfd fp22,((JB_FPRS+8)*8)(3)
	std  r23,((JB_GPRS+9)*8)(3)
	stfd fp23,((JB_FPRS+9)*8)(3)
	std  r24,((JB_GPRS+10)*8)(3)
	stfd fp24,((JB_FPRS+10)*8)(3)
	std  r25,((JB_GPRS+11)*8)(3)
	stfd fp25,((JB_FPRS+11)*8)(3)
	std  r26,((JB_GPRS+12)*8)(3)
	stfd fp26,((JB_FPRS+12)*8)(3)
	std  r27,((JB_GPRS+13)*8)(3)
	stfd fp27,((JB_FPRS+13)*8)(3)
	std  r28,((JB_GPRS+14)*8)(3)
	stfd fp28,((JB_FPRS+14)*8)(3)
	std  r29,((JB_GPRS+15)*8)(3)
	stfd fp29,((JB_FPRS+15)*8)(3)
	std  r30,((JB_GPRS+16)*8)(3)
	stfd fp30,((JB_FPRS+16)*8)(3)
	std  r31,((JB_GPRS+17)*8)(3)
	stfd fp31,((JB_FPRS+17)*8)(3)
#ifndef __NO_VMX__
	ld    r6,.LC__dl_hwcap@toc(r2)
# ifdef SHARED
	/* Load _rtld-global._dl_hwcap.  */
	ld    r6,RTLD_GLOBAL_RO_DL_HWCAP_OFFSET(r6)
# else
	ld    r6,0(r6) /* Load extern _dl_hwcap.  */
# endif
	andis.  r6,r6,(PPC_FEATURE_HAS_ALTIVEC >> 16)
	beq	L(no_vmx)
	la	r5,((JB_VRS)*8)(3)
	andi.	r6,r5,0xf
	mfspr	r0,VRSAVE
	stw	r0,((JB_VRSAVE)*8)(3)	/* 32-bit VRSAVE.  */
	addi	r6,r5,16
	beq+	L(aligned_save_vmx)

	lvsr	v0,0,r5
	lvsl	v1,0,r5
	addi	r6,r5,-16

# define save_misaligned_vmx(savevr,prevvr,shiftvr,tmpvr,savegpr,addgpr) \
	addi	addgpr,addgpr,32;					 \
	vperm	tmpvr,prevvr,savevr,shiftvr;				 \
	stvx	tmpvr,0,savegpr

	/*
	 * We have to be careful not to corrupt the data below v20 and
	 * above v31. To keep things simple we just rotate both ends in
	 * the opposite direction to our main permute so we can use
	 * the common macro.
	 */

	/* load and rotate data below v20 */
	lvx	v2,0,r5
	vperm	v2,v2,v2,v1
	save_misaligned_vmx(v20,v2,v0,v3,r5,r6)
	save_misaligned_vmx(v21,v20,v0,v3,r6,r5)
	save_misaligned_vmx(v22,v21,v0,v3,r5,r6)
	save_misaligned_vmx(v23,v22,v0,v3,r6,r5)
	save_misaligned_vmx(v24,v23,v0,v3,r5,r6)
	save_misaligned_vmx(v25,v24,v0,v3,r6,r5)
	save_misaligned_vmx(v26,v25,v0,v3,r5,r6)
	save_misaligned_vmx(v27,v26,v0,v3,r6,r5)
	save_misaligned_vmx(v28,v27,v0,v3,r5,r6)
	save_misaligned_vmx(v29,v28,v0,v3,r6,r5)
	save_misaligned_vmx(v30,v29,v0,v3,r5,r6)
	save_misaligned_vmx(v31,v30,v0,v3,r6,r5)
	/* load and rotate data above v31 */
	lvx	v2,0,r6
	vperm	v2,v2,v2,v1
	save_misaligned_vmx(v2,v31,v0,v3,r5,r6)

	b	L(no_vmx)

L(aligned_save_vmx):
	stvx	20,0,r5
	addi	r5,r5,32
	stvx	21,0,r6
	addi	r6,r6,32
	stvx	22,0,r5
	addi	r5,r5,32
	stvx	23,0,r6
	addi	r6,r6,32
	stvx	24,0,r5
	addi	r5,r5,32
	stvx	25,0,r6
	addi	r6,r6,32
	stvx	26,0,r5
	addi	r5,r5,32
	stvx	27,0,r6
	addi	r6,r6,32
	stvx	28,0,r5
	addi	r5,r5,32
	stvx	29,0,r6
	addi	r6,r6,32
	stvx	30,0,r5
	stvx	31,0,r6
L(no_vmx):
#else
	li	r6,0
#endif
#if IS_IN (rtld)
	li	r3,0
	blr
#elif defined SHARED
	b	JUMPTARGET (NOTOC (__sigjmp_save_symbol))
#else
	mflr	r0
	std	r0,FRAME_LR_SAVE(r1)
	stdu	r1,-FRAME_MIN_SIZE(r1)
	cfi_adjust_cfa_offset(FRAME_MIN_SIZE)
	cfi_offset(lr,FRAME_LR_SAVE)
	bl	JUMPTARGET (__sigjmp_save_symbol)
	nop
	ld	r0,FRAME_MIN_SIZE+FRAME_LR_SAVE(r1)
	addi	r1,r1,FRAME_MIN_SIZE
	mtlr	r0
	blr
#endif
END (__sigsetjmp_symbol)

#if defined SHARED && !IS_IN (rtld) && !defined __NO_VMX__
/* When called from within libc we need a special version of __sigsetjmp
   that saves r2 since the call won't go via a plt call stub.  See
   bugz #269.  */
ENTRY (__GI___sigsetjmp)
	std r2,FRAME_TOC_SAVE(r1) /* Save the callers TOC in the save area.  */
	CALL_MCOUNT 1
	b JUMPTARGET (GLUE(__sigsetjmp_symbol,_ent))
END (__GI___sigsetjmp)
#endif
