/* Save current context, powerpc32 common.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
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

/* This is the common implementation of getcontext for powerpc32.
   It not complete in itself should be included in to a framework that
   defines:
     __CONTEXT_FUNC_NAME
   and if appropriate:
     __CONTEXT_ENABLE_FPRS
     __CONTEXT_ENABLE_VRS
   Any architecture that implements the Vector unit is assumed to also
   implement the floating unit.  */

/* Stack frame offsets.  */
#define _FRAME_BACKCHAIN	0
#define _FRAME_LR_SAVE		4
#define _FRAME_PARM_SAVE1	8
#define _FRAME_PARM_SAVE2	12
#define _FRAME_PARM_SAVE3	16
#define _FRAME_PARM_SAVE4	20

#ifdef __CONTEXT_ENABLE_VRS
	.machine	"altivec"
#endif
ENTRY(__CONTEXT_FUNC_NAME)
	stwu	r1,-16(r1)
	cfi_adjust_cfa_offset (16)
/* Insure that the _UC_REGS start on a quadword boundary.  */
	stw	r3,_FRAME_PARM_SAVE1(r1)
	addi	r3,r3,_UC_REG_SPACE+12
	clrrwi  r3,r3,4

/* Save the general purpose registers */
	stw	r0,_UC_GREGS+(PT_R0*4)(r3)
	mflr	r0
	stw	r2,_UC_GREGS+(PT_R2*4)(r3)
	stw	r4,_UC_GREGS+(PT_R4*4)(r3)
/* Set the callers LR_SAVE, and the ucontext LR and NIP to the callers
   return address.  */
	stw	r0,_UC_GREGS+(PT_LNK*4)(r3)
	stw	r0,_UC_GREGS+(PT_NIP*4)(r3)
	stw	r0,_FRAME_LR_SAVE+16(r1)
	cfi_offset (lr, _FRAME_LR_SAVE)
	stw	r5,_UC_GREGS+(PT_R5*4)(r3)
	stw	r6,_UC_GREGS+(PT_R6*4)(r3)
	stw	r7,_UC_GREGS+(PT_R7*4)(r3)
	stw	r8,_UC_GREGS+(PT_R8*4)(r3)
	stw	r9,_UC_GREGS+(PT_R9*4)(r3)
	stw	r10,_UC_GREGS+(PT_R10*4)(r3)
	stw	r11,_UC_GREGS+(PT_R11*4)(r3)
	stw	r12,_UC_GREGS+(PT_R12*4)(r3)
	stw	r13,_UC_GREGS+(PT_R13*4)(r3)
	stw	r14,_UC_GREGS+(PT_R14*4)(r3)
	stw	r15,_UC_GREGS+(PT_R15*4)(r3)
	stw	r16,_UC_GREGS+(PT_R16*4)(r3)
	stw	r17,_UC_GREGS+(PT_R17*4)(r3)
	stw	r18,_UC_GREGS+(PT_R18*4)(r3)
	stw	r19,_UC_GREGS+(PT_R19*4)(r3)
	stw	r20,_UC_GREGS+(PT_R20*4)(r3)
	stw	r21,_UC_GREGS+(PT_R21*4)(r3)
	stw	r22,_UC_GREGS+(PT_R22*4)(r3)
	stw	r23,_UC_GREGS+(PT_R23*4)(r3)
	stw	r24,_UC_GREGS+(PT_R24*4)(r3)
	stw	r25,_UC_GREGS+(PT_R25*4)(r3)
	stw	r26,_UC_GREGS+(PT_R26*4)(r3)
	stw	r27,_UC_GREGS+(PT_R27*4)(r3)
	stw	r28,_UC_GREGS+(PT_R28*4)(r3)
	stw	r29,_UC_GREGS+(PT_R29*4)(r3)
	stw	r30,_UC_GREGS+(PT_R30*4)(r3)
	stw	r31,_UC_GREGS+(PT_R31*4)(r3)
/* Save the value of R1.  We had to push the stack before we
   had the address of uc_reg_space.  So compute the address of
   the callers stack pointer and save it as R1.  */
	addi	r8,r1,16
	li	r0,0
/* Save the count, exception and condition registers.  */
	mfctr	r11
	mfxer	r10
	mfcr	r9
	stw	r8,_UC_GREGS+(PT_R1*4)(r3)
	stw	r11,_UC_GREGS+(PT_CTR*4)(r3)
	stw	r10,_UC_GREGS+(PT_XER*4)(r3)
	stw	r9,_UC_GREGS+(PT_CCR*4)(r3)
/* Set the return value of getcontext to "success".  R3 is the only
   register whose value is not preserved in the saved context.  */
	stw	r0,_UC_GREGS+(PT_R3*4)(r3)

/* Zero fill fields that can't be set in user state. */
	stw	r0,_UC_GREGS+(PT_MSR*4)(r3)
	stw	r0,_UC_GREGS+(PT_MQ*4)(r3)

#ifdef __CONTEXT_ENABLE_FPRS
/* Save the floating-point registers */
	stfd	fp0,_UC_FREGS+(0*8)(r3)
	stfd	fp1,_UC_FREGS+(1*8)(r3)
	stfd	fp2,_UC_FREGS+(2*8)(r3)
	stfd	fp3,_UC_FREGS+(3*8)(r3)
	stfd	fp4,_UC_FREGS+(4*8)(r3)
	stfd	fp5,_UC_FREGS+(5*8)(r3)
	stfd	fp6,_UC_FREGS+(6*8)(r3)
	stfd	fp7,_UC_FREGS+(7*8)(r3)
	stfd	fp8,_UC_FREGS+(8*8)(r3)
	stfd	fp9,_UC_FREGS+(9*8)(r3)
	stfd	fp10,_UC_FREGS+(10*8)(r3)
	stfd	fp11,_UC_FREGS+(11*8)(r3)
	stfd	fp12,_UC_FREGS+(12*8)(r3)
	stfd	fp13,_UC_FREGS+(13*8)(r3)
	stfd	fp14,_UC_FREGS+(14*8)(r3)
	stfd	fp15,_UC_FREGS+(15*8)(r3)
	stfd	fp16,_UC_FREGS+(16*8)(r3)
	stfd	fp17,_UC_FREGS+(17*8)(r3)
	stfd	fp18,_UC_FREGS+(18*8)(r3)
	stfd	fp19,_UC_FREGS+(19*8)(r3)
	stfd	fp20,_UC_FREGS+(20*8)(r3)
	stfd	fp21,_UC_FREGS+(21*8)(r3)
	stfd	fp22,_UC_FREGS+(22*8)(r3)
	stfd	fp23,_UC_FREGS+(23*8)(r3)
	stfd	fp24,_UC_FREGS+(24*8)(r3)
	stfd	fp25,_UC_FREGS+(25*8)(r3)
	stfd	fp26,_UC_FREGS+(26*8)(r3)
	stfd	fp27,_UC_FREGS+(27*8)(r3)
	stfd	fp28,_UC_FREGS+(28*8)(r3)
	stfd	fp29,_UC_FREGS+(29*8)(r3)
	mffs	fp0
	stfd	fp30,_UC_FREGS+(30*8)(r3)
	stfd	fp31,_UC_FREGS+(31*8)(r3)
	stfd	fp0,_UC_FREGS+(32*8)(r3)

# ifdef __CONTEXT_ENABLE_VRS
#  ifdef PIC
	mflr    r8
#  define got_label GENERATE_GOT_LABEL (__CONTEXT_FUNC_NAME)
	SETUP_GOT_ACCESS(r7,got_label)
	addis	r7,r7,_GLOBAL_OFFSET_TABLE_-got_label@ha
	addi	r7,r7,_GLOBAL_OFFSET_TABLE_-got_label@l
#   ifdef SHARED
	lwz     r7,_rtld_global_ro@got(r7)
	mtlr    r8
	lwz     r7,RTLD_GLOBAL_RO_DL_HWCAP_OFFSET+LOWORD(r7)
#   else
	lwz     r7,_dl_hwcap@got(r7)
	mtlr    r8
	lwz     r7,LOWORD(r7)
#   endif
#  else
	lis	r7,(_dl_hwcap+LOWORD)@ha
	lwz     r7,(_dl_hwcap+LOWORD)@l(r7)
#  endif
	andis.	r7,r7,(PPC_FEATURE_HAS_ALTIVEC >> 16)

	la	r10,(_UC_VREGS)(r3)
	la	r9,(_UC_VREGS+16)(r3)

	beq	2f	/* L(no_vec) */
/* address of the combined VSCR/VSAVE quadword.  */
	la	r8,(_UC_VREGS+512)(r3)

/* Save the vector registers */
	stvx  v0,0,r10
	stvx  v1,0,r9
	addi  r10,r10,32
	addi  r9,r9,32
/* We need to get the Vector Status and Control Register early to avoid
   store order problems later with the VSAVE register that shares the
   same quadword.  */
	mfvscr	v0

	stvx  v2,0,r10
	stvx  v3,0,r9
	addi  r10,r10,32
	addi  r9,r9,32

	stvx	v0,0,r8

	stvx  v4,0,r10
	stvx  v5,0,r9
	addi  r10,r10,32
	addi  r9,r9,32

	stvx  v6,0,r10
	stvx  v7,0,r9
	addi  r10,r10,32
	addi  r9,r9,32

	stvx  v8,0,r10
	stvx  v9,0,r9
	addi  r10,r10,32
	addi  r9,r9,32

	stvx  v10,0,r10
	stvx  v11,0,r9
	addi  r10,r10,32
	addi  r9,r9,32

	stvx  v12,0,r10
	stvx  v13,0,r9
	addi  r10,r10,32
	addi  r9,r9,32

	stvx  v14,0,r10
	stvx  v15,0,r9
	addi  r10,r10,32
	addi  r9,r9,32

	stvx  v16,0,r10
	stvx  v17,0,r9
	addi  r10,r10,32
	addi  r9,r9,32

	stvx  v18,0,r10
	stvx  v19,0,r9
	addi  r10,r10,32
	addi  r9,r9,32

	stvx  v20,0,r10
	stvx  v21,0,r9
	addi  r10,r10,32
	addi  r9,r9,32

	stvx  v22,0,r10
	stvx  v23,0,r9
	addi  r10,r10,32
	addi  r9,r9,32

	stvx  v24,0,r10
	stvx  v25,0,r9
	addi  r10,r10,32
	addi  r9,r9,32

	stvx  v26,0,r10
	stvx  v27,0,r9
	addi  r10,r10,32
	addi  r9,r9,32

	stvx  v28,0,r10
	stvx  v29,0,r9
	addi  r10,r10,32
	addi  r9,r9,32

	mfspr	r0,VRSAVE
	stvx  v30,0,r10
	stvx  v31,0,r9

 	stw	r0,0(r8)

2: /* L(no_vec): */
# endif
#endif

/* We need to set up parms and call sigprocmask which will clobber
   volatile registers. So before the call we need to retrieve the
   original ucontext ptr (parm1) from stack and store the UC_REGS_PTR
   (current R3).  */
	lwz	r12,_FRAME_PARM_SAVE1(r1)
	li	r4,0
	stw	r3,_UC_REGS_PTR(r12)
	addi	r5,r12,_UC_SIGMASK
	li	r3,SIG_BLOCK
	bl	__sigprocmask@local

	lwz	r0,_FRAME_LR_SAVE+16(r1)
	addi	r1,r1,16
	mtlr	r0
	blr
END(__CONTEXT_FUNC_NAME)
