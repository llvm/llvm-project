/* Save current context.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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
#include <rtld-global-offsets.h>
#include <shlib-compat.h>

#define __ASSEMBLY__
#include <asm/ptrace.h>
#include <asm/errno.h>
#include "ucontext_i.h"


#if SHLIB_COMPAT (libc, GLIBC_2_3, GLIBC_2_3_4)
ENTRY(__novec_getcontext)
	CALL_MCOUNT 1
  std  r0,(SIGCONTEXT_GP_REGS+(PT_R0*8))(r3)
  std  r1,(SIGCONTEXT_GP_REGS+(PT_R1*8))(r3)
  mflr  r0
  std  r2,(SIGCONTEXT_GP_REGS+(PT_R2*8))(r3)
  std  r0,FRAME_LR_SAVE(r1)
  cfi_offset (lr, FRAME_LR_SAVE)
  std  r0,(SIGCONTEXT_GP_REGS+(PT_LNK*8))(r3)
  std  r0,(SIGCONTEXT_GP_REGS+(PT_NIP*8))(r3)
  stdu  r1,-128(r1)
  cfi_adjust_cfa_offset (128)
  std  r4,(SIGCONTEXT_GP_REGS+(PT_R4*8))(r3)
  std  r5,(SIGCONTEXT_GP_REGS+(PT_R5*8))(r3)
  std  r6,(SIGCONTEXT_GP_REGS+(PT_R6*8))(r3)
  std  r7,(SIGCONTEXT_GP_REGS+(PT_R7*8))(r3)
  std  r8,(SIGCONTEXT_GP_REGS+(PT_R8*8))(r3)
  std  r9,(SIGCONTEXT_GP_REGS+(PT_R9*8))(r3)
  std  r10,(SIGCONTEXT_GP_REGS+(PT_R10*8))(r3)
  std  r11,(SIGCONTEXT_GP_REGS+(PT_R11*8))(r3)
  std  r12,(SIGCONTEXT_GP_REGS+(PT_R12*8))(r3)
  std  r13,(SIGCONTEXT_GP_REGS+(PT_R13*8))(r3)
  std  r14,(SIGCONTEXT_GP_REGS+(PT_R14*8))(r3)
  std  r15,(SIGCONTEXT_GP_REGS+(PT_R15*8))(r3)
  std  r16,(SIGCONTEXT_GP_REGS+(PT_R16*8))(r3)
  std  r17,(SIGCONTEXT_GP_REGS+(PT_R17*8))(r3)
  std  r18,(SIGCONTEXT_GP_REGS+(PT_R18*8))(r3)
  std  r19,(SIGCONTEXT_GP_REGS+(PT_R19*8))(r3)
  std  r20,(SIGCONTEXT_GP_REGS+(PT_R20*8))(r3)
  std  r21,(SIGCONTEXT_GP_REGS+(PT_R21*8))(r3)
  std  r22,(SIGCONTEXT_GP_REGS+(PT_R22*8))(r3)
  std  r23,(SIGCONTEXT_GP_REGS+(PT_R23*8))(r3)
  std  r24,(SIGCONTEXT_GP_REGS+(PT_R24*8))(r3)
  std  r25,(SIGCONTEXT_GP_REGS+(PT_R25*8))(r3)
  std  r26,(SIGCONTEXT_GP_REGS+(PT_R26*8))(r3)
  std  r27,(SIGCONTEXT_GP_REGS+(PT_R27*8))(r3)
  std  r28,(SIGCONTEXT_GP_REGS+(PT_R28*8))(r3)
  std  r29,(SIGCONTEXT_GP_REGS+(PT_R29*8))(r3)
  std  r30,(SIGCONTEXT_GP_REGS+(PT_R30*8))(r3)
  std  r31,(SIGCONTEXT_GP_REGS+(PT_R31*8))(r3)
  mfctr  r0
  std  r0,(SIGCONTEXT_GP_REGS+(PT_CTR*8))(r3)
  mfxer  r0
  std  r0,(SIGCONTEXT_GP_REGS+(PT_XER*8))(r3)
  mfcr  r0
  std  r0,(SIGCONTEXT_GP_REGS+(PT_CCR*8))(r3)

  /* Set the return value of swapcontext to "success".  R3 is the only
     register whose value is not preserved in the saved context.  */
  li   r0,0
  std  r0,(SIGCONTEXT_GP_REGS+(PT_R3*8))(r3)

  /* Zero fill fields that can't be set in user state or are unused.  */
  std  r0,(SIGCONTEXT_GP_REGS+(PT_MSR*8))(r3)
  std  r0,(SIGCONTEXT_GP_REGS+(34*8))(r3)
  std  r0,(SIGCONTEXT_GP_REGS+(PT_SOFTE*8))(r3)
  std  r0,(SIGCONTEXT_GP_REGS+(40*8))(r3)
  std  r0,(SIGCONTEXT_GP_REGS+(41*8))(r3)
  std  r0,(SIGCONTEXT_GP_REGS+(42*8))(r3)
  std  r0,(SIGCONTEXT_GP_REGS+(PT_RESULT*8))(r3)

  /* Set the PT_REGS pointer to the address of sigcontext's gp_regs
     field.  Struct pt_regs and elf_gregset_t are the same thing.
     We kept the regs field for backwards compatibility with
     libraries built before we extended sigcontext.  */
  addi r0,r3,SIGCONTEXT_GP_REGS
  std  r0,SIGCONTEXT_PT_REGS(r3)

  stfd  fp0,(SIGCONTEXT_FP_REGS+(PT_R0*8))(r3)
  stfd  fp1,(SIGCONTEXT_FP_REGS+(PT_R1*8))(r3)
  stfd  fp2,(SIGCONTEXT_FP_REGS+(PT_R2*8))(r3)
  stfd  fp3,(SIGCONTEXT_FP_REGS+(PT_R3*8))(r3)
  stfd  fp4,(SIGCONTEXT_FP_REGS+(PT_R4*8))(r3)
  stfd  fp5,(SIGCONTEXT_FP_REGS+(PT_R5*8))(r3)
  stfd  fp6,(SIGCONTEXT_FP_REGS+(PT_R6*8))(r3)
  stfd  fp7,(SIGCONTEXT_FP_REGS+(PT_R7*8))(r3)
  stfd  fp8,(SIGCONTEXT_FP_REGS+(PT_R8*8))(r3)
  stfd  fp9,(SIGCONTEXT_FP_REGS+(PT_R9*8))(r3)
  stfd  fp10,(SIGCONTEXT_FP_REGS+(PT_R10*8))(r3)
  stfd  fp11,(SIGCONTEXT_FP_REGS+(PT_R11*8))(r3)
  stfd  fp12,(SIGCONTEXT_FP_REGS+(PT_R12*8))(r3)
  stfd  fp13,(SIGCONTEXT_FP_REGS+(PT_R13*8))(r3)
  stfd  fp14,(SIGCONTEXT_FP_REGS+(PT_R14*8))(r3)
  stfd  fp15,(SIGCONTEXT_FP_REGS+(PT_R15*8))(r3)
  stfd  fp16,(SIGCONTEXT_FP_REGS+(PT_R16*8))(r3)
  stfd  fp17,(SIGCONTEXT_FP_REGS+(PT_R17*8))(r3)
  stfd  fp18,(SIGCONTEXT_FP_REGS+(PT_R18*8))(r3)
  stfd  fp19,(SIGCONTEXT_FP_REGS+(PT_R19*8))(r3)
  stfd  fp20,(SIGCONTEXT_FP_REGS+(PT_R20*8))(r3)
  stfd  fp21,(SIGCONTEXT_FP_REGS+(PT_R21*8))(r3)
  stfd  fp22,(SIGCONTEXT_FP_REGS+(PT_R22*8))(r3)
  stfd  fp23,(SIGCONTEXT_FP_REGS+(PT_R23*8))(r3)
  stfd  fp24,(SIGCONTEXT_FP_REGS+(PT_R24*8))(r3)
  stfd  fp25,(SIGCONTEXT_FP_REGS+(PT_R25*8))(r3)
  stfd  fp26,(SIGCONTEXT_FP_REGS+(PT_R26*8))(r3)
  stfd  fp27,(SIGCONTEXT_FP_REGS+(PT_R27*8))(r3)
  stfd  fp28,(SIGCONTEXT_FP_REGS+(PT_R28*8))(r3)
  stfd  fp29,(SIGCONTEXT_FP_REGS+(PT_R29*8))(r3)
  mffs  fp0
  stfd  fp30,(SIGCONTEXT_FP_REGS+(PT_R30*8))(r3)
  stfd  fp31,(SIGCONTEXT_FP_REGS+(PT_R31*8))(r3)
  stfd  fp0,(SIGCONTEXT_FP_REGS+(32*8))(r3)

  addi  r5,r3,UCONTEXT_SIGMASK
  li  r4,0
  li  r3,SIG_BLOCK
  bl  JUMPTARGET(__sigprocmask)
  nop

  ld    r0,128+FRAME_LR_SAVE(r1)
  addi  r1,r1,128
  mtlr  r0
  blr
PSEUDO_END(__novec_getcontext)

compat_symbol (libc, __novec_getcontext, getcontext, GLIBC_2_3)

#endif

	.section	".toc","aw"
.LC__dl_hwcap:
#ifdef SHARED
	.tc _rtld_global_ro[TC],_rtld_global_ro
#else
	.tc _dl_hwcap[TC],_dl_hwcap
#endif
	.section ".text"

	.machine	"altivec"
ENTRY(__getcontext)
	CALL_MCOUNT 1
  std  r0,(SIGCONTEXT_GP_REGS+(PT_R0*8))(r3)
  std  r1,(SIGCONTEXT_GP_REGS+(PT_R1*8))(r3)
  mflr  r0
  std  r2,(SIGCONTEXT_GP_REGS+(PT_R2*8))(r3)
  std  r0,FRAME_LR_SAVE(r1)
  cfi_offset (lr, FRAME_LR_SAVE)
  std  r0,(SIGCONTEXT_GP_REGS+(PT_LNK*8))(r3)
  std  r0,(SIGCONTEXT_GP_REGS+(PT_NIP*8))(r3)
  stdu  r1,-128(r1)
  cfi_adjust_cfa_offset (128)
  std  r4,(SIGCONTEXT_GP_REGS+(PT_R4*8))(r3)
  std  r5,(SIGCONTEXT_GP_REGS+(PT_R5*8))(r3)
  std  r6,(SIGCONTEXT_GP_REGS+(PT_R6*8))(r3)
  std  r7,(SIGCONTEXT_GP_REGS+(PT_R7*8))(r3)
  std  r8,(SIGCONTEXT_GP_REGS+(PT_R8*8))(r3)
  std  r9,(SIGCONTEXT_GP_REGS+(PT_R9*8))(r3)
  std  r10,(SIGCONTEXT_GP_REGS+(PT_R10*8))(r3)
  std  r11,(SIGCONTEXT_GP_REGS+(PT_R11*8))(r3)
  std  r12,(SIGCONTEXT_GP_REGS+(PT_R12*8))(r3)
  std  r13,(SIGCONTEXT_GP_REGS+(PT_R13*8))(r3)
  std  r14,(SIGCONTEXT_GP_REGS+(PT_R14*8))(r3)
  std  r15,(SIGCONTEXT_GP_REGS+(PT_R15*8))(r3)
  std  r16,(SIGCONTEXT_GP_REGS+(PT_R16*8))(r3)
  std  r17,(SIGCONTEXT_GP_REGS+(PT_R17*8))(r3)
  std  r18,(SIGCONTEXT_GP_REGS+(PT_R18*8))(r3)
  std  r19,(SIGCONTEXT_GP_REGS+(PT_R19*8))(r3)
  std  r20,(SIGCONTEXT_GP_REGS+(PT_R20*8))(r3)
  std  r21,(SIGCONTEXT_GP_REGS+(PT_R21*8))(r3)
  std  r22,(SIGCONTEXT_GP_REGS+(PT_R22*8))(r3)
  std  r23,(SIGCONTEXT_GP_REGS+(PT_R23*8))(r3)
  std  r24,(SIGCONTEXT_GP_REGS+(PT_R24*8))(r3)
  std  r25,(SIGCONTEXT_GP_REGS+(PT_R25*8))(r3)
  std  r26,(SIGCONTEXT_GP_REGS+(PT_R26*8))(r3)
  std  r27,(SIGCONTEXT_GP_REGS+(PT_R27*8))(r3)
  std  r28,(SIGCONTEXT_GP_REGS+(PT_R28*8))(r3)
  std  r29,(SIGCONTEXT_GP_REGS+(PT_R29*8))(r3)
  std  r30,(SIGCONTEXT_GP_REGS+(PT_R30*8))(r3)
  std  r31,(SIGCONTEXT_GP_REGS+(PT_R31*8))(r3)
  mfctr  r0
  std  r0,(SIGCONTEXT_GP_REGS+(PT_CTR*8))(r3)
  mfxer  r0
  std  r0,(SIGCONTEXT_GP_REGS+(PT_XER*8))(r3)
  mfcr  r0
  std  r0,(SIGCONTEXT_GP_REGS+(PT_CCR*8))(r3)

  /* Set the return value of swapcontext to "success".  R3 is the only
     register whose value is not preserved in the saved context.  */
  li   r0,0
  std  r0,(SIGCONTEXT_GP_REGS+(PT_R3*8))(r3)

  /* Zero fill fields that can't be set in user state or are unused.  */
  std  r0,(SIGCONTEXT_GP_REGS+(PT_MSR*8))(r3)
  std  r0,(SIGCONTEXT_GP_REGS+(34*8))(r3)
  std  r0,(SIGCONTEXT_GP_REGS+(PT_SOFTE*8))(r3)
  std  r0,(SIGCONTEXT_GP_REGS+(40*8))(r3)
  std  r0,(SIGCONTEXT_GP_REGS+(41*8))(r3)
  std  r0,(SIGCONTEXT_GP_REGS+(42*8))(r3)
  std  r0,(SIGCONTEXT_GP_REGS+(PT_RESULT*8))(r3)

  /* Set the PT_REGS pointer to the address of sigcontext's gp_regs
     field.  Struct pt_regs and elf_gregset_t are the same thing.
     We kept the regs field for backwards compatibility with
     libraries built before we extended sigcontext.  */
  addi r0,r3,SIGCONTEXT_GP_REGS
  std  r0,SIGCONTEXT_PT_REGS(r3)

  stfd  fp0,(SIGCONTEXT_FP_REGS+(PT_R0*8))(r3)
  stfd  fp1,(SIGCONTEXT_FP_REGS+(PT_R1*8))(r3)
  stfd  fp2,(SIGCONTEXT_FP_REGS+(PT_R2*8))(r3)
  stfd  fp3,(SIGCONTEXT_FP_REGS+(PT_R3*8))(r3)
  stfd  fp4,(SIGCONTEXT_FP_REGS+(PT_R4*8))(r3)
  stfd  fp5,(SIGCONTEXT_FP_REGS+(PT_R5*8))(r3)
  stfd  fp6,(SIGCONTEXT_FP_REGS+(PT_R6*8))(r3)
  stfd  fp7,(SIGCONTEXT_FP_REGS+(PT_R7*8))(r3)
  stfd  fp8,(SIGCONTEXT_FP_REGS+(PT_R8*8))(r3)
  stfd  fp9,(SIGCONTEXT_FP_REGS+(PT_R9*8))(r3)
  stfd  fp10,(SIGCONTEXT_FP_REGS+(PT_R10*8))(r3)
  stfd  fp11,(SIGCONTEXT_FP_REGS+(PT_R11*8))(r3)
  stfd  fp12,(SIGCONTEXT_FP_REGS+(PT_R12*8))(r3)
  stfd  fp13,(SIGCONTEXT_FP_REGS+(PT_R13*8))(r3)
  stfd  fp14,(SIGCONTEXT_FP_REGS+(PT_R14*8))(r3)
  stfd  fp15,(SIGCONTEXT_FP_REGS+(PT_R15*8))(r3)
  stfd  fp16,(SIGCONTEXT_FP_REGS+(PT_R16*8))(r3)
  stfd  fp17,(SIGCONTEXT_FP_REGS+(PT_R17*8))(r3)
  stfd  fp18,(SIGCONTEXT_FP_REGS+(PT_R18*8))(r3)
  stfd  fp19,(SIGCONTEXT_FP_REGS+(PT_R19*8))(r3)
  stfd  fp20,(SIGCONTEXT_FP_REGS+(PT_R20*8))(r3)
  stfd  fp21,(SIGCONTEXT_FP_REGS+(PT_R21*8))(r3)
  stfd  fp22,(SIGCONTEXT_FP_REGS+(PT_R22*8))(r3)
  stfd  fp23,(SIGCONTEXT_FP_REGS+(PT_R23*8))(r3)
  stfd  fp24,(SIGCONTEXT_FP_REGS+(PT_R24*8))(r3)
  stfd  fp25,(SIGCONTEXT_FP_REGS+(PT_R25*8))(r3)
  stfd  fp26,(SIGCONTEXT_FP_REGS+(PT_R26*8))(r3)
  stfd  fp27,(SIGCONTEXT_FP_REGS+(PT_R27*8))(r3)
  stfd  fp28,(SIGCONTEXT_FP_REGS+(PT_R28*8))(r3)
  stfd  fp29,(SIGCONTEXT_FP_REGS+(PT_R29*8))(r3)
  mffs  fp0
  stfd  fp30,(SIGCONTEXT_FP_REGS+(PT_R30*8))(r3)
  stfd  fp31,(SIGCONTEXT_FP_REGS+(PT_R31*8))(r3)
  stfd  fp0,(SIGCONTEXT_FP_REGS+(32*8))(r3)

  ld    r5,.LC__dl_hwcap@toc(r2)
# ifdef SHARED
/* Load _rtld-global._dl_hwcap.  */
  ld    r5,RTLD_GLOBAL_RO_DL_HWCAP_OFFSET(r5)
# else
  ld    r5,0(r5) /* Load extern _dl_hwcap.  */
# endif
  la    r10,(SIGCONTEXT_V_RESERVE+8)(r3)
  la    r9,(SIGCONTEXT_V_RESERVE+24)(r3)

  andis.  r5,r5,(PPC_FEATURE_HAS_ALTIVEC >> 16)

  clrrdi  r10,r10,4
  beq   L(has_no_vec)
  clrrdi  r9,r9,4
  mr    r5,r10	/* Capture *v_regs value in r5.  */

  stvx  v0,0,r10
  stvx  v1,0,r9
  addi  r10,r10,32
  addi  r9,r9,32

  stvx  v2,0,r10
  stvx  v3,0,r9
  addi  r10,r10,32
  addi  r9,r9,32

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

  stvx  v30,0,r10
  stvx  v31,0,r9
  addi  r10,r10,32
  addi  r9,r9,32

  mfvscr  v0
  mfspr r0,VRSAVE
  stvx  v0,0,r10
  stw   r0,0(9)

L(has_no_vec):
/*
   Store either a NULL or a quadword aligned pointer to the Vector register
   array into *v_regs.
*/
  std   r5,(SIGCONTEXT_V_REGS_PTR)(r3)

  addi  r5,r3,UCONTEXT_SIGMASK
  li  r4,0
  li  r3,SIG_BLOCK
  bl  JUMPTARGET(__sigprocmask)
  nop

  ld    r0,128+FRAME_LR_SAVE(r1)
  addi  r1,r1,128
  mtlr  r0
  blr
PSEUDO_END(__getcontext)

versioned_symbol (libc, __getcontext, getcontext, GLIBC_2_3_4)
