/* Switch to context.
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
#include "ucontext_i.h"
#include <asm/errno.h>

	.section	".toc","aw"
.LC__dl_hwcap:
#ifdef SHARED
	.tc _rtld_global_ro[TC],_rtld_global_ro
#else
	.tc _dl_hwcap[TC],_dl_hwcap
#endif
	.section ".text"

#if SHLIB_COMPAT (libc, GLIBC_2_3, GLIBC_2_3_4)
ENTRY(__novec_setcontext)
	CALL_MCOUNT 1
  mflr  r0
  std   r31,-8(1)
  cfi_offset(r31,-8)
  std   r0,FRAME_LR_SAVE(r1)
  cfi_offset (lr, FRAME_LR_SAVE)
  stdu  r1,-128(r1)
  cfi_adjust_cfa_offset (128)
  mr    r31,r3

  li    r5,0
  addi  r4,r3,UCONTEXT_SIGMASK
  li    r3,SIG_SETMASK
  bl    JUMPTARGET(__sigprocmask)
  nop
  cmpdi r3,0
  bne   L(nv_error_exit)

# ifdef SHARED
/* Load _rtld-global._dl_hwcap.  */
  ld    r5,RTLD_GLOBAL_RO_DL_HWCAP_OFFSET(r5)
# else
  ld    r5,0(r5) /* Load extern _dl_hwcap.  */
# endif

  lfd  fp0,(SIGCONTEXT_FP_REGS+(32*8))(r31)
  lfd  fp31,(SIGCONTEXT_FP_REGS+(PT_R31*8))(r31)
  lfd  fp30,(SIGCONTEXT_FP_REGS+(PT_R30*8))(r31)

# ifdef _ARCH_PWR6
  /* Use the extended four-operand version of the mtfsf insn.  */
  .machine push
  .machine "power6"

  mtfsf  0xff,fp0,1,0

  .machine pop
# else
  /* Availability of DFP indicates a 64-bit FPSCR.  */
  andi.  r6,r5,PPC_FEATURE_HAS_DFP
  beq    5f
  /* Use the extended four-operand version of the mtfsf insn.  */
  .machine push
  .machine "power6"

  mtfsf  0xff,fp0,1,0

  .machine pop

  b      6f
  /* Continue to operate on the FPSCR as if it were 32-bits.  */
5:
  mtfsf  0xff,fp0
6:
# endif /* _ARCH_PWR6 */

  lfd  fp29,(SIGCONTEXT_FP_REGS+(PT_R29*8))(r31)
  lfd  fp28,(SIGCONTEXT_FP_REGS+(PT_R28*8))(r31)
  lfd  fp27,(SIGCONTEXT_FP_REGS+(PT_R27*8))(r31)
  lfd  fp26,(SIGCONTEXT_FP_REGS+(PT_R26*8))(r31)
  lfd  fp25,(SIGCONTEXT_FP_REGS+(PT_R25*8))(r31)
  lfd  fp24,(SIGCONTEXT_FP_REGS+(PT_R24*8))(r31)
  lfd  fp23,(SIGCONTEXT_FP_REGS+(PT_R23*8))(r31)
  lfd  fp22,(SIGCONTEXT_FP_REGS+(PT_R22*8))(r31)
  lfd  fp21,(SIGCONTEXT_FP_REGS+(PT_R21*8))(r31)
  lfd  fp20,(SIGCONTEXT_FP_REGS+(PT_R20*8))(r31)
  lfd  fp19,(SIGCONTEXT_FP_REGS+(PT_R19*8))(r31)
  lfd  fp18,(SIGCONTEXT_FP_REGS+(PT_R18*8))(r31)
  lfd  fp17,(SIGCONTEXT_FP_REGS+(PT_R17*8))(r31)
  lfd  fp16,(SIGCONTEXT_FP_REGS+(PT_R16*8))(r31)
  lfd  fp15,(SIGCONTEXT_FP_REGS+(PT_R15*8))(r31)
  lfd  fp14,(SIGCONTEXT_FP_REGS+(PT_R14*8))(r31)
  lfd  fp13,(SIGCONTEXT_FP_REGS+(PT_R13*8))(r31)
  lfd  fp12,(SIGCONTEXT_FP_REGS+(PT_R12*8))(r31)
  lfd  fp11,(SIGCONTEXT_FP_REGS+(PT_R11*8))(r31)
  lfd  fp10,(SIGCONTEXT_FP_REGS+(PT_R10*8))(r31)
  lfd  fp9,(SIGCONTEXT_FP_REGS+(PT_R9*8))(r31)
  lfd  fp8,(SIGCONTEXT_FP_REGS+(PT_R8*8))(r31)
  lfd  fp7,(SIGCONTEXT_FP_REGS+(PT_R7*8))(r31)
  lfd  fp6,(SIGCONTEXT_FP_REGS+(PT_R6*8))(r31)
  lfd  fp5,(SIGCONTEXT_FP_REGS+(PT_R5*8))(r31)
  lfd  fp4,(SIGCONTEXT_FP_REGS+(PT_R4*8))(r31)
  lfd  fp3,(SIGCONTEXT_FP_REGS+(PT_R3*8))(r31)
  lfd  fp2,(SIGCONTEXT_FP_REGS+(PT_R2*8))(r31)
  lfd  fp1,(SIGCONTEXT_FP_REGS+(PT_R1*8))(r31)
  lfd  fp0,(SIGCONTEXT_FP_REGS+(PT_R0*8))(r31)

  /* End FDE now, because the unwind info would be wrong while
     we're reloading registers to switch to the new context.  */
  cfi_endproc

  ld   r0,(SIGCONTEXT_GP_REGS+(PT_LNK*8))(r31)
  ld   r1,(SIGCONTEXT_GP_REGS+(PT_R1*8))(r31)
  mtlr r0
  ld   r2,(SIGCONTEXT_GP_REGS+(PT_R2*8))(r31)
  ld   r0,(SIGCONTEXT_GP_REGS+(PT_XER*8))(r31)
  ld   r3,(SIGCONTEXT_GP_REGS+(PT_R3*8))(r31)
  mtxer r0
  ld   r4,(SIGCONTEXT_GP_REGS+(PT_R4*8))(r31)
  ld   r0,(SIGCONTEXT_GP_REGS+(PT_CCR*8))(r31)
  ld   r5,(SIGCONTEXT_GP_REGS+(PT_R5*8))(r31)
  mtcr r0
  ld   r6,(SIGCONTEXT_GP_REGS+(PT_R6*8))(r31)
  ld   r7,(SIGCONTEXT_GP_REGS+(PT_R7*8))(r31)
  ld   r8,(SIGCONTEXT_GP_REGS+(PT_R8*8))(r31)
  ld   r9,(SIGCONTEXT_GP_REGS+(PT_R9*8))(r31)
  ld   r10,(SIGCONTEXT_GP_REGS+(PT_R10*8))(r31)
  ld   r11,(SIGCONTEXT_GP_REGS+(PT_R11*8))(r31)
  ld   r12,(SIGCONTEXT_GP_REGS+(PT_R12*8))(r31)
  /* Don't reload the thread ID or TLS pointer (r13).  */
  ld   r14,(SIGCONTEXT_GP_REGS+(PT_R14*8))(r31)
  ld   r15,(SIGCONTEXT_GP_REGS+(PT_R15*8))(r31)
  ld   r16,(SIGCONTEXT_GP_REGS+(PT_R16*8))(r31)
  ld   r17,(SIGCONTEXT_GP_REGS+(PT_R17*8))(r31)
  ld   r18,(SIGCONTEXT_GP_REGS+(PT_R18*8))(r31)
  ld   r19,(SIGCONTEXT_GP_REGS+(PT_R19*8))(r31)
  ld   r20,(SIGCONTEXT_GP_REGS+(PT_R20*8))(r31)
  ld   r21,(SIGCONTEXT_GP_REGS+(PT_R21*8))(r31)
  ld   r22,(SIGCONTEXT_GP_REGS+(PT_R22*8))(r31)
  ld   r23,(SIGCONTEXT_GP_REGS+(PT_R23*8))(r31)
  ld   r24,(SIGCONTEXT_GP_REGS+(PT_R24*8))(r31)
  ld   r25,(SIGCONTEXT_GP_REGS+(PT_R25*8))(r31)
  ld   r26,(SIGCONTEXT_GP_REGS+(PT_R26*8))(r31)
  ld   r27,(SIGCONTEXT_GP_REGS+(PT_R27*8))(r31)
  ld   r28,(SIGCONTEXT_GP_REGS+(PT_R28*8))(r31)
  ld   r29,(SIGCONTEXT_GP_REGS+(PT_R29*8))(r31)
  ld   r30,(SIGCONTEXT_GP_REGS+(PT_R30*8))(r31)

  /* Now we branch to the "Next Instruction Pointer" from the saved
     context.  With the powerpc64 instruction set there is no good way to
     do this (from user state) without clobbering either the LR or CTR.
     The makecontext and swapcontext functions depend on the callers
     LR being preserved so we use the CTR.  */
  ld   r0,(SIGCONTEXT_GP_REGS+(PT_NIP*8))(r31)
  mtctr r0
  ld   r0,(SIGCONTEXT_GP_REGS+(PT_R0*8))(r31)
  ld   r31,(SIGCONTEXT_GP_REGS+(PT_R31*8))(r31)
  bctr

  /* Re-establish FDE for the rest of the actual setcontext routine.  */
  cfi_startproc
  cfi_offset (lr, FRAME_LR_SAVE)
  cfi_adjust_cfa_offset (128)

L(nv_error_exit):
  ld   r0,128+FRAME_LR_SAVE(r1)
  addi r1,r1,128
  mtlr r0
	ld   r31,-8(r1)
  blr
PSEUDO_END(__novec_setcontext)

compat_symbol (libc, __novec_setcontext, setcontext, GLIBC_2_3)

#endif

	.section ".text"
	.machine	"altivec"
ENTRY(__setcontext)
	CALL_MCOUNT 1
  mflr  r0
  std   r31,-8(1)
  cfi_offset(r31,-8)
  std   r0,FRAME_LR_SAVE(r1)
  cfi_offset (lr, FRAME_LR_SAVE)
  stdu  r1,-128(r1)
  cfi_adjust_cfa_offset (128)
  mr    r31,r3

  li    r5,0
  addi  r4,r3,UCONTEXT_SIGMASK
  li    r3,SIG_SETMASK
  bl    JUMPTARGET(__sigprocmask)
  nop
  cmpdi r3,0
  bne   L(error_exit)

  ld    r5,.LC__dl_hwcap@toc(r2)
  ld    r10,(SIGCONTEXT_V_REGS_PTR)(r31)
# ifdef SHARED
/* Load _rtld-global._dl_hwcap.  */
  ld    r5,RTLD_GLOBAL_RO_DL_HWCAP_OFFSET(r5)
# else
  ld    r5,0(r5) /* Load extern _dl_hwcap.  */
# endif
  andis.  r6,r5,(PPC_FEATURE_HAS_ALTIVEC >> 16)
  beq   L(has_no_vec)

  cmpdi r10,0
  beq   L(has_no_vec)
  lwz   r0,(33*16)(r10)

  li    r9,(16*32)
  mtspr VRSAVE,r0
  cmpwi r0,0
  beq   L(has_no_vec)

  lvx   v19,r9,r10
  la    r9,(16)(r10)

  lvx   v0,0,r10
  lvx   v1,0,r9
  addi  r10,r10,32
  addi  r9,r9,32

  mtvscr  v19
  lvx   v2,0,r10
  lvx   v3,0,r9
  addi  r10,r10,32
  addi  r9,r9,32

  lvx   v4,0,r10
  lvx   v5,0,r9
  addi  r10,r10,32
  addi  r9,r9,32

  lvx   v6,0,r10
  lvx   v7,0,r9
  addi  r10,r10,32
  addi  r9,r9,32

  lvx   v8,0,r10
  lvx   v9,0,r9
  addi  r10,r10,32
  addi  r9,r9,32

  lvx   v10,0,r10
  lvx   v11,0,r9
  addi  r10,r10,32
  addi  r9,r9,32

  lvx   v12,0,r10
  lvx   v13,0,r9
  addi  r10,r10,32
  addi  r9,r9,32

  lvx   v14,0,r10
  lvx   v15,0,r9
  addi  r10,r10,32
  addi  r9,r9,32

  lvx   v16,0,r10
  lvx   v17,0,r9
  addi  r10,r10,32
  addi  r9,r9,32

  lvx   v18,0,r10
  lvx   v19,0,r9
  addi  r10,r10,32
  addi  r9,r9,32

  lvx   v20,0,r10
  lvx   v21,0,r9
  addi  r10,r10,32
  addi  r9,r9,32

  lvx   v22,0,r10
  lvx   v23,0,r9
  addi  r10,r10,32
  addi  r9,r9,32

  lvx   v24,0,r10
  lvx   v25,0,r9
  addi  r10,r10,32
  addi  r9,r9,32

  lvx   v26,0,r10
  lvx   v27,0,r9
  addi  r10,r10,32
  addi  r9,r9,32

  lvx   v28,0,r10
  lvx   v29,0,r9
  addi  r10,r10,32
  addi  r9,r9,32

  lvx   v30,0,r10
  lvx   v31,0,r9
  addi  r10,r10,32
  addi  r9,r9,32

  lvx   v10,0,r10
  lvx   v11,0,r9
  addi  r10,r10,32
  addi  r9,r9,32

L(has_no_vec):
  lfd  fp0,(SIGCONTEXT_FP_REGS+(32*8))(r31)
  lfd  fp31,(SIGCONTEXT_FP_REGS+(PT_R31*8))(r31)
  lfd  fp30,(SIGCONTEXT_FP_REGS+(PT_R30*8))(r31)

# ifdef _ARCH_PWR6
  /* Use the extended four-operand version of the mtfsf insn.  */
  .machine push
  .machine "power6"

  mtfsf  0xff,fp0,1,0

  .machine pop
# else
  /* Availability of DFP indicates a 64-bit FPSCR.  */
  andi.  r6,r5,PPC_FEATURE_HAS_DFP
  beq    7f
  /* Use the extended four-operand version of the mtfsf insn.  */
  .machine push
  .machine "power6"

  mtfsf  0xff,fp0,1,0

  .machine pop

  b      8f
  /* Continue to operate on the FPSCR as if it were 32-bits.  */
7:
  mtfsf  0xff,fp0
8:
# endif /* _ARCH_PWR6 */

  lfd  fp29,(SIGCONTEXT_FP_REGS+(PT_R29*8))(r31)
  lfd  fp28,(SIGCONTEXT_FP_REGS+(PT_R28*8))(r31)
  lfd  fp27,(SIGCONTEXT_FP_REGS+(PT_R27*8))(r31)
  lfd  fp26,(SIGCONTEXT_FP_REGS+(PT_R26*8))(r31)
  lfd  fp25,(SIGCONTEXT_FP_REGS+(PT_R25*8))(r31)
  lfd  fp24,(SIGCONTEXT_FP_REGS+(PT_R24*8))(r31)
  lfd  fp23,(SIGCONTEXT_FP_REGS+(PT_R23*8))(r31)
  lfd  fp22,(SIGCONTEXT_FP_REGS+(PT_R22*8))(r31)
  lfd  fp21,(SIGCONTEXT_FP_REGS+(PT_R21*8))(r31)
  lfd  fp20,(SIGCONTEXT_FP_REGS+(PT_R20*8))(r31)
  lfd  fp19,(SIGCONTEXT_FP_REGS+(PT_R19*8))(r31)
  lfd  fp18,(SIGCONTEXT_FP_REGS+(PT_R18*8))(r31)
  lfd  fp17,(SIGCONTEXT_FP_REGS+(PT_R17*8))(r31)
  lfd  fp16,(SIGCONTEXT_FP_REGS+(PT_R16*8))(r31)
  lfd  fp15,(SIGCONTEXT_FP_REGS+(PT_R15*8))(r31)
  lfd  fp14,(SIGCONTEXT_FP_REGS+(PT_R14*8))(r31)
  lfd  fp13,(SIGCONTEXT_FP_REGS+(PT_R13*8))(r31)
  lfd  fp12,(SIGCONTEXT_FP_REGS+(PT_R12*8))(r31)
  lfd  fp11,(SIGCONTEXT_FP_REGS+(PT_R11*8))(r31)
  lfd  fp10,(SIGCONTEXT_FP_REGS+(PT_R10*8))(r31)
  lfd  fp9,(SIGCONTEXT_FP_REGS+(PT_R9*8))(r31)
  lfd  fp8,(SIGCONTEXT_FP_REGS+(PT_R8*8))(r31)
  lfd  fp7,(SIGCONTEXT_FP_REGS+(PT_R7*8))(r31)
  lfd  fp6,(SIGCONTEXT_FP_REGS+(PT_R6*8))(r31)
  lfd  fp5,(SIGCONTEXT_FP_REGS+(PT_R5*8))(r31)
  lfd  fp4,(SIGCONTEXT_FP_REGS+(PT_R4*8))(r31)
  lfd  fp3,(SIGCONTEXT_FP_REGS+(PT_R3*8))(r31)
  lfd  fp2,(SIGCONTEXT_FP_REGS+(PT_R2*8))(r31)
  lfd  fp1,(SIGCONTEXT_FP_REGS+(PT_R1*8))(r31)
  lfd  fp0,(SIGCONTEXT_FP_REGS+(PT_R0*8))(r31)

  /* End FDE now, because the unwind info would be wrong while
     we're reloading registers to switch to the new context.  */
  cfi_endproc

  ld   r0,(SIGCONTEXT_GP_REGS+(PT_LNK*8))(r31)
  ld   r1,(SIGCONTEXT_GP_REGS+(PT_R1*8))(r31)
  mtlr r0
  ld   r2,(SIGCONTEXT_GP_REGS+(PT_R2*8))(r31)
  ld   r0,(SIGCONTEXT_GP_REGS+(PT_XER*8))(r31)
  ld   r3,(SIGCONTEXT_GP_REGS+(PT_R3*8))(r31)
  mtxer r0
  ld   r4,(SIGCONTEXT_GP_REGS+(PT_R4*8))(r31)
  ld   r0,(SIGCONTEXT_GP_REGS+(PT_CCR*8))(r31)
  ld   r5,(SIGCONTEXT_GP_REGS+(PT_R5*8))(r31)
  ld   r6,(SIGCONTEXT_GP_REGS+(PT_R6*8))(r31)
  ld   r7,(SIGCONTEXT_GP_REGS+(PT_R7*8))(r31)
  ld   r8,(SIGCONTEXT_GP_REGS+(PT_R8*8))(r31)
  ld   r9,(SIGCONTEXT_GP_REGS+(PT_R9*8))(r31)
  mtcr r0
  ld   r10,(SIGCONTEXT_GP_REGS+(PT_R10*8))(r31)
  ld   r11,(SIGCONTEXT_GP_REGS+(PT_R11*8))(r31)
  ld   r12,(SIGCONTEXT_GP_REGS+(PT_R12*8))(r31)
  /* Don't reload the thread ID or TLS pointer (r13).  */
  ld   r14,(SIGCONTEXT_GP_REGS+(PT_R14*8))(r31)
  ld   r15,(SIGCONTEXT_GP_REGS+(PT_R15*8))(r31)
  ld   r16,(SIGCONTEXT_GP_REGS+(PT_R16*8))(r31)
  ld   r17,(SIGCONTEXT_GP_REGS+(PT_R17*8))(r31)
  ld   r18,(SIGCONTEXT_GP_REGS+(PT_R18*8))(r31)
  ld   r19,(SIGCONTEXT_GP_REGS+(PT_R19*8))(r31)
  ld   r20,(SIGCONTEXT_GP_REGS+(PT_R20*8))(r31)
  ld   r21,(SIGCONTEXT_GP_REGS+(PT_R21*8))(r31)
  ld   r22,(SIGCONTEXT_GP_REGS+(PT_R22*8))(r31)
  ld   r23,(SIGCONTEXT_GP_REGS+(PT_R23*8))(r31)
  ld   r24,(SIGCONTEXT_GP_REGS+(PT_R24*8))(r31)
  ld   r25,(SIGCONTEXT_GP_REGS+(PT_R25*8))(r31)
  ld   r26,(SIGCONTEXT_GP_REGS+(PT_R26*8))(r31)
  ld   r27,(SIGCONTEXT_GP_REGS+(PT_R27*8))(r31)
  ld   r28,(SIGCONTEXT_GP_REGS+(PT_R28*8))(r31)
  ld   r29,(SIGCONTEXT_GP_REGS+(PT_R29*8))(r31)
  ld   r30,(SIGCONTEXT_GP_REGS+(PT_R30*8))(r31)

  /* Now we branch to the "Next Instruction Pointer" from the saved
     context.  With the powerpc64 instruction set there is no good way to
     do this (from user state) without clobbering either the LR or CTR.
     The makecontext and swapcontext functions depend on the callers
     LR being preserved so we use the CTR.  */
  ld   r0,(SIGCONTEXT_GP_REGS+(PT_NIP*8))(r31)
  mtctr r0
  ld   r0,(SIGCONTEXT_GP_REGS+(PT_R0*8))(r31)
  ld   r31,(SIGCONTEXT_GP_REGS+(PT_R31*8))(r31)
  bctr

  /* Re-establish FDE for the rest of the actual setcontext routine.  */
  cfi_startproc
  cfi_offset (lr, FRAME_LR_SAVE)
  cfi_adjust_cfa_offset (128)

L(error_exit):
  ld   r0,128+FRAME_LR_SAVE(r1)
  addi r1,r1,128
  mtlr r0
	ld   r31,-8(r1)
  blr

PSEUDO_END(__setcontext)

versioned_symbol (libc, __setcontext, setcontext, GLIBC_2_3_4)
