/* Create new context.
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

#define __ASSEMBLY__
#include <asm/ptrace.h>
#include "ucontext_i.h"
#include <asm/errno.h>

ENTRY (__makecontext)
	CALL_MCOUNT 3
  /* Save parameters into the parameter save area of callers frame.  */
  std  r3,FRAME_PARM_SAVE+0(r1) /* ucontext_t *ucp  */
  std  r4,FRAME_PARM_SAVE+8(r1) /* void (*func)(void)  */
  std  r5,FRAME_PARM_SAVE+16(r1) /* int argc  */
  std  r6,FRAME_PARM_SAVE+24(r1) /* ... */
  std  r7,FRAME_PARM_SAVE+32(r1)
  std  r8,FRAME_PARM_SAVE+40(r1)
  std  r9,FRAME_PARM_SAVE+48(r1)
  std  r10,FRAME_PARM_SAVE+56(r1)
  mflr  r0
  /* Get the address of the target functions first parameter.  */
  addi  r6,r1,FRAME_PARM_SAVE+24
  std   r0,FRAME_LR_SAVE(r1)
  cfi_offset (lr, FRAME_LR_SAVE)
  stdu  r1,-128(r1)
  cfi_adjust_cfa_offset (128)

  /* Get the ucontexts stack pointer and size.  Compute the top of stack
     and round down to a quadword boundary.  Then stack a dummy frame
     with a null back chain.  We store the context pointer in the frames
     "compiler double word" field so we can recover if is the function
     returns.  Finally save the callers link register and TOC pointer
     into this frame so the debugger can display a backtrace.
  */
  ld    r7,UCONTEXT_STACK_SP(r3)
  ld    r0,UCONTEXT_STACK_SIZE(r3)
  add   r7,r7,r0
  clrrdi  r7,r7,4
  li    r0,0
  stdu  r0,-64(r7)
  std   r3,FRAME_PARM_SAVE(r7) /* Store context in dummy parm1.  */
  mflr  r0
  std   r2,FRAME_TOC_SAVE(r7)  /* Store the TOC pointer for later.  */
  std   r0,FRAME_LR_SAVE(r7)

  /* Now we need to stack another frame to hold the parameter save area
     for the function.  We need to allocate a frame with the minimum 48
     byte header and 8 parameter register.  However if there are more
     than 8 parameters addition space is need to hold all the parameters.
     The total size it rounded up to a quadword multiple then a frame is
     stacked.  This address is stored in the ucontext as GPR 1.  */

  cmpdi cr1,r5,8
  sldi  r8,r5,3
  bgt   cr1,L(gt8)
  li    r8,64
L(gt8):
  addi  r8,r8,FRAME_PARM_SAVE+8 /* Add header plus rounding factor.  */
  clrrdi  r8,r8,4  /* Round down to quadword.  */

  subf  r8,r8,r7
  std   r7,0(r8)   /* Stack the frame.  */
  std   r8,(SIGCONTEXT_GP_REGS+(PT_R1*8))(r3)

  /* Now we need to copy the target functions parameters.  The functions
     parameters are saved in the parameter save area.  We skip over the
     first three parameters and copy up to 8 double word into the
     SIGCONTEXT_GP_REGS starting with R3.  If there are more than 8
     parameters then doublewords 8-N are copied into the parameter
     save area of the context frame.  */
  cmpdi r5,0
  beq   L(noparms)
  mr    r0,r5
  ble   cr1,L(le8)
  li    r0,8
L(le8):
  mtctr r0
  addi  r7,r6,-8
  addi  r9,r3,(SIGCONTEXT_GP_REGS+(PT_R3*8)-8)
L(parmloop2):
  ldu   r0,8(r7)
  stdu  r0,8(r9)
  bdnz  L(parmloop2)

  addi  r0,r5,-8
  ble   cr1,L(noparms)
  mtctr r0
  addi  r9,r8,FRAME_PARM_SAVE+64-8
L(parmloop):
  ldu   r0,8(r7)
  stdu  r0,8(r9)
  bdnz  L(parmloop)

L(noparms):

#if _CALL_ELF != 2
  /* Load the function address and TOC from the function descriptor
     and store them in the ucontext as NIP and r2.  Store the 3rd
     field of the function descriptor into the ucontext as r11 in case
     the calling language needs the "environment pointer".  */
  ld    r0,0(r4)
  ld    r10,8(r4);
  ld    r9,16(r4);
  std   r0,(SIGCONTEXT_GP_REGS+(PT_NIP*8))(r3)
  std   r10,(SIGCONTEXT_GP_REGS+(PT_R2*8))(r3)
  std   r9,(SIGCONTEXT_GP_REGS+(PT_R11*8))(r3)
#else
  /* In the ELFv2 ABI, the function pointer is already the address.
     Store it as NIP and r12 as required by the ABI.  */
  std   r4,(SIGCONTEXT_GP_REGS+(PT_NIP*8))(r3)
  std   r4,(SIGCONTEXT_GP_REGS+(PT_R12*8))(r3)
#endif

  /* If the target function returns we need to do some cleanup.  We use a
     code trick to get the address of our cleanup function into the link
     register.  Do not add any code between here and L(exitcode).
     Use this conditional form of branch and link to avoid destroying
     the cpu link stack used to predict blr return addresses.  */
  bcl	20,31,L(gotexitcodeaddr);

  /* End FDE now, because while executing on the context's stack
     the unwind info would be wrong otherwise.  */
  cfi_endproc

	/* This is the helper code which gets called if a function which
	   is registered with 'makecontext' returns.  In this case we
	   have to install the context listed in the uc_link element of
	   the context 'makecontext' manipulated at the time of the
	   'makecontext' call.  If the pointer is NULL the process must
	   terminate.  */
L(exitcode):
	/* Recover the ucontext and TOC from the dummy frame.  */
	ld    r1,FRAME_BACKCHAIN(r1)  /* Unstack the parameter save area frame.  */
	ld    r3,FRAME_PARM_SAVE(r1)
	ld    r2,FRAME_TOC_SAVE(r1)
	ld    r3,UCONTEXT_LINK(r3)  /* Load the resume context.  */
	cmpdi r3,0
	beq   L(do_exit)
	bl    JUMPTARGET(__setcontext)
	nop
	/* If setcontext returns (which can happen if the syscall fails) we will
	   exit the program with error status (-1).  */
	li    r3,-1
L(do_exit):
#ifdef SHARED
	b     JUMPTARGET (NOTOC (__GI_exit));
#else
	b     JUMPTARGET(exit);
	nop
#endif
	b    L(do_exit)

  /* Re-establish FDE for the rest of the actual makecontext routine.  */
  cfi_startproc
  cfi_offset (lr, FRAME_LR_SAVE)
  cfi_adjust_cfa_offset (128)

  /* The address of the exit code is in the link register.  Store the lr
     in the ucontext as LNK so the target function will return to our
     exit code.  */
L(gotexitcodeaddr):
  mflr  r0
  std   r0,(SIGCONTEXT_GP_REGS+(PT_LNK*8))(r3)
  ld    r0,128+FRAME_LR_SAVE(r1)
  addi  r1,r1,128
  mtlr  r0
  blr
END(__makecontext)

weak_alias (__makecontext, makecontext)
