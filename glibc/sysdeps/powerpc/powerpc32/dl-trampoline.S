/* PLT trampolines.  PPC32 version.
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

#include <sysdep.h>

	.section ".text"
	.align 2
	.globl _dl_runtime_resolve
	.type _dl_runtime_resolve,@function
_dl_runtime_resolve:
	cfi_startproc
 # We need to save the registers used to pass parameters, and register 0,
 # which is used by _mcount; the registers are saved in a stack frame.
	stwu r1,-64(r1)
	cfi_adjust_cfa_offset (64)
	stw r0,12(r1)
	stw r3,16(r1)
	stw r4,20(r1)
 # The code that calls this has put parameters for `fixup' in r12 and r11.
	mr r3,r12
	stw r5,24(r1)
	mr r4,r11
	stw r6,28(r1)
	mflr r0
 # We also need to save some of the condition register fields
	stw r7,32(r1)
 # Don't clobber the caller's LRSAVE, it is needed by _mcount.
	stw r0,48(r1)
 	cfi_offset (lr, -16)
	stw r8,36(r1)
	mfcr r0
	stw r9,40(r1)
	stw r10,44(r1)
	stw r0,8(r1)
	bl _dl_fixup@local
 # 'fixup' returns the address we want to branch to.
	mtctr r3
 # Put the registers back...
	lwz r0,48(r1)
	lwz r10,44(r1)
	lwz r9,40(r1)
	mtlr r0
	lwz r8,36(r1)
	lwz r0,8(r1)
	lwz r7,32(r1)
	lwz r6,28(r1)
	mtcrf 0xFF,r0
	lwz r5,24(r1)
	lwz r4,20(r1)
	lwz r3,16(r1)
	lwz r0,12(r1)
 # ...unwind the stack frame, and jump to the PLT entry we updated.
	addi r1,r1,64
	bctr
	cfi_endproc
	.size	 _dl_runtime_resolve,.-_dl_runtime_resolve

#ifndef PROF
	.align 2
	.globl _dl_prof_resolve
	.type _dl_prof_resolve,@function
_dl_prof_resolve:
	cfi_startproc
 # We need to save the registers used to pass parameters, and register 0,
 # which is used by _mcount; the registers are saved in a stack frame.
	stwu r1,-320(r1)
	cfi_adjust_cfa_offset (320)
	/* Stack layout:

	  +312   stackframe
	  +308   lr
	  +304   r1
	  +288   v12
	  +272   v11
	  +256   v10
	  +240   v9
	  +224   v8
	  +208   v7
	  +192   v6
	  +176   v5
	  +160   v4
	  +144   v3
	  +128   v2
	  +112   v1
	  +104   fp8
	  +96    fp7
	  +88    fp6
	  +80    fp5
	  +72    fp4
	  +64    fp3
	  +56    fp2
	  +48    fp1
	  +44    r10
	  +40    r9
	  +36    r8
	  +32    r7
	  +28    r6
	  +24    r5
	  +20    r4
	  +16    r3
	  +12    r0
	  +8     cr
	   r1    link
	*/
        stw r0,12(r1)
	stw r3,16(r1)
	stw r4,20(r1)
 # The code that calls this has put parameters for `fixup' in r12 and r11.
	mr r3,r12
	stw r5,24(r1)
	mr r4,r11
	stw r6,28(r1)
	mflr r5
 # We also need to save some of the condition register fields.
	stw r7,32(r1)
 # Don't clobber the caller's LRSAVE, it is needed by _mcount.
	stw r5,308(r1)
	cfi_offset (lr, -12)
	stw r8,36(r1)
	mfcr r0
	stw r9,40(r1)
	stw r10,44(r1)
	stw r0,8(r1)
#ifndef __NO_FPRS__
 # Save the floating point registers
	stfd fp1,48(r1)
	stfd fp2,56(r1)
	stfd fp3,64(r1)
	stfd fp4,72(r1)
	stfd fp5,80(r1)
	stfd fp6,88(r1)
	stfd fp7,96(r1)
	stfd fp8,104(r1)
#endif
 # XXX TODO: store vmx registers
 # Load the extra parameters.
	addi r6,r1,16
	addi r7,r1,312
	li r0,-1
	stw r0,0(r7)
	bl _dl_profile_fixup@local
 # 'fixup' returns the address we want to branch to.
	mtctr r3
 # Put the registers back...
	lwz r0,308(r1)
	lwz r10,44(r1)
	lwz r9,40(r1)
	mtlr r0
	lwz r8,36(r1)
	lwz r0,8(r1)
	lwz r7,32(r1)
	lwz r6,28(r1)
	mtcrf 0xFF,r0
	lwz r5,24(r1)
	lwz r4,20(r1)
	lwz r3,16(r1)
        lwz r0,12(r1)
#ifndef __NO_FPRS__
 # Load the floating point registers.
	lfd fp1,48(r1)
	lfd fp2,56(r1)
	lfd fp3,64(r1)
	lfd fp4,72(r1)
	lfd fp5,80(r1)
	lfd fp6,88(r1)
	lfd fp7,96(r1)
	lfd fp8,104(r1)
#endif
 # ...unwind the stack frame, and jump to the PLT entry we updated.
	addi r1,r1,320
	bctr
	cfi_endproc
	.size	 _dl_prof_resolve,.-_dl_prof_resolve
#endif
