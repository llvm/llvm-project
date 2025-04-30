/* setjmp for PowerPC.
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
# include <novmxsetjmp.h>
#else
# include <jmpbuf-offsets.h>
#endif

ENTRY (__sigsetjmp_symbol)

#ifdef PTR_MANGLE
	mr   r5,r1
	PTR_MANGLE(r5, r10)
	stw  r5,(JB_GPR1*4)(3)
#else
	stw  r1,(JB_GPR1*4)(3)
#endif
	mflr r0
	/* setjmp probe expects longjmp first argument (4@3), second argument
	   (-4@4), and target address (4@0), respectively.  */
	LIBC_PROBE (setjmp, 3, 4@3, -4@4, 4@0)
	stw  r14,((JB_GPRS+14-14)*4)(3)
#ifdef PTR_MANGLE
	PTR_MANGLE2 (r0, r10)
	li   r10,0
#endif
	stw  r0,(JB_LR*4)(3)
	stw  r15,((JB_GPRS+15-14)*4)(3)
	mfcr r0
	stw  r16,((JB_GPRS+16-14)*4)(3)
	stw  r0,(JB_CR*4)(3)
	stw  r17,((JB_GPRS+17-14)*4)(3)
	stw  r18,((JB_GPRS+18-14)*4)(3)
	stw  r19,((JB_GPRS+19-14)*4)(3)
	stw  r20,((JB_GPRS+20-14)*4)(3)
	stw  r21,((JB_GPRS+21-14)*4)(3)
	stw  r22,((JB_GPRS+22-14)*4)(3)
	stw  r23,((JB_GPRS+23-14)*4)(3)
	stw  r24,((JB_GPRS+24-14)*4)(3)
	stw  r25,((JB_GPRS+25-14)*4)(3)
	stw  r26,((JB_GPRS+26-14)*4)(3)
	stw  r27,((JB_GPRS+27-14)*4)(3)
	stw  r28,((JB_GPRS+28-14)*4)(3)
	stw  r29,((JB_GPRS+29-14)*4)(3)
	stw  r30,((JB_GPRS+30-14)*4)(3)
	stw  r31,((JB_GPRS+31-14)*4)(3)
#if IS_IN (rtld)
	li   r3,0
	blr
#else
	b __sigjmp_save_symbol@local
#endif
END (__sigsetjmp_symbol)
