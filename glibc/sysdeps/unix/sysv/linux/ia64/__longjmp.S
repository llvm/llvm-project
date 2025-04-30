/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
   Contributed by David Mosberger-Tang <davidm@hpl.hp.com>.

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
   <https://www.gnu.org/licenses/>.

   Note that __sigsetjmp() did NOT flush the register stack.  Instead,
   we do it here since __longjmp() is usually much less frequently
   invoked than __sigsetjmp(). The only difficulty is that __sigsetjmp()
   didn't (and wouldn't be able to) save ar.rnat either.  This is a problem
   because if we're not careful, we could end up loading random NaT bits.
   There are two cases:

	(i)  ar.bsp < ia64_rse_rnat_addr(jmpbuf.ar_bsp)
		ar.rnat contains the desired bits---preserve ar.rnat
		across loadrs and write to ar.bspstore

	(ii) ar.bsp >= ia64_rse_rnat_addr(jmpbuf.ar_bsp)
		The desired ar.rnat is stored in
		ia64_rse_rnat_addr(jmpbuf.ar_bsp).  Load those
		bits into ar.rnat after setting ar.bspstore. */

#include <sysdep.h>
#include <features.h>

#	define	pPos	p6	/* is rotate count positive? */
#	define	pNeg	p7	/* is rotate count negative? */


	/* __longjmp(__jmp_buf buf, int val) */

LEAF(__longjmp)
#ifdef CHECK_RSP
	alloc r8=ar.pfs,2,1,3,0
	CHECK_RSP
#else
	alloc r8=ar.pfs,2,0,0,0
#endif
	mov r27=ar.rsc
	add r2=0x98,in0		// r2 <- &jmpbuf.orig_jmp_buf_addr
	;;
	ld8 r8=[r2],-16		// r8 <- orig_jmp_buf_addr
	mov r10=ar.bsp
	and r11=~0x3,r27	// clear ar.rsc.mode
	;;
	flushrs			// flush dirty regs to backing store (must be first in insn grp)
	ld8 r23=[r2],8		// r23 <- jmpbuf.ar_bsp
	sub r8=r8,in0		// r8 <- &orig_jmpbuf - &jmpbuf
	;;
	ld8 r25=[r2]		// r25 <- jmpbuf.ar_unat
	extr.u r8=r8,3,6	// r8 <- (&orig_jmpbuf - &jmpbuf)/8 & 0x3f
	;;
	cmp.lt pNeg,pPos=r8,r0
	mov r2=in0
	;;
(pPos)	mov r16=r8
(pNeg)	add r16=64,r8
(pPos)	sub r17=64,r8
(pNeg)	sub r17=r0,r8
	;;
	mov ar.rsc=r11		// put RSE in enforced lazy mode
	shr.u r8=r25,r16
	add r3=8,in0		// r3 <- &jmpbuf.r1
	shl r9=r25,r17
	;;
	ld8.fill.nta r28=[r2],16	// r28 <- jmpbuf.sp
	or r25=r8,r9
	;;
	mov r26=ar.rnat
	mov ar.unat=r25		// setup ar.unat (NaT bits for r1, r4-r7, and r12)
	;;
	ld8.fill.nta gp=[r3],32		// r1 (gp)
	dep r11=-1,r23,3,6	// r11 <- ia64_rse_rnat_addr(jmpbuf.ar_bsp)
	mov sp=r28		// r12 (sp)
	;;
	ld8.nta r16=[r2],16		// caller's unat
//	ld8.nta r17=[r3],16		// fpsr
	;;
	ld8.fill.nta r4=[r2],16	// r4
	ld8.fill.nta r5=[r3],16		// r5 (gp)
	cmp.geu p8,p0=r10,r11	// p8 <- (ar.bsp >= jmpbuf.ar_bsp)
	;;
	ld8.fill.nta r6=[r2],16	// r6
	ld8.fill.nta r7=[r3],16		// r7
	;;
	mov ar.unat=r16			// restore caller's unat
//	mov ar.fpsr=r17			// restore fpsr
	;;
	ld8.nta r16=[r2],16		// b0
	ld8.nta r17=[r3],16		// b1
	;;
(p8)	ld8 r26=[r11]		// r26 <- *ia64_rse_rnat_addr(jmpbuf.ar_bsp)
	mov ar.bspstore=r23	// restore ar.bspstore
	;;
	ld8.nta r18=[r2],16		// b2
	ld8.nta r19=[r3],16		// b3
	;;
#ifdef PTR_DEMANGLE
	PTR_DEMANGLE (r16, r24)
#endif
	ld8.nta r20=[r2],16		// b4
	ld8.nta r21=[r3],16		// b5
	;;
	ld8.nta r11=[r2],16		// ar.pfs
	ld8.nta r22=[r3],56		// ar.lc
	;;
	ld8.nta r24=[r2],32		// pr
	mov b0=r16
	;;
	ldf.fill.nta f2=[r2],32
	ldf.fill.nta f3=[r3],32
	mov b1=r17
	;;
	ldf.fill.nta f4=[r2],32
	ldf.fill.nta f5=[r3],32
	mov b2=r18
	;;
	ldf.fill.nta f16=[r2],32
	ldf.fill.nta f17=[r3],32
	mov b3=r19
	;;
	ldf.fill.nta f18=[r2],32
	ldf.fill.nta f19=[r3],32
	mov b4=r20
	;;
	ldf.fill.nta f20=[r2],32
	ldf.fill.nta f21=[r3],32
	mov b5=r21
	;;
	ldf.fill.nta f22=[r2],32
	ldf.fill.nta f23=[r3],32
	mov ar.lc=r22
	;;
	ldf.fill.nta f24=[r2],32
	ldf.fill.nta f25=[r3],32
	cmp.eq p8,p9=0,in1
	;;
	ldf.fill.nta f26=[r2],32
	ldf.fill.nta f27=[r3],32
	mov ar.pfs=r11
	;;
	ldf.fill.nta f28=[r2],32
	ldf.fill.nta f29=[r3],32
	;;
	ldf.fill.nta f30=[r2]
	ldf.fill.nta f31=[r3]
(p8)	mov r8=1

	mov ar.rnat=r26		// restore ar.rnat
	;;
	mov ar.rsc=r27		// restore ar.rsc
(p9)	mov r8=in1

	invala			// virt. -> phys. regnum mapping may change
	mov pr=r24,-1
	ret
END(__longjmp)
