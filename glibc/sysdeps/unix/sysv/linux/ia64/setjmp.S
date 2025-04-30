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

   The layout of the jmp_buf is as follows.  This is subject to change
   and user-code should never depend on the particular layout of
   jmp_buf!


	offset:	description:
	-------	------------
	0x000	stack pointer (r12)	; unchangeable (see _JMPBUF_UNWINDS)
	0x008	r1 (gp)
	0x010	caller's unat
	0x018	fpsr			; disabled per the C standard; BZ 16379
	0x020	r4
	0x028	r5
	0x030	r6
	0x038	r7
	0x040	rp (b0)
	0x048	b1
	0x050	b2
	0x058	b3
	0x060	b4
	0x068	b5
	0x070	ar.pfs
	0x078	ar.lc
	0x080	pr
	0x088	ar.bsp			; unchangeable (see __longjmp.S)
	0x090	ar.unat
	0x098	&__jmp_buf	; address of the jmpbuf (needed to locate NaT bits in unat)
	0x0a0	 f2
	0x0b0	 f3
	0x0c0	 f4
	0x0d0	 f5
	0x0e0	f16
	0x0f0	f17
	0x100	f18
	0x110	f19
	0x120	f20
	0x130	f21
	0x130	f22
	0x140	f23
	0x150	f24
	0x160	f25
	0x170	f26
	0x180	f27
	0x190	f28
	0x1a0	f29
	0x1b0	f30
	0x1c0	f31 */

#include <sysdep.h>
#include <features.h>

	/* The following two entry points are the traditional entry points: */

LEAF(setjmp)
	alloc r8=ar.pfs,2,0,0,0
	mov in1=1
	br.cond.sptk.many HIDDEN_JUMPTARGET(__sigsetjmp)
END(setjmp)

LEAF(_setjmp)
	alloc r8=ar.pfs,2,0,0,0
	mov in1=0
	br.cond.sptk.many HIDDEN_JUMPTARGET(__sigsetjmp)
END(_setjmp)
libc_hidden_def (_setjmp)

	/* __sigsetjmp(__jmp_buf buf, int savemask) */

ENTRY(__sigsetjmp)
	.prologue ASM_UNW_PRLG_RP|ASM_UNW_PRLG_PFS, ASM_UNW_PRLG_GRSAVE(2)
	alloc loc1=ar.pfs,2,5,2,0
	.save ar.unat, loc2
	mov loc2=ar.unat
	;;
//	mov r17=ar.fpsr
	mov r2=in0
	add r3=8,in0
	;;
.mem.offset 8,0;	st8.spill.nta [r2]=sp,16	// r12 (sp)
.mem.offset 0,0;	st8.spill.nta [r3]=gp,32	// r1 (gp)
	;;
	st8.nta [r2]=loc2,16		// save caller's unat
//	st8.nta [r3]=r17,16		// save fpsr
	add r8=0xa0,in0
	;;
.mem.offset 8,0;	st8.spill.nta [r2]=r4,16	// r4
.mem.offset 0,0;	st8.spill.nta [r3]=r5,16	// r5
	add r9=0xb0,in0
	;;
	stf.spill.nta [r8]=f2,32
	stf.spill.nta [r9]=f3,32
	mov loc0=rp
	.body
	;;
	stf.spill.nta [r8]=f4,32
	stf.spill.nta [r9]=f5,32
	mov r17=b1
	;;
	stf.spill.nta [r8]=f16,32
	stf.spill.nta [r9]=f17,32
	mov r18=b2
	;;
	stf.spill.nta [r8]=f18,32
	stf.spill.nta [r9]=f19,32
	mov r19=b3
	;;
	stf.spill.nta [r8]=f20,32
	stf.spill.nta [r9]=f21,32
	mov r20=b4
	;;
	stf.spill.nta [r8]=f22,32
	stf.spill.nta [r9]=f23,32
	mov r21=b5
	;;
	stf.spill.nta [r8]=f24,32
	stf.spill.nta [r9]=f25,32
	mov r22=ar.lc
	;;
	stf.spill.nta [r8]=f26,32
	stf.spill.nta [r9]=f27,32
	mov r24=pr
	;;
	stf.spill.nta [r8]=f28,32
	stf.spill.nta [r9]=f29,32
#ifdef PTR_MANGLE
	mov loc3=loc0
	;;
	PTR_MANGLE (loc3, loc4)
#else
	;;
#endif
	stf.spill.nta [r8]=f30
	stf.spill.nta [r9]=f31

.mem.offset 8,0;	st8.spill.nta [r2]=r6,16	// r6
.mem.offset 0,0;	st8.spill.nta [r3]=r7,16	// r7
	;;
	mov r23=ar.bsp
	mov r25=ar.unat
	mov out0=in0

#ifdef PTR_MANGLE
	st8.nta [r2]=loc3,16		// b0
#else
	st8.nta [r2]=loc0,16		// b0
#endif
	st8.nta [r3]=r17,16		// b1
	mov out1=in1
	;;
	st8.nta [r2]=r18,16		// b2
	st8.nta [r3]=r19,16		// b3
	;;
	st8.nta [r2]=r20,16		// b4
	st8.nta [r3]=r21,16		// b5
	;;
	st8.nta [r2]=loc1,16		// ar.pfs
	st8.nta [r3]=r22,16		// ar.lc
	;;
	st8.nta [r2]=r24,16		// pr
	st8.nta [r3]=r23,16		// ar.bsp
	;;
	st8.nta [r2]=r25		// ar.unat
	st8.nta [r3]=in0		// &__jmp_buf
#if IS_IN (rtld)
	/* In ld.so we never save the signal mask.  */
	;;
#else
	br.call.dpnt.few rp=__sigjmp_save
#endif
.ret0:					// force a new bundle ::q
	mov.m ar.unat=loc2		// restore caller's unat
	mov rp=loc0
	mov ar.pfs=loc1
	mov r8=0
	ret
END(__sigsetjmp)
libc_hidden_def (__sigsetjmp)
rtld_hidden_def (__sigsetjmp)

weak_extern(_setjmp)
weak_extern(setjmp)
