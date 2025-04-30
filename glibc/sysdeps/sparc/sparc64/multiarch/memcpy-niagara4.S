/* Copy SIZE bytes from SRC to DEST.  For SUN4V Niagara-4.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David S. Miller (davem@davemloft.net)

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

#define ASI_BLK_INIT_QUAD_LDD_P	0xe2

#define FPRS_FEF		0x04

/* On T4 it is very expensive to access ASRs like %fprs and
 * %asi, avoiding a read or a write can save ~50 cycles.
 */
#define FPU_ENTER			\
	rd	%fprs, %o5;		\
	andcc	%o5, FPRS_FEF, %g0;	\
	be,a,pn	%icc, 999f;		\
	 wr	%g0, FPRS_FEF, %fprs;	\
	999:

#define VISEntryHalf FPU_ENTER
#define VISExitHalf and %o5, FPRS_FEF, %o5; wr %o5, 0x0, %fprs

#define GLOBAL_SPARE	%g5

#define STORE_ASI	ASI_BLK_INIT_QUAD_LDD_P
#define EX_LD(x)	x
#define EX_ST(x)	x
#define EX_RETVAL(x)	x
#define LOAD(type,addr,dest)	type [addr], dest
#define STORE(type,src,addr)	type src, [addr]
#define STORE_INIT(src,addr)	stxa src, [addr] STORE_ASI

#if IS_IN (libc)

	.register	%g2,#scratch
	.register	%g3,#scratch
	.register	%g6,#scratch

	.text

ENTRY(__mempcpy_niagara4)
	ba,pt		%icc, 101f
	 add		%o0, %o2, %o3
END(__mempcpy_niagara4)

	.align		32
ENTRY(__memcpy_niagara4)
100:	/* %o0=dst, %o1=src, %o2=len */
	mov		%o0, %o3
101:
#ifndef __arch64__
	srl		%o2, 0, %o2
#endif
	brz,pn		%o2, .Lexit
	 cmp		%o2, 3
	ble,pn		%icc, .Ltiny
	 cmp		%o2, 19
	ble,pn		%icc, .Lsmall
	 or		%o0, %o1, %g2
	cmp		%o2, 128
	bl,pn		%icc, .Lmedium
	 nop

.Llarge:/* len >= 0x80 */
	/* First get dest 8 byte aligned.  */
	sub		%g0, %o0, %g1
	and		%g1, 0x7, %g1
	brz,pt		%g1, 51f
	 sub		%o2, %g1, %o2

1:	EX_LD(LOAD(ldub, %o1 + 0x00, %g2))
	add		%o1, 1, %o1
	subcc		%g1, 1, %g1
	add		%o0, 1, %o0
	bne,pt		%icc, 1b
	 EX_ST(STORE(stb, %g2, %o0 - 0x01))

51:	LOAD(prefetch, %o1 + 0x040, #n_reads_strong)
	LOAD(prefetch, %o1 + 0x080, #n_reads_strong)
	LOAD(prefetch, %o1 + 0x0c0, #n_reads_strong)
	LOAD(prefetch, %o1 + 0x100, #n_reads_strong)
	LOAD(prefetch, %o1 + 0x140, #n_reads_strong)
	LOAD(prefetch, %o1 + 0x180, #n_reads_strong)
	LOAD(prefetch, %o1 + 0x1c0, #n_reads_strong)
	LOAD(prefetch, %o1 + 0x200, #n_reads_strong)

	/* Check if we can use the straight fully aligned
	 * loop, or we require the alignaddr/faligndata variant.
	 */
	andcc		%o1, 0x7, %o5
	bne,pn		%icc, .Llarge_src_unaligned
	 sub		%g0, %o0, %g1

	/* Legitimize the use of initializing stores by getting dest
	 * to be 64-byte aligned.
	 */
	and		%g1, 0x3f, %g1
	brz,pt		%g1, .Llarge_aligned
	 sub		%o2, %g1, %o2

1:	EX_LD(LOAD(ldx, %o1 + 0x00, %g2))
	add		%o1, 8, %o1
	subcc		%g1, 8, %g1
	add		%o0, 8, %o0
	bne,pt		%icc, 1b
	 EX_ST(STORE(stx, %g2, %o0 - 0x08))

.Llarge_aligned:
	/* len >= 0x80 && src 8-byte aligned && dest 8-byte aligned */
	andn		%o2, 0x3f, %o4
	sub		%o2, %o4, %o2

1:	EX_LD(LOAD(ldx, %o1 + 0x00, %g1))
	add		%o1, 0x40, %o1
	EX_LD(LOAD(ldx, %o1 - 0x38, %g2))
	subcc		%o4, 0x40, %o4
	EX_LD(LOAD(ldx, %o1 - 0x30, %g3))
	EX_LD(LOAD(ldx, %o1 - 0x28, GLOBAL_SPARE))
	EX_LD(LOAD(ldx, %o1 - 0x20, %o5))
	EX_ST(STORE_INIT(%g1, %o0))
	add		%o0, 0x08, %o0
	EX_ST(STORE_INIT(%g2, %o0))
	add		%o0, 0x08, %o0
	EX_LD(LOAD(ldx, %o1 - 0x18, %g2))
	EX_ST(STORE_INIT(%g3, %o0))
	add		%o0, 0x08, %o0
	EX_LD(LOAD(ldx, %o1 - 0x10, %g3))
	EX_ST(STORE_INIT(GLOBAL_SPARE, %o0))
	add		%o0, 0x08, %o0
	EX_LD(LOAD(ldx, %o1 - 0x08, GLOBAL_SPARE))
	EX_ST(STORE_INIT(%o5, %o0))
	add		%o0, 0x08, %o0
	EX_ST(STORE_INIT(%g2, %o0))
	add		%o0, 0x08, %o0
	EX_ST(STORE_INIT(%g3, %o0))
	add		%o0, 0x08, %o0
	EX_ST(STORE_INIT(GLOBAL_SPARE, %o0))
	add		%o0, 0x08, %o0
	bne,pt		%icc, 1b
	 LOAD(prefetch, %o1 + 0x200, #n_reads_strong)

	membar		#StoreLoad | #StoreStore

	brz,pn		%o2, .Lexit
	 cmp		%o2, 19
	ble,pn		%icc, .Lsmall_unaligned
	 nop
	ba,a,pt		%icc, .Lmedium_noprefetch

.Lexit:	retl
	 mov		EX_RETVAL(%o3), %o0

.Llarge_src_unaligned:
	andn		%o2, 0x3f, %o4
	sub		%o2, %o4, %o2
	VISEntryHalf
	alignaddr	%o1, %g0, %g1
	add		%o1, %o4, %o1
	EX_LD(LOAD(ldd, %g1 + 0x00, %f0))
1:	EX_LD(LOAD(ldd, %g1 + 0x08, %f2))
	subcc		%o4, 0x40, %o4
	EX_LD(LOAD(ldd, %g1 + 0x10, %f4))
	EX_LD(LOAD(ldd, %g1 + 0x18, %f6))
	EX_LD(LOAD(ldd, %g1 + 0x20, %f8))
	EX_LD(LOAD(ldd, %g1 + 0x28, %f10))
	EX_LD(LOAD(ldd, %g1 + 0x30, %f12))
	EX_LD(LOAD(ldd, %g1 + 0x38, %f14))
	faligndata	%f0, %f2, %f16
	EX_LD(LOAD(ldd, %g1 + 0x40, %f0))
	faligndata	%f2, %f4, %f18
	add		%g1, 0x40, %g1
	faligndata	%f4, %f6, %f20
	faligndata	%f6, %f8, %f22
	faligndata	%f8, %f10, %f24
	faligndata	%f10, %f12, %f26
	faligndata	%f12, %f14, %f28
	faligndata	%f14, %f0, %f30
	EX_ST(STORE(std, %f16, %o0 + 0x00))
	EX_ST(STORE(std, %f18, %o0 + 0x08))
	EX_ST(STORE(std, %f20, %o0 + 0x10))
	EX_ST(STORE(std, %f22, %o0 + 0x18))
	EX_ST(STORE(std, %f24, %o0 + 0x20))
	EX_ST(STORE(std, %f26, %o0 + 0x28))
	EX_ST(STORE(std, %f28, %o0 + 0x30))
	EX_ST(STORE(std, %f30, %o0 + 0x38))
	add		%o0, 0x40, %o0
	bne,pt		%icc, 1b
	 LOAD(prefetch, %g1 + 0x200, #n_reads_strong)
	VISExitHalf

	brz,pn		%o2, .Lexit
	 cmp		%o2, 19
	ble,pn		%icc, .Lsmall_unaligned
	 nop
	ba,a,pt		%icc, .Lmedium_unaligned

.Lmedium:
	LOAD(prefetch, %o1 + 0x40, #n_reads_strong)
	andcc		%g2, 0x7, %g0
	bne,pn		%icc, .Lmedium_unaligned
	 nop
.Lmedium_noprefetch:
	andncc		%o2, 0x20 - 1, %o5
	be,pn		%icc, 2f
	 sub		%o2, %o5, %o2
1:	EX_LD(LOAD(ldx, %o1 + 0x00, %g1))
	EX_LD(LOAD(ldx, %o1 + 0x08, %g2))
	EX_LD(LOAD(ldx, %o1 + 0x10, GLOBAL_SPARE))
	EX_LD(LOAD(ldx, %o1 + 0x18, %o4))
	add		%o1, 0x20, %o1
	subcc		%o5, 0x20, %o5
	EX_ST(STORE(stx, %g1, %o0 + 0x00))
	EX_ST(STORE(stx, %g2, %o0 + 0x08))
	EX_ST(STORE(stx, GLOBAL_SPARE, %o0 + 0x10))
	EX_ST(STORE(stx, %o4, %o0 + 0x18))
	bne,pt		%icc, 1b
	 add		%o0, 0x20, %o0
2:	andcc		%o2, 0x18, %o5
	be,pt		%icc, 3f
	 sub		%o2, %o5, %o2
1:	EX_LD(LOAD(ldx, %o1 + 0x00, %g1))
	add		%o1, 0x08, %o1
	add		%o0, 0x08, %o0
	subcc		%o5, 0x08, %o5
	bne,pt		%icc, 1b
	 EX_ST(STORE(stx, %g1, %o0 - 0x08))
3:	brz,pt		%o2, .Lexit
	 cmp		%o2, 0x04
	bl,pn		%icc, .Ltiny
	 nop
	EX_LD(LOAD(lduw, %o1 + 0x00, %g1))
	add		%o1, 0x04, %o1
	add		%o0, 0x04, %o0
	subcc		%o2, 0x04, %o2
	bne,pn		%icc, .Ltiny
	 EX_ST(STORE(stw, %g1, %o0 - 0x04))
	ba,a,pt		%icc, .Lexit
.Lmedium_unaligned:
	/* First get dest 8 byte aligned.  */
	sub		%g0, %o0, %g1
	and		%g1, 0x7, %g1
	brz,pt		%g1, 2f
	 sub		%o2, %g1, %o2

1:	EX_LD(LOAD(ldub, %o1 + 0x00, %g2))
	add		%o1, 1, %o1
	subcc		%g1, 1, %g1
	add		%o0, 1, %o0
	bne,pt		%icc, 1b
	 EX_ST(STORE(stb, %g2, %o0 - 0x01))
2:
	and		%o1, 0x7, %g1
	brz,pn		%g1, .Lmedium_noprefetch
	 sll		%g1, 3, %g1
	mov		64, %g2
	sub		%g2, %g1, %g2
	andn		%o1, 0x7, %o1
	EX_LD(LOAD(ldx, %o1 + 0x00, %o4))
	sllx		%o4, %g1, %o4
	andn		%o2, 0x08 - 1, %o5
	sub		%o2, %o5, %o2
1:	EX_LD(LOAD(ldx, %o1 + 0x08, %g3))
	add		%o1, 0x08, %o1
	subcc		%o5, 0x08, %o5
	srlx		%g3, %g2, GLOBAL_SPARE
	or		GLOBAL_SPARE, %o4, GLOBAL_SPARE
	EX_ST(STORE(stx, GLOBAL_SPARE, %o0 + 0x00))
	add		%o0, 0x08, %o0
	bne,pt		%icc, 1b
	 sllx		%g3, %g1, %o4
	srl		%g1, 3, %g1
	add		%o1, %g1, %o1
	brz,pn		%o2, .Lexit
	 nop
	ba,pt		%icc, .Lsmall_unaligned

.Ltiny:
	EX_LD(LOAD(ldub, %o1 + 0x00, %g1))
	subcc		%o2, 1, %o2
	be,pn		%icc, .Lexit
	 EX_ST(STORE(stb, %g1, %o0 + 0x00))
	EX_LD(LOAD(ldub, %o1 + 0x01, %g1))
	subcc		%o2, 1, %o2
	be,pn		%icc, .Lexit
	 EX_ST(STORE(stb, %g1, %o0 + 0x01))
	EX_LD(LOAD(ldub, %o1 + 0x02, %g1))
	ba,pt		%icc, .Lexit
	 EX_ST(STORE(stb, %g1, %o0 + 0x02))

.Lsmall:
	andcc		%g2, 0x3, %g0
	bne,pn		%icc, .Lsmall_unaligned
	 andn		%o2, 0x4 - 1, %o5
	sub		%o2, %o5, %o2
1:
	EX_LD(LOAD(lduw, %o1 + 0x00, %g1))
	add		%o1, 0x04, %o1
	subcc		%o5, 0x04, %o5
	add		%o0, 0x04, %o0
	bne,pt		%icc, 1b
	 EX_ST(STORE(stw, %g1, %o0 - 0x04))
	brz,pt		%o2, .Lexit
	 nop
	ba,a,pt		%icc, .Ltiny

.Lsmall_unaligned:
1:	EX_LD(LOAD(ldub, %o1 + 0x00, %g1))
	add		%o1, 1, %o1
	add		%o0, 1, %o0
	subcc		%o2, 1, %o2
	bne,pt		%icc, 1b
	 EX_ST(STORE(stb, %g1, %o0 - 0x01))
	ba,a,pt		%icc, .Lexit
END(__memcpy_niagara4)

#endif
