/* Copy SIZE bytes from SRC to DEST.
   For UltraSPARC-III.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David S. Miller (davem@redhat.com)

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

#define ASI_BLK_P 0xf0
#define FPRS_FEF  0x04
#define VISEntryHalf rd %fprs, %o5; wr %g0, FPRS_FEF, %fprs
#define VISExitHalf and %o5, FPRS_FEF, %o5; wr %o5, 0x0, %fprs

#ifndef XCC
#define USE_BPR
#define XCC xcc
#endif

#if IS_IN (libc)

	.register	%g2,#scratch
	.register	%g3,#scratch
	.register	%g6,#scratch

	.text

ENTRY(__mempcpy_ultra3)
	ba,pt		%XCC, 101f
	 add		%o0, %o2, %g5
END(__mempcpy_ultra3)

	/* Special/non-trivial issues of this code:
	 *
	 * 1) %o5 is preserved from VISEntryHalf to VISExitHalf
	 * 2) Only low 32 FPU registers are used so that only the
	 *    lower half of the FPU register set is dirtied by this
	 *    code.  This is especially important in the kernel.
	 * 3) This code never prefetches cachelines past the end
	 *    of the source buffer.
	 *
	 * The cheetah's flexible spine, oversized liver, enlarged heart,
	 * slender muscular body, and claws make it the swiftest hunter
	 * in Africa and the fastest animal on land.  Can reach speeds
	 * of up to 2.4GB per second.
	 */
	.align		32
ENTRY(__memcpy_ultra3)

100: /* %o0=dst, %o1=src, %o2=len */
	mov		%o0, %g5
101:
	cmp		%o2, 0
	be,pn		%XCC, out
218:	 or		%o0, %o1, %o3
	cmp		%o2, 16
	bleu,a,pn	%XCC, small_copy
	 or		%o3, %o2, %o3

	cmp		%o2, 256
	blu,pt		%XCC, medium_copy
	 andcc		%o3, 0x7, %g0

	ba,pt		%xcc, enter
	 andcc		%o0, 0x3f, %g2

	/* Here len >= 256 and condition codes reflect execution
	 * of "andcc %o0, 0x7, %g2", done by caller.
	 */
	.align		64
enter:
	/* Is 'dst' already aligned on an 64-byte boundary? */
	be,pt		%XCC, 2f

	/* Compute abs((dst & 0x3f) - 0x40) into %g2.  This is the number
	 * of bytes to copy to make 'dst' 64-byte aligned.  We pre-
	 * subtract this from 'len'.
	 */
	 sub		%g2, 0x40, %g2
	sub		%g0, %g2, %g2
	sub		%o2, %g2, %o2

	/* Copy %g2 bytes from src to dst, one byte at a time. */
1:	ldub		[%o1 + 0x00], %o3
	add		%o1, 0x1, %o1
	add		%o0, 0x1, %o0
	subcc		%g2, 0x1, %g2

	bg,pt		%XCC, 1b
	 stb		%o3, [%o0 + -1]

2:	VISEntryHalf
	and		%o1, 0x7, %g1
	ba,pt		%xcc, begin
	 alignaddr	%o1, %g0, %o1

	.align		64
begin:
	prefetch	[%o1 + 0x000], #one_read
	prefetch	[%o1 + 0x040], #one_read
	andn		%o2, (0x40 - 1), %o4
	prefetch	[%o1 + 0x080], #one_read
	prefetch	[%o1 + 0x0c0], #one_read
	ldd		[%o1 + 0x000], %f0
	prefetch	[%o1 + 0x100], #one_read
	ldd		[%o1 + 0x008], %f2
	prefetch	[%o1 + 0x140], #one_read
	ldd		[%o1 + 0x010], %f4
	prefetch	[%o1 + 0x180], #one_read
	faligndata	%f0, %f2, %f16
	ldd		[%o1 + 0x018], %f6
	faligndata	%f2, %f4, %f18
	ldd		[%o1 + 0x020], %f8
	faligndata	%f4, %f6, %f20
	ldd		[%o1 + 0x028], %f10
	faligndata	%f6, %f8, %f22

	ldd		[%o1 + 0x030], %f12
	faligndata	%f8, %f10, %f24
	ldd		[%o1 + 0x038], %f14
	faligndata	%f10, %f12, %f26
	ldd		[%o1 + 0x040], %f0

	sub		%o4, 0x80, %o4
	add		%o1, 0x40, %o1
	ba,pt		%xcc, loop
	 srl		%o4, 6, %o3

	.align		64
loop:
	ldd		[%o1 + 0x008], %f2
	faligndata	%f12, %f14, %f28
	ldd		[%o1 + 0x010], %f4
	faligndata	%f14, %f0, %f30
	stda		%f16, [%o0] ASI_BLK_P
	ldd		[%o1 + 0x018], %f6
	faligndata	%f0, %f2, %f16

	ldd		[%o1 + 0x020], %f8
	faligndata	%f2, %f4, %f18
	ldd		[%o1 + 0x028], %f10
	faligndata	%f4, %f6, %f20
	ldd		[%o1 + 0x030], %f12
	faligndata	%f6, %f8, %f22
	ldd		[%o1 + 0x038], %f14
	faligndata	%f8, %f10, %f24

	ldd		[%o1 + 0x040], %f0
	prefetch	[%o1 + 0x180], #one_read
	faligndata	%f10, %f12, %f26
	subcc		%o3, 0x01, %o3
	add		%o1, 0x40, %o1
	bg,pt		%XCC, loop
	 add		%o0, 0x40, %o0

	/* Finally we copy the last full 64-byte block. */
loopfini:
	ldd		[%o1 + 0x008], %f2
	faligndata	%f12, %f14, %f28
	ldd		[%o1 + 0x010], %f4
	faligndata	%f14, %f0, %f30
	stda		%f16, [%o0] ASI_BLK_P
	ldd		[%o1 + 0x018], %f6
	faligndata	%f0, %f2, %f16
	ldd		[%o1 + 0x020], %f8
	faligndata	%f2, %f4, %f18
	ldd		[%o1 + 0x028], %f10
	faligndata	%f4, %f6, %f20
	ldd		[%o1 + 0x030], %f12
	faligndata	%f6, %f8, %f22
	ldd		[%o1 + 0x038], %f14
	faligndata	%f8, %f10, %f24
	cmp		%g1, 0
	be,pt		%XCC, 1f
	 add		%o0, 0x40, %o0
	ldd		[%o1 + 0x040], %f0
1:	faligndata	%f10, %f12, %f26
	faligndata	%f12, %f14, %f28
	faligndata	%f14, %f0, %f30
	stda		%f16, [%o0] ASI_BLK_P
	add		%o0, 0x40, %o0
	add		%o1, 0x40, %o1
	membar		#Sync

	/* Now we copy the (len modulo 64) bytes at the end.
	 * Note how we borrow the %f0 loaded above.
	 *
	 * Also notice how this code is careful not to perform a
	 * load past the end of the src buffer.
	 */
loopend:
	and		%o2, 0x3f, %o2
	andcc		%o2, 0x38, %g2
	be,pn		%XCC, endcruft
	 subcc		%g2, 0x8, %g2
	be,pn		%XCC, endcruft
	 cmp		%g1, 0

	be,a,pt		%XCC, 1f
	 ldd		[%o1 + 0x00], %f0

1:	ldd		[%o1 + 0x08], %f2
	add		%o1, 0x8, %o1
	sub		%o2, 0x8, %o2
	subcc		%g2, 0x8, %g2
	faligndata	%f0, %f2, %f8
	std		%f8, [%o0 + 0x00]
	be,pn		%XCC, endcruft
	 add		%o0, 0x8, %o0
	ldd		[%o1 + 0x08], %f0
	add		%o1, 0x8, %o1
	sub		%o2, 0x8, %o2
	subcc		%g2, 0x8, %g2
	faligndata	%f2, %f0, %f8
	std		%f8, [%o0 + 0x00]
	bne,pn		%XCC, 1b
	 add		%o0, 0x8, %o0

	/* If anything is left, we copy it one byte at a time.
	 * Note that %g1 is (src & 0x3) saved above before the
	 * alignaddr was performed.
	 */
endcruft:
	cmp		%o2, 0
	add		%o1, %g1, %o1
	VISExitHalf
	be,pn		%XCC, out
	 sub		%o0, %o1, %o3

	andcc		%g1, 0x7, %g0
	bne,pn		%icc, small_copy_unaligned
	 andcc		%o2, 0x8, %g0
	be,pt		%icc, 1f
	 nop
	ldx		[%o1], %o5
	stx		%o5, [%o1 + %o3]
	add		%o1, 0x8, %o1

1:	andcc		%o2, 0x4, %g0
	be,pt		%icc, 1f
	 nop
	lduw		[%o1], %o5
	stw		%o5, [%o1 + %o3]
	add		%o1, 0x4, %o1

1:	andcc		%o2, 0x2, %g0
	be,pt		%icc, 1f
	 nop
	lduh		[%o1], %o5
	sth		%o5, [%o1 + %o3]
	add		%o1, 0x2, %o1

1:	andcc		%o2, 0x1, %g0
	be,pt		%icc, out
	 nop
	ldub		[%o1], %o5
	ba,pt		%xcc, out
	 stb		%o5, [%o1 + %o3]

medium_copy: /* 16 < len <= 64 */
	bne,pn		%XCC, small_copy_unaligned
	 sub		%o0, %o1, %o3

medium_copy_aligned:
	andn		%o2, 0x7, %o4
	and		%o2, 0x7, %o2
1:	subcc		%o4, 0x8, %o4
	ldx		[%o1], %o5
	stx		%o5, [%o1 + %o3]
	bgu,pt		%XCC, 1b
	 add		%o1, 0x8, %o1
	andcc		%o2, 0x4, %g0
	be,pt		%XCC, 1f
	 nop
	sub		%o2, 0x4, %o2
	lduw		[%o1], %o5
	stw		%o5, [%o1 + %o3]
	add		%o1, 0x4, %o1
1:	cmp		%o2, 0
	be,pt		%XCC, out
	 nop
	ba,pt		%xcc, small_copy_unaligned
	 nop

small_copy: /* 0 < len <= 16 */
	andcc		%o3, 0x3, %g0
	bne,pn		%XCC, small_copy_unaligned
	 sub		%o0, %o1, %o3

small_copy_aligned:
	subcc		%o2, 4, %o2
	lduw		[%o1], %g1
	stw		%g1, [%o1 + %o3]
	bgu,pt		%XCC, small_copy_aligned
	 add		%o1, 4, %o1

out:	retl
	 mov		%g5, %o0

	.align	32
small_copy_unaligned:
	subcc		%o2, 1, %o2
	ldub		[%o1], %g1
	stb		%g1, [%o1 + %o3]
	bgu,pt		%XCC, small_copy_unaligned
	 add		%o1, 1, %o1
	retl
	 mov		%g5, %o0

END(__memcpy_ultra3)

#endif
