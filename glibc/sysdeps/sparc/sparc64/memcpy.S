/* Copy SIZE bytes from SRC to DEST.
   For UltraSPARC.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David S. Miller (davem@caip.rutgers.edu) and
		  Jakub Jelinek (jakub@redhat.com).

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
#include <asm/asi.h>
#ifndef XCC
#define USE_BPR
	.register	%g2, #scratch
	.register	%g3, #scratch
	.register	%g6, #scratch
#define XCC	xcc
#endif
#define FPRS_FEF	4

#define FREG_FROB(f1, f2, f3, f4, f5, f6, f7, f8, f9)		\
	faligndata	%f1, %f2, %f48;				\
	faligndata	%f2, %f3, %f50;				\
	faligndata	%f3, %f4, %f52;				\
	faligndata	%f4, %f5, %f54;				\
	faligndata	%f5, %f6, %f56;				\
	faligndata	%f6, %f7, %f58;				\
	faligndata	%f7, %f8, %f60;				\
	faligndata	%f8, %f9, %f62;

#define MAIN_LOOP_CHUNK(src, dest, fdest, fsrc, len, jmptgt)	\
	ldda		[%src] %asi, %fdest;			\
	add		%src, 0x40, %src;			\
	add		%dest, 0x40, %dest;			\
	subcc		%len, 0x40, %len;			\
	be,pn		%xcc, jmptgt;				\
	 stda		%fsrc, [%dest - 0x40] %asi;

#define LOOP_CHUNK1(src, dest, len, branch_dest)		\
	MAIN_LOOP_CHUNK(src, dest, f0,  f48, len, branch_dest)
#define LOOP_CHUNK2(src, dest, len, branch_dest)		\
	MAIN_LOOP_CHUNK(src, dest, f16, f48, len, branch_dest)
#define LOOP_CHUNK3(src, dest, len, branch_dest)		\
	MAIN_LOOP_CHUNK(src, dest, f32, f48, len, branch_dest)

#define STORE_SYNC(dest, fsrc)					\
	stda		%fsrc, [%dest] %asi;			\
	add		%dest, 0x40, %dest;

#define STORE_JUMP(dest, fsrc, target)				\
	stda		%fsrc, [%dest] %asi;			\
	add		%dest, 0x40, %dest;			\
	ba,pt		%xcc, target;

#define VISLOOP_PAD nop; nop; nop; nop; 			\
		    nop; nop; nop; nop; 			\
		    nop; nop; nop; nop; 			\
		    nop; nop; nop;

#define FINISH_VISCHUNK(dest, f0, f1, left)			\
	subcc		%left, 8, %left;			\
	bl,pn		%xcc, 205f;				\
	 faligndata	%f0, %f1, %f48;				\
	std		%f48, [%dest];				\
	add		%dest, 8, %dest;

#define UNEVEN_VISCHUNK(dest, f0, f1, left)			\
	subcc		%left, 8, %left;			\
	bl,pn		%xcc, 205f;				\
	 fsrc2		%f0, %f1;				\
	ba,a,pt		%xcc, 204f;

	/* Macros for non-VIS memcpy code. */
#define MOVE_BIGCHUNK(src, dst, offset, t0, t1, t2, t3)		\
	ldx		[%src + offset + 0x00], %t0; 		\
	ldx		[%src + offset + 0x08], %t1; 		\
	ldx		[%src + offset + 0x10], %t2; 		\
	ldx		[%src + offset + 0x18], %t3; 		\
	stw		%t0, [%dst + offset + 0x04]; 		\
	srlx		%t0, 32, %t0;				\
	stw		%t0, [%dst + offset + 0x00]; 		\
	stw		%t1, [%dst + offset + 0x0c]; 		\
	srlx		%t1, 32, %t1;				\
	stw		%t1, [%dst + offset + 0x08]; 		\
	stw		%t2, [%dst + offset + 0x14]; 		\
	srlx		%t2, 32, %t2;				\
	stw		%t2, [%dst + offset + 0x10]; 		\
	stw		%t3, [%dst + offset + 0x1c];		\
	srlx		%t3, 32, %t3;				\
	stw		%t3, [%dst + offset + 0x18];

#define MOVE_BIGALIGNCHUNK(src, dst, offset, t0, t1, t2, t3)	\
	ldx		[%src + offset + 0x00], %t0; 		\
	ldx		[%src + offset + 0x08], %t1; 		\
	ldx		[%src + offset + 0x10], %t2; 		\
	ldx		[%src + offset + 0x18], %t3; 		\
	stx		%t0, [%dst + offset + 0x00]; 		\
	stx		%t1, [%dst + offset + 0x08]; 		\
	stx		%t2, [%dst + offset + 0x10]; 		\
	stx		%t3, [%dst + offset + 0x18]; 		\
	ldx		[%src + offset + 0x20], %t0; 		\
	ldx		[%src + offset + 0x28], %t1; 		\
	ldx		[%src + offset + 0x30], %t2; 		\
	ldx		[%src + offset + 0x38], %t3; 		\
	stx		%t0, [%dst + offset + 0x20]; 		\
	stx		%t1, [%dst + offset + 0x28]; 		\
	stx		%t2, [%dst + offset + 0x30]; 		\
	stx		%t3, [%dst + offset + 0x38];

#define MOVE_LASTCHUNK(src, dst, offset, t0, t1, t2, t3)	\
	ldx		[%src - offset - 0x10], %t0;		\
	ldx		[%src - offset - 0x08], %t1; 		\
	stw		%t0, [%dst - offset - 0x0c]; 		\
	srlx		%t0, 32, %t2;				\
	stw		%t2, [%dst - offset - 0x10]; 		\
	stw		%t1, [%dst - offset - 0x04]; 		\
	srlx		%t1, 32, %t3;				\
	stw		%t3, [%dst - offset - 0x08];

#define MOVE_LASTALIGNCHUNK(src, dst, offset, t0, t1)		\
	ldx		[%src - offset - 0x10], %t0; 		\
	ldx		[%src - offset - 0x08], %t1; 		\
	stx		%t0, [%dst - offset - 0x10]; 		\
	stx		%t1, [%dst - offset - 0x08];

	.text
	.align		32
ENTRY(__memcpy_large)
200:	be,pt		%xcc, 201f			/* CTI				*/
	 andcc		%o0, 0x38, %g5			/* IEU1		Group		*/
	mov		8, %g1				/* IEU0				*/
	sub		%g1, %g2, %g2			/* IEU0		Group		*/
	andcc		%o0, 1, %g0			/* IEU1				*/
	be,pt		%icc, 2f			/* CTI				*/
	 sub		%o2, %g2, %o2			/* IEU0		Group		*/
1:	ldub		[%o1], %o5			/* Load		Group		*/
	add		%o1, 1, %o1			/* IEU0				*/
	add		%o0, 1, %o0			/* IEU1				*/
	subcc		%g2, 1, %g2			/* IEU1		Group		*/
	be,pn		%xcc, 3f			/* CTI				*/
	 stb		%o5, [%o0 - 1]			/* Store			*/
2:	ldub		[%o1], %o5			/* Load		Group		*/
	add		%o0, 2, %o0			/* IEU0				*/
	ldub		[%o1 + 1], %g3			/* Load		Group		*/
	subcc		%g2, 2, %g2			/* IEU1		Group		*/
	stb		%o5, [%o0 - 2]			/* Store			*/
	add		%o1, 2, %o1			/* IEU0				*/
	bne,pt		%xcc, 2b			/* CTI		Group		*/
	 stb		%g3, [%o0 - 1]			/* Store			*/
3:	andcc		%o0, 0x38, %g5			/* IEU1		Group		*/
201:	be,pt		%icc, 202f			/* CTI				*/
	 mov		64, %g1				/* IEU0				*/
	fsrc2		%f0, %f2			/* FPU				*/
	sub		%g1, %g5, %g5			/* IEU0		Group		*/
	alignaddr	%o1, %g0, %g1			/* GRU		Group		*/
	ldd		[%g1], %f4			/* Load		Group		*/
	sub		%o2, %g5, %o2			/* IEU0				*/
1:	ldd		[%g1 + 0x8], %f6		/* Load		Group		*/
	add		%g1, 0x8, %g1			/* IEU0		Group		*/
	subcc		%g5, 8, %g5			/* IEU1				*/
	faligndata	%f4, %f6, %f0			/* GRU		Group		*/
	std		%f0, [%o0]			/* Store			*/
	add		%o1, 8, %o1			/* IEU0		Group		*/
	be,pn		%xcc, 202f			/* CTI				*/
	 add		%o0, 8, %o0			/* IEU1				*/
	ldd		[%g1 + 0x8], %f4		/* Load		Group		*/
	add		%g1, 8, %g1			/* IEU0				*/
	subcc		%g5, 8, %g5			/* IEU1				*/
	faligndata	%f6, %f4, %f0			/* GRU		Group		*/
	std		%f0, [%o0]			/* Store			*/
	add		%o1, 8, %o1			/* IEU0				*/
	bne,pt		%xcc, 1b			/* CTI		Group		*/
	 add		%o0, 8, %o0			/* IEU0				*/
202:	membar	  #LoadStore | #StoreStore | #StoreLoad	/* LSU		Group		*/
	wr		%g0, ASI_BLK_P, %asi		/* LSU		Group		*/
	subcc		%o2, 0x40, %g6			/* IEU1		Group		*/
	mov		%o1, %g1			/* IEU0				*/
	andncc		%g6, (0x40 - 1), %g6		/* IEU1		Group		*/
	srl		%g1, 3, %g2			/* IEU0				*/
	sub		%o2, %g6, %g3			/* IEU0		Group		*/
	andn		%o1, (0x40 - 1), %o1		/* IEU1				*/
	and		%g2, 7, %g2			/* IEU0		Group		*/
	andncc		%g3, 0x7, %g3			/* IEU1				*/
	fsrc2		%f0, %f2			/* FPU				*/
	sub		%g3, 0x10, %g3			/* IEU0		Group		*/
	sub		%o2, %g6, %o2			/* IEU1				*/
	alignaddr	%g1, %g0, %g0			/* GRU		Group		*/
	add		%g1, %g6, %g1			/* IEU0		Group		*/
	subcc		%o2, %g3, %o2			/* IEU1				*/
	ldda		[%o1 + 0x00] %asi, %f0		/* LSU		Group		*/
	add		%g1, %g3, %g1			/* IEU0				*/
	ldda		[%o1 + 0x40] %asi, %f16		/* LSU		Group		*/
	sub		%g6, 0x80, %g6			/* IEU0				*/
	ldda		[%o1 + 0x80] %asi, %f32		/* LSU		Group		*/
							/* Clk1		Group 8-(	*/
							/* Clk2		Group 8-(	*/
							/* Clk3		Group 8-(	*/
							/* Clk4		Group 8-(	*/
203:	rd		%pc, %g5			/* PDU		Group 8-(	*/
	addcc		%g5, %lo(300f - 203b), %g5	/* IEU1		Group		*/
	sll		%g2, 9, %g2			/* IEU0				*/
	jmpl		%g5 + %g2, %g0			/* CTI		Group brk forced*/
	 addcc		%o1, 0xc0, %o1			/* IEU1		Group		*/

	.align		512		/* OK, here comes the fun part... */
300:	FREG_FROB(f0, f2, f4, f6, f8, f10,f12,f14,f16)	LOOP_CHUNK1(o1, o0, g6, 301f)
	FREG_FROB(f16,f18,f20,f22,f24,f26,f28,f30,f32)	LOOP_CHUNK2(o1, o0, g6, 302f)
	FREG_FROB(f32,f34,f36,f38,f40,f42,f44,f46,f0)	LOOP_CHUNK3(o1, o0, g6, 303f)
	b,pt		%xcc, 300b+4; faligndata %f0, %f2, %f48
301:	FREG_FROB(f16,f18,f20,f22,f24,f26,f28,f30,f32)	STORE_SYNC(o0, f48) membar #Sync
	FREG_FROB(f32,f34,f36,f38,f40,f42,f44,f46,f0)	STORE_JUMP(o0, f48, 400f) membar #Sync
302:	FREG_FROB(f32,f34,f36,f38,f40,f42,f44,f46,f0)	STORE_SYNC(o0, f48) membar #Sync
	FREG_FROB(f0, f2, f4, f6, f8, f10,f12,f14,f16)	STORE_JUMP(o0, f48, 416f) membar #Sync
303:	FREG_FROB(f0, f2, f4, f6, f8, f10,f12,f14,f16)	STORE_SYNC(o0, f48) membar #Sync
	FREG_FROB(f16,f18,f20,f22,f24,f26,f28,f30,f32)	STORE_JUMP(o0, f48, 432f) membar #Sync
	VISLOOP_PAD
310:	FREG_FROB(f2, f4, f6, f8, f10,f12,f14,f16,f18)	LOOP_CHUNK1(o1, o0, g6, 311f)
	FREG_FROB(f18,f20,f22,f24,f26,f28,f30,f32,f34)	LOOP_CHUNK2(o1, o0, g6, 312f)
	FREG_FROB(f34,f36,f38,f40,f42,f44,f46,f0, f2)	LOOP_CHUNK3(o1, o0, g6, 313f)
	b,pt		%xcc, 310b+4; faligndata %f2, %f4, %f48
311:	FREG_FROB(f18,f20,f22,f24,f26,f28,f30,f32,f34)	STORE_SYNC(o0, f48) membar #Sync
	FREG_FROB(f34,f36,f38,f40,f42,f44,f46,f0, f2)	STORE_JUMP(o0, f48, 402f) membar #Sync
312:	FREG_FROB(f34,f36,f38,f40,f42,f44,f46,f0, f2)	STORE_SYNC(o0, f48) membar #Sync
	FREG_FROB(f2, f4, f6, f8, f10,f12,f14,f16,f18)	STORE_JUMP(o0, f48, 418f) membar #Sync
313:	FREG_FROB(f2, f4, f6, f8, f10,f12,f14,f16,f18)	STORE_SYNC(o0, f48) membar #Sync
	FREG_FROB(f18,f20,f22,f24,f26,f28,f30,f32,f34)	STORE_JUMP(o0, f48, 434f) membar #Sync
	VISLOOP_PAD
320:	FREG_FROB(f4, f6, f8, f10,f12,f14,f16,f18,f20)	LOOP_CHUNK1(o1, o0, g6, 321f)
	FREG_FROB(f20,f22,f24,f26,f28,f30,f32,f34,f36)	LOOP_CHUNK2(o1, o0, g6, 322f)
	FREG_FROB(f36,f38,f40,f42,f44,f46,f0, f2, f4)	LOOP_CHUNK3(o1, o0, g6, 323f)
	b,pt		%xcc, 320b+4; faligndata %f4, %f6, %f48
321:	FREG_FROB(f20,f22,f24,f26,f28,f30,f32,f34,f36)	STORE_SYNC(o0, f48) membar #Sync
	FREG_FROB(f36,f38,f40,f42,f44,f46,f0, f2, f4)	STORE_JUMP(o0, f48, 404f) membar #Sync
322:	FREG_FROB(f36,f38,f40,f42,f44,f46,f0, f2, f4)	STORE_SYNC(o0, f48) membar #Sync
	FREG_FROB(f4, f6, f8, f10,f12,f14,f16,f18,f20)	STORE_JUMP(o0, f48, 420f) membar #Sync
323:	FREG_FROB(f4, f6, f8, f10,f12,f14,f16,f18,f20)	STORE_SYNC(o0, f48) membar #Sync
	FREG_FROB(f20,f22,f24,f26,f28,f30,f32,f34,f36)	STORE_JUMP(o0, f48, 436f) membar #Sync
	VISLOOP_PAD
330:	FREG_FROB(f6, f8, f10,f12,f14,f16,f18,f20,f22)	LOOP_CHUNK1(o1, o0, g6, 331f)
	FREG_FROB(f22,f24,f26,f28,f30,f32,f34,f36,f38)	LOOP_CHUNK2(o1, o0, g6, 332f)
	FREG_FROB(f38,f40,f42,f44,f46,f0, f2, f4, f6)	LOOP_CHUNK3(o1, o0, g6, 333f)
	b,pt		%xcc, 330b+4; faligndata %f6, %f8, %f48
331:	FREG_FROB(f22,f24,f26,f28,f30,f32,f34,f36,f38)	STORE_SYNC(o0, f48) membar #Sync
	FREG_FROB(f38,f40,f42,f44,f46,f0, f2, f4, f6)	STORE_JUMP(o0, f48, 406f) membar #Sync
332:	FREG_FROB(f38,f40,f42,f44,f46,f0, f2, f4, f6)	STORE_SYNC(o0, f48) membar #Sync
	FREG_FROB(f6, f8, f10,f12,f14,f16,f18,f20,f22)	STORE_JUMP(o0, f48, 422f) membar #Sync
333:	FREG_FROB(f6, f8, f10,f12,f14,f16,f18,f20,f22)	STORE_SYNC(o0, f48) membar #Sync
	FREG_FROB(f22,f24,f26,f28,f30,f32,f34,f36,f38)	STORE_JUMP(o0, f48, 438f) membar #Sync
	VISLOOP_PAD
340:	FREG_FROB(f8, f10,f12,f14,f16,f18,f20,f22,f24)	LOOP_CHUNK1(o1, o0, g6, 341f)
	FREG_FROB(f24,f26,f28,f30,f32,f34,f36,f38,f40)	LOOP_CHUNK2(o1, o0, g6, 342f)
	FREG_FROB(f40,f42,f44,f46,f0, f2, f4, f6, f8)	LOOP_CHUNK3(o1, o0, g6, 343f)
	b,pt		%xcc, 340b+4; faligndata %f8, %f10, %f48
341:	FREG_FROB(f24,f26,f28,f30,f32,f34,f36,f38,f40)	STORE_SYNC(o0, f48) membar #Sync
	FREG_FROB(f40,f42,f44,f46,f0, f2, f4, f6, f8)	STORE_JUMP(o0, f48, 408f) membar #Sync
342:	FREG_FROB(f40,f42,f44,f46,f0, f2, f4, f6, f8)	STORE_SYNC(o0, f48) membar #Sync
	FREG_FROB(f8, f10,f12,f14,f16,f18,f20,f22,f24)	STORE_JUMP(o0, f48, 424f) membar #Sync
343:	FREG_FROB(f8, f10,f12,f14,f16,f18,f20,f22,f24)	STORE_SYNC(o0, f48) membar #Sync
	FREG_FROB(f24,f26,f28,f30,f32,f34,f36,f38,f40)	STORE_JUMP(o0, f48, 440f) membar #Sync
	VISLOOP_PAD
350:	FREG_FROB(f10,f12,f14,f16,f18,f20,f22,f24,f26)	LOOP_CHUNK1(o1, o0, g6, 351f)
	FREG_FROB(f26,f28,f30,f32,f34,f36,f38,f40,f42)	LOOP_CHUNK2(o1, o0, g6, 352f)
	FREG_FROB(f42,f44,f46,f0, f2, f4, f6, f8, f10)	LOOP_CHUNK3(o1, o0, g6, 353f)
	b,pt		%xcc, 350b+4; faligndata %f10, %f12, %f48
351:	FREG_FROB(f26,f28,f30,f32,f34,f36,f38,f40,f42)	STORE_SYNC(o0, f48) membar #Sync
	FREG_FROB(f42,f44,f46,f0, f2, f4, f6, f8, f10)	STORE_JUMP(o0, f48, 410f) membar #Sync
352:	FREG_FROB(f42,f44,f46,f0, f2, f4, f6, f8, f10)	STORE_SYNC(o0, f48) membar #Sync
	FREG_FROB(f10,f12,f14,f16,f18,f20,f22,f24,f26)	STORE_JUMP(o0, f48, 426f) membar #Sync
353:	FREG_FROB(f10,f12,f14,f16,f18,f20,f22,f24,f26)	STORE_SYNC(o0, f48) membar #Sync
	FREG_FROB(f26,f28,f30,f32,f34,f36,f38,f40,f42)	STORE_JUMP(o0, f48, 442f) membar #Sync
	VISLOOP_PAD
360:	FREG_FROB(f12,f14,f16,f18,f20,f22,f24,f26,f28)	LOOP_CHUNK1(o1, o0, g6, 361f)
	FREG_FROB(f28,f30,f32,f34,f36,f38,f40,f42,f44)	LOOP_CHUNK2(o1, o0, g6, 362f)
	FREG_FROB(f44,f46,f0, f2, f4, f6, f8, f10,f12)	LOOP_CHUNK3(o1, o0, g6, 363f)
	b,pt		%xcc, 360b+4; faligndata %f12, %f14, %f48
361:	FREG_FROB(f28,f30,f32,f34,f36,f38,f40,f42,f44)	STORE_SYNC(o0, f48) membar #Sync
	FREG_FROB(f44,f46,f0, f2, f4, f6, f8, f10,f12)	STORE_JUMP(o0, f48, 412f) membar #Sync
362:	FREG_FROB(f44,f46,f0, f2, f4, f6, f8, f10,f12)	STORE_SYNC(o0, f48) membar #Sync
	FREG_FROB(f12,f14,f16,f18,f20,f22,f24,f26,f28)	STORE_JUMP(o0, f48, 428f) membar #Sync
363:	FREG_FROB(f12,f14,f16,f18,f20,f22,f24,f26,f28)	STORE_SYNC(o0, f48) membar #Sync
	FREG_FROB(f28,f30,f32,f34,f36,f38,f40,f42,f44)	STORE_JUMP(o0, f48, 444f) membar #Sync
	VISLOOP_PAD
370:	FREG_FROB(f14,f16,f18,f20,f22,f24,f26,f28,f30)	LOOP_CHUNK1(o1, o0, g6, 371f)
	FREG_FROB(f30,f32,f34,f36,f38,f40,f42,f44,f46)	LOOP_CHUNK2(o1, o0, g6, 372f)
	FREG_FROB(f46,f0, f2, f4, f6, f8, f10,f12,f14)	LOOP_CHUNK3(o1, o0, g6, 373f)
	b,pt		%xcc, 370b+4; faligndata %f14, %f16, %f48
371:	FREG_FROB(f30,f32,f34,f36,f38,f40,f42,f44,f46)	STORE_SYNC(o0, f48) membar #Sync
	FREG_FROB(f46,f0, f2, f4, f6, f8, f10,f12,f14)	STORE_JUMP(o0, f48, 414f) membar #Sync
372:	FREG_FROB(f46,f0, f2, f4, f6, f8, f10,f12,f14)	STORE_SYNC(o0, f48) membar #Sync
	FREG_FROB(f14,f16,f18,f20,f22,f24,f26,f28,f30)	STORE_JUMP(o0, f48, 430f) membar #Sync
373:	FREG_FROB(f14,f16,f18,f20,f22,f24,f26,f28,f30)	STORE_SYNC(o0, f48) membar #Sync
	FREG_FROB(f30,f32,f34,f36,f38,f40,f42,f44,f46)	STORE_JUMP(o0, f48, 446f) membar #Sync
	VISLOOP_PAD
400:	FINISH_VISCHUNK(o0, f0,  f2,  g3)
402:	FINISH_VISCHUNK(o0, f2,  f4,  g3)
404:	FINISH_VISCHUNK(o0, f4,  f6,  g3)
406:	FINISH_VISCHUNK(o0, f6,  f8,  g3)
408:	FINISH_VISCHUNK(o0, f8,  f10, g3)
410:	FINISH_VISCHUNK(o0, f10, f12, g3)
412:	FINISH_VISCHUNK(o0, f12, f14, g3)
414:	UNEVEN_VISCHUNK(o0, f14, f0,  g3)
416:	FINISH_VISCHUNK(o0, f16, f18, g3)
418:	FINISH_VISCHUNK(o0, f18, f20, g3)
420:	FINISH_VISCHUNK(o0, f20, f22, g3)
422:	FINISH_VISCHUNK(o0, f22, f24, g3)
424:	FINISH_VISCHUNK(o0, f24, f26, g3)
426:	FINISH_VISCHUNK(o0, f26, f28, g3)
428:	FINISH_VISCHUNK(o0, f28, f30, g3)
430:	UNEVEN_VISCHUNK(o0, f30, f0,  g3)
432:	FINISH_VISCHUNK(o0, f32, f34, g3)
434:	FINISH_VISCHUNK(o0, f34, f36, g3)
436:	FINISH_VISCHUNK(o0, f36, f38, g3)
438:	FINISH_VISCHUNK(o0, f38, f40, g3)
440:	FINISH_VISCHUNK(o0, f40, f42, g3)
442:	FINISH_VISCHUNK(o0, f42, f44, g3)
444:	FINISH_VISCHUNK(o0, f44, f46, g3)
446:	UNEVEN_VISCHUNK(o0, f46, f0,  g3)
204:	ldd		[%o1], %f2			/* Load		Group		*/
	add		%o1, 8, %o1			/* IEU0				*/
	subcc		%g3, 8, %g3			/* IEU1				*/
	faligndata	%f0, %f2, %f8			/* GRU		Group		*/
	std		%f8, [%o0]			/* Store			*/
	bl,pn		%xcc, 205f			/* CTI				*/
	 add		%o0, 8, %o0			/* IEU0		Group		*/
	ldd		[%o1], %f0			/* Load		Group		*/
	add		%o1, 8, %o1			/* IEU0				*/
	subcc		%g3, 8, %g3			/* IEU1				*/
	faligndata	%f2, %f0, %f8			/* GRU		Group		*/
	std		%f8, [%o0]			/* Store			*/
	bge,pt		%xcc, 204b			/* CTI				*/
	 add		%o0, 8, %o0			/* IEU0		Group		*/
205:	brz,pt		%o2, 207f			/* CTI		Group		*/
	 mov		%g1, %o1			/* IEU0				*/
206:	ldub		[%o1], %g5			/* LOAD				*/
	add		%o1, 1, %o1			/* IEU0				*/
	add		%o0, 1, %o0			/* IEU1				*/
	subcc		%o2, 1, %o2			/* IEU1				*/
	bne,pt		%xcc, 206b			/* CTI				*/
	 stb		%g5, [%o0 - 1]			/* Store	Group		*/
207:	membar		#StoreLoad | #StoreStore	/* LSU		Group		*/
	wr		%g0, FPRS_FEF, %fprs
	retl
	 mov		%g4, %o0

208:	andcc		%o2, 1, %g0			/* IEU1		Group		*/
	be,pt		%icc, 2f+4			/* CTI				*/
1:	 ldub		[%o1], %g5			/* LOAD		Group		*/
	add		%o1, 1, %o1			/* IEU0				*/
	add		%o0, 1, %o0			/* IEU1				*/
	subcc		%o2, 1, %o2			/* IEU1		Group		*/
	be,pn		%xcc, 209f			/* CTI				*/
	 stb		%g5, [%o0 - 1]			/* Store			*/
2:	ldub		[%o1], %g5			/* LOAD		Group		*/
	add		%o0, 2, %o0			/* IEU0				*/
	ldub		[%o1 + 1], %o5			/* LOAD		Group		*/
	add		%o1, 2, %o1			/* IEU0				*/
	subcc		%o2, 2, %o2			/* IEU1		Group		*/
	stb		%g5, [%o0 - 2]			/* Store			*/
	bne,pt		%xcc, 2b			/* CTI				*/
	 stb		%o5, [%o0 - 1]			/* Store			*/
209:	retl
	 mov		%g4, %o0
END(__memcpy_large)

ENTRY(__mempcpy)
	ba,pt		%xcc, 210f
	 add		%o0, %o2, %g4
END(__mempcpy)

	.align		32
ENTRY(memcpy)
	 mov		%o0, %g4			/* IEU0		Group		*/
210:
#ifndef USE_BPR
	srl		%o2, 0, %o2			/* IEU1				*/
#endif
	brz,pn		%o2, 209b			/* CTI		Group		*/
218:	 cmp		%o2, 15				/* IEU1				*/
	bleu,pn		%xcc, 208b			/* CTI		Group		*/
	 cmp		%o2, (64 * 6)			/* IEU1				*/
	bgeu,pn		%xcc, 200b			/* CTI		Group		*/
	 andcc		%o0, 7, %g2			/* IEU1				*/
	sub		%o0, %o1, %g5			/* IEU0				*/
	andcc		%g5, 3, %o5			/* IEU1		Group		*/
	bne,pn		%xcc, 212f			/* CTI				*/
	 andcc		%o1, 3, %g0			/* IEU1		Group		*/
	be,a,pt		%xcc, 216f			/* CTI				*/
	 andcc		%o1, 4, %g0			/* IEU1		Group		*/
	andcc		%o1, 1, %g0			/* IEU1		Group		*/
	be,pn		%xcc, 4f			/* CTI				*/
	 andcc		%o1, 2, %g0			/* IEU1		Group		*/
	ldub		[%o1], %g2			/* Load		Group		*/
	add		%o1, 1, %o1			/* IEU0				*/
	add		%o0, 1, %o0			/* IEU1				*/
	sub		%o2, 1, %o2			/* IEU0		Group		*/
	bne,pn		%xcc, 5f			/* CTI		Group		*/
	 stb		%g2, [%o0 - 1]			/* Store			*/
4:	lduh		[%o1], %g2			/* Load		Group		*/
	add		%o1, 2, %o1			/* IEU0				*/
	add		%o0, 2, %o0			/* IEU1				*/
	sub		%o2, 2, %o2			/* IEU0				*/
	sth		%g2, [%o0 - 2]			/* Store	Group + bubble	*/
5:	andcc		%o1, 4, %g0			/* IEU1				*/
216:	be,a,pn		%xcc, 2f			/* CTI				*/
	 andcc		%o2, -128, %g6			/* IEU1		Group		*/
	lduw		[%o1], %g5			/* Load		Group		*/
	add		%o1, 4, %o1			/* IEU0				*/
	add		%o0, 4, %o0			/* IEU1				*/
	sub		%o2, 4, %o2			/* IEU0		Group		*/
	stw		%g5, [%o0 - 4]			/* Store			*/
	andcc		%o2, -128, %g6			/* IEU1		Group		*/
2:	be,pn		%xcc, 215f			/* CTI				*/
	 andcc		%o0, 4, %g0			/* IEU1		Group		*/
	be,pn		%xcc, 82f + 4			/* CTI		Group		*/
5:	MOVE_BIGCHUNK(o1, o0, 0x00, g1, g3, g5, o5)
	MOVE_BIGCHUNK(o1, o0, 0x20, g1, g3, g5, o5)
	MOVE_BIGCHUNK(o1, o0, 0x40, g1, g3, g5, o5)
	MOVE_BIGCHUNK(o1, o0, 0x60, g1, g3, g5, o5)
35:	subcc		%g6, 128, %g6			/* IEU1		Group		*/
	add		%o1, 128, %o1			/* IEU0				*/
	bne,pt		%xcc, 5b			/* CTI				*/
	 add		%o0, 128, %o0			/* IEU0		Group		*/
215:	andcc		%o2, 0x70, %g6			/* IEU1		Group		*/
41:	be,pn		%xcc, 80f			/* CTI				*/
	 andcc		%o2, 8, %g0			/* IEU1		Group		*/
							/* Clk1 8-(			*/
							/* Clk2 8-(			*/
							/* Clk3 8-(			*/
							/* Clk4 8-(			*/
79:	rd		%pc, %o5			/* PDU		Group		*/
	sll		%g6, 1, %g5			/* IEU0		Group		*/
	add		%o1, %g6, %o1			/* IEU1				*/
	sub		%o5, %g5, %o5			/* IEU0  	Group		*/
	jmpl		%o5 + %lo(80f - 79b), %g0	/* CTI		Group brk forced*/
	 add		%o0, %g6, %o0			/* IEU0		Group		*/
36:	MOVE_LASTCHUNK(o1, o0, 0x60, g2, g3, g5, o5)
	MOVE_LASTCHUNK(o1, o0, 0x50, g2, g3, g5, o5)
	MOVE_LASTCHUNK(o1, o0, 0x40, g2, g3, g5, o5)
	MOVE_LASTCHUNK(o1, o0, 0x30, g2, g3, g5, o5)
	MOVE_LASTCHUNK(o1, o0, 0x20, g2, g3, g5, o5)
	MOVE_LASTCHUNK(o1, o0, 0x10, g2, g3, g5, o5)
	MOVE_LASTCHUNK(o1, o0, 0x00, g2, g3, g5, o5)
80:	be,pt		%xcc, 81f			/* CTI				*/
	 andcc		%o2, 4, %g0			/* IEU1				*/
	ldx		[%o1], %g2			/* Load		Group		*/
	add		%o0, 8, %o0			/* IEU0				*/
	stw		%g2, [%o0 - 0x4]		/* Store	Group		*/
	add		%o1, 8, %o1			/* IEU1				*/
	srlx		%g2, 32, %g2			/* IEU0		Group		*/
	stw		%g2, [%o0 - 0x8]		/* Store			*/
81:	be,pt		%xcc, 1f			/* CTI				*/
	 andcc		%o2, 2, %g0			/* IEU1		Group		*/
	lduw		[%o1], %g2			/* Load		Group		*/
	add		%o1, 4, %o1			/* IEU0				*/
	stw		%g2, [%o0]			/* Store	Group		*/
	add		%o0, 4, %o0			/* IEU0				*/
1:	be,pt		%xcc, 1f			/* CTI				*/
	 andcc		%o2, 1, %g0			/* IEU1		Group		*/
	lduh		[%o1], %g2			/* Load		Group		*/
	add		%o1, 2, %o1			/* IEU0				*/
	sth		%g2, [%o0]			/* Store	Group		*/
	add		%o0, 2, %o0			/* IEU0				*/
1:	be,pt		%xcc, 211f			/* CTI				*/
	 nop						/* IEU1				*/
	ldub		[%o1], %g2			/* Load		Group		*/
	stb		%g2, [%o0]			/* Store	Group + bubble	*/
211:	retl
	 mov		%g4, %o0

82:	MOVE_BIGALIGNCHUNK(o1, o0, 0x00, g1, g3, g5, o5)
	MOVE_BIGALIGNCHUNK(o1, o0, 0x40, g1, g3, g5, o5)
37:	subcc		%g6, 128, %g6			/* IEU1		Group		*/
	add		%o1, 128, %o1			/* IEU0				*/
	bne,pt		%xcc, 82b			/* CTI				*/
	 add		%o0, 128, %o0			/* IEU0		Group		*/
	andcc		%o2, 0x70, %g6			/* IEU1				*/
	be,pn		%xcc, 84f			/* CTI				*/
	 andcc		%o2, 8, %g0			/* IEU1		Group		*/
							/* Clk1 8-(			*/
							/* Clk2 8-(			*/
							/* Clk3 8-(			*/
							/* Clk4 8-(			*/
83:	rd		%pc, %o5			/* PDU		Group		*/
	add		%o1, %g6, %o1			/* IEU0		Group		*/
	sub		%o5, %g6, %o5			/* IEU1				*/
	jmpl		%o5 + %lo(84f - 83b), %g0	/* CTI		Group brk forced*/
	 add		%o0, %g6, %o0			/* IEU0		Group		*/
38:	MOVE_LASTALIGNCHUNK(o1, o0, 0x60, g2, g3)
	MOVE_LASTALIGNCHUNK(o1, o0, 0x50, g2, g3)
	MOVE_LASTALIGNCHUNK(o1, o0, 0x40, g2, g3)
	MOVE_LASTALIGNCHUNK(o1, o0, 0x30, g2, g3)
	MOVE_LASTALIGNCHUNK(o1, o0, 0x20, g2, g3)
	MOVE_LASTALIGNCHUNK(o1, o0, 0x10, g2, g3)
	MOVE_LASTALIGNCHUNK(o1, o0, 0x00, g2, g3)
84:	be,pt		%xcc, 85f			/* CTI		Group		*/
	 andcc		%o2, 4, %g0			/* IEU1				*/
	ldx		[%o1], %g2			/* Load		Group		*/
	add		%o0, 8, %o0			/* IEU0				*/
	add		%o1, 8, %o1			/* IEU0		Group		*/
	stx		%g2, [%o0 - 0x8]		/* Store			*/
85:	be,pt		%xcc, 1f			/* CTI				*/
	 andcc		%o2, 2, %g0			/* IEU1		Group		*/
	lduw		[%o1], %g2			/* Load		Group		*/
	add		%o0, 4, %o0			/* IEU0				*/
	add		%o1, 4, %o1			/* IEU0		Group		*/
	stw		%g2, [%o0 - 0x4]		/* Store			*/
1:	be,pt		%xcc, 1f			/* CTI				*/
	 andcc		%o2, 1, %g0			/* IEU1		Group		*/
	lduh		[%o1], %g2			/* Load		Group		*/
	add		%o0, 2, %o0			/* IEU0				*/
	add		%o1, 2, %o1			/* IEU0		Group		*/
	sth		%g2, [%o0 - 0x2]		/* Store			*/
1:	be,pt		%xcc, 1f			/* CTI				*/
	 nop						/* IEU0		Group		*/
	ldub		[%o1], %g2			/* Load		Group		*/
	stb		%g2, [%o0]			/* Store	Group + bubble	*/
1:	retl
	 mov		%g4, %o0

212:	brz,pt		%g2, 2f				/* CTI		Group		*/
	 mov		8, %g1				/* IEU0				*/
	sub		%g1, %g2, %g2			/* IEU0		Group		*/
	sub		%o2, %g2, %o2			/* IEU0		Group		*/
1:	ldub		[%o1], %g5			/* Load		Group		*/
	add		%o1, 1, %o1			/* IEU0				*/
	add		%o0, 1, %o0			/* IEU1				*/
	subcc		%g2, 1, %g2			/* IEU1		Group		*/
	bne,pt		%xcc, 1b			/* CTI				*/
	 stb		%g5, [%o0 - 1]			/* Store			*/
2:	andn		%o2, 7, %g5 			/* IEU0		Group		*/
	and		%o2, 7, %o2			/* IEU1				*/
	fsrc2		%f0, %f2			/* FPU				*/
	alignaddr	%o1, %g0, %g1			/* GRU		Group		*/
	ldd		[%g1], %f4			/* Load		Group		*/
1:	ldd		[%g1 + 0x8], %f6		/* Load		Group		*/
	add		%g1, 0x8, %g1			/* IEU0		Group		*/
	subcc		%g5, 8, %g5			/* IEU1				*/
	faligndata	%f4, %f6, %f0			/* GRU		Group		*/
	std		%f0, [%o0]			/* Store			*/
	add		%o1, 8, %o1			/* IEU0		Group		*/
	be,pn		%xcc, 213f			/* CTI				*/
	 add		%o0, 8, %o0			/* IEU1				*/
	ldd		[%g1 + 0x8], %f4		/* Load		Group		*/
	add		%g1, 8, %g1			/* IEU0				*/
	subcc		%g5, 8, %g5			/* IEU1				*/
	faligndata	%f6, %f4, %f0			/* GRU		Group		*/
	std		%f0, [%o0]			/* Store			*/
	add		%o1, 8, %o1			/* IEU0				*/
	bne,pn		%xcc, 1b			/* CTI		Group		*/
	 add		%o0, 8, %o0			/* IEU0				*/
213:	brz,pn		%o2, 214f			/* CTI		Group		*/
	 nop						/* IEU0				*/
	ldub		[%o1], %g5			/* LOAD				*/
	add		%o1, 1, %o1			/* IEU0				*/
	add		%o0, 1, %o0			/* IEU1				*/
	subcc		%o2, 1, %o2			/* IEU1				*/
	bne,pt		%xcc, 206b			/* CTI				*/
	 stb		%g5, [%o0 - 1]			/* Store	Group		*/
214:	wr		%g0, FPRS_FEF, %fprs
	retl
	 mov		%g4, %o0
END(memcpy)

libc_hidden_builtin_def (memcpy)

libc_hidden_def (__mempcpy)
weak_alias (__mempcpy, mempcpy)
libc_hidden_builtin_def (mempcpy)
