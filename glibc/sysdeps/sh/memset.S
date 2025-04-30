/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Kazumoto Kojima <kkojima@rr.iij4u.or.jp>
   Optimized by Toshiyasu Morita <toshiyasu.morita@hsa.hitachi.com>

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

/* void *memset (t, c, len);  */

ENTRY(memset)
	mov	#12,r0
	cmp/gt	r6,r0
	bt.s	L_byte_loop_init
	mov	r4,r7

	extu.b	r5,r5
	swap.b	r5,r1
	or	r1,r5
	swap.w	r5,r1
	or	r1,r5

	mov	r4,r0
	tst	#1,r0
	bt	L_wordalign

	mov.b	r5,@r4
	add	#-1,r6
	add	#1,r4
	mov	r4,r0

	.balignw 4,0x0009
L_wordalign:
	tst	#2,r0
	bt	L_word_loop_init

	mov.w	r5,@r4
	add	#-2,r6
	add	#2,r4
	mov	r4,r0

	.balignw 4,0x0009
L_word_loop_init:
	mov	r6,r3
	shlr2	r3
	mov	#7,r0
	shlr	r3
	and	r0,r6

	.balignw 4,0x0009
L_2word_loop:
	mov.l	r5,@r4
	dt	r3
	mov.l	r5,@(4,r4)
	bf.s	L_2word_loop
	add	#8,r4

	.balignw 4,0x0009
L_byte_loop_init:
	tst	r6,r6
	bt	L_byte_exit

	.balignw 4,0x0009
L_byte_loop:
	mov.b	r5,@r4
	dt	r6
	bf.s	L_byte_loop
	add	#1,r4

	.balignw 4,0x0009
L_byte_exit:
	rts
	mov	r7,r0
END(memset)
libc_hidden_builtin_def (memset)
