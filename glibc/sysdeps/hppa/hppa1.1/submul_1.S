;! HP-PA-1.1 __mpn_submul_1 -- Multiply a limb vector with a limb and
;! subtract the result from a second limb vector.

;! Copyright (C) 1992-2021 Free Software Foundation, Inc.

;! This file is part of the GNU MP Library.

;! The GNU MP Library is free software; you can redistribute it and/or modify
;! it under the terms of the GNU Lesser General Public License as published by
;! the Free Software Foundation; either version 2.1 of the License, or (at your
;! option) any later version.

;! The GNU MP Library is distributed in the hope that it will be useful, but
;! WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
;! or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
;! License for more details.

;! You should have received a copy of the GNU Lesser General Public License
;! along with the GNU MP Library.  If not, see
;! <https://www.gnu.org/licenses/>.


;! INPUT PARAMETERS
;! res_ptr	r26
;! s1_ptr	r25
;! size		r24
;! s2_limb	r23

;! This runs at 12 cycles/limb on a PA7000.  With the used instructions, it
;! can not become faster due to data cache contention after a store.  On the
;! PA7100 it runs at 11 cycles/limb, and that can not be improved either,
;! since only the xmpyu does not need the integer pipeline, so the only
;! dual-issue we will get are addc+xmpyu.  Unrolling could gain a cycle/limb
;! on the PA7100.

;! There are some ideas described in mul_1.s that applies to this code too.

;! It seems possible to make this run as fast as __mpn_addmul_1, if we use
;! 	sub,>>=	%r29,%r20,%r22
;!	addi	1,%r28,%r28
;! but that requires reworking the hairy software pipeline...

	.text
	.export		__mpn_submul_1
__mpn_submul_1:
	.proc
	.callinfo	frame=64,no_calls
	.entry

	ldo		64(%r30),%r30
	fldws,ma	4(%r25),%fr5
	stw		%r23,-16(%r30)		;! move s2_limb ...
	addib,=		-1,%r24,L$just_one_limb
	 fldws		-16(%r30),%fr4		;! ... into fr4
	add		%r0,%r0,%r0		;! clear carry
	xmpyu		%fr4,%fr5,%fr6
	fldws,ma	4(%r25),%fr7
	fstds		%fr6,-16(%r30)
	xmpyu		%fr4,%fr7,%fr8
	ldw		-12(%r30),%r20		;! least significant limb in product
	ldw		-16(%r30),%r28

	fstds		%fr8,-16(%r30)
	addib,=		-1,%r24,L$end
	 ldw		-12(%r30),%r1

;! Main loop
L$loop:
	ldws		0(%r26),%r29
	fldws,ma	4(%r25),%fr5
	sub		%r29,%r20,%r22
	add		%r22,%r20,%r0
	stws,ma		%r22,4(%r26)
	addc		%r28,%r1,%r20
	xmpyu		%fr4,%fr5,%fr6
	ldw		-16(%r30),%r28
	fstds		%fr6,-16(%r30)
	addc		%r0,%r28,%r28
	addib,<>	-1,%r24,L$loop
	 ldw		-12(%r30),%r1

L$end:
	ldw		0(%r26),%r29
	sub		%r29,%r20,%r22
	add		%r22,%r20,%r0
	stws,ma		%r22,4(%r26)
	addc		%r28,%r1,%r20
	ldw		-16(%r30),%r28
	ldws		0(%r26),%r29
	addc		%r0,%r28,%r28
	sub		%r29,%r20,%r22
	add		%r22,%r20,%r0
	stws,ma		%r22,4(%r26)
	addc		%r0,%r28,%r28
	bv		0(%r2)
	 ldo		-64(%r30),%r30

L$just_one_limb:
	xmpyu		%fr4,%fr5,%fr6
	ldw		0(%r26),%r29
	fstds		%fr6,-16(%r30)
	ldw		-12(%r30),%r1
	ldw		-16(%r30),%r28
	sub		%r29,%r1,%r22
	add		%r22,%r1,%r0
	stw		%r22,0(%r26)
	addc		%r0,%r28,%r28
	bv		0(%r2)
	 ldo		-64(%r30),%r30

	.exit
	.procend
