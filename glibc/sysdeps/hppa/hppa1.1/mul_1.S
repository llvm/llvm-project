;! HP-PA-1.1 __mpn_mul_1 -- Multiply a limb vector with a limb and store
;! the result in a second limb vector.

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

;! This runs at 9 cycles/limb on a PA7000.  With the used instructions, it can
;! not become faster due to data cache contention after a store.  On the
;! PA7100 it runs at 7 cycles/limb, and that can not be improved either, since
;! only the xmpyu does not need the integer pipeline, so the only dual-issue
;! we will get are addc+xmpyu.  Unrolling would not help either CPU.

;! We could use fldds to read two limbs at a time from the S1 array, and that
;! could bring down the times to 8.5 and 6.5 cycles/limb for the PA7000 and
;! PA7100, respectively.  We don't do that since it does not seem worth the
;! (alignment) troubles...

;! At least the PA7100 is rumored to be able to deal with cache-misses
;! without stalling instruction issue.  If this is true, and the cache is
;! actually also lockup-free, we should use a deeper software pipeline, and
;! load from S1 very early;  (The loads and stores to -12(sp) will surely be
;! in the cache.)

	.text
	.export		__mpn_mul_1
__mpn_mul_1:
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
	fstds	 	%fr6,-16(%r30)
	xmpyu		%fr4,%fr7,%fr8
	ldw		-12(%r30),%r20		;! least significant limb in product
	ldw		-16(%r30),%r28

	fstds		%fr8,-16(%r30)
	addib,=		-1,%r24,L$end
	 ldw		-12(%r30),%r1

;! Main loop
L$loop:
	fldws,ma	4(%r25),%fr5
	stws,ma		%r20,4(%r26)
	addc		%r28,%r1,%r20
	xmpyu		%fr4,%fr5,%fr6
	ldw		-16(%r30),%r28
	fstds		%fr6,-16(%r30)
	addib,<>	-1,%r24,L$loop
	 ldw		-12(%r30),%r1

L$end:
	stws,ma		%r20,4(%r26)
	addc		%r28,%r1,%r20
	ldw		-16(%r30),%r28
	stws,ma		%r20,4(%r26)
	addc		%r0,%r28,%r28
	bv		0(%r2)
	 ldo		-64(%r30),%r30

L$just_one_limb:
	xmpyu		%fr4,%fr5,%fr6
	fstds		%fr6,-16(%r30)
	ldw		-16(%r30),%r28
	ldo		-64(%r30),%r30
	bv		0(%r2)
	 fstws		%fr6R,0(%r26)

	.exit
	.procend
