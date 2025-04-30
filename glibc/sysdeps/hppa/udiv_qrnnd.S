;! HP-PA  __udiv_qrnnd division support, used from longlong.h.
;! This version runs fast on pre-PA7000 CPUs.

;! Copyright (C) 1993-2021 Free Software Foundation, Inc.

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
;! rem_ptr	gr26
;! n1		gr25
;! n0		gr24
;! d		gr23

;! The code size is a bit excessive.  We could merge the last two ds;addc
;! sequences by simply moving the "bb,< Odd" instruction down.  The only
;! trouble is the FFFFFFFF code that would need some hacking.

	.text
	.export		__udiv_qrnnd
__udiv_qrnnd:
	.proc
	.callinfo	frame=0,no_calls
	.entry

	comb,<		%r23,%r0,L$largedivisor
	 sub		%r0,%r23,%r1		;! clear cy as side-effect
	ds		%r0,%r1,%r0
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r28
	ds		%r25,%r23,%r25
	comclr,>=	%r25,%r0,%r0
	addl		%r25,%r23,%r25
	stws		%r25,0(%r26)
	bv		0(%r2)
	 addc		%r28,%r28,%r28

L$largedivisor:
	extru		%r24,31,1,%r20		;! r20 = n0 & 1
	bb,<		%r23,31,L$odd
	 extru		%r23,30,31,%r22		;! r22 = d >> 1
	shd		%r25,%r24,1,%r24	;! r24 = new n0
	extru		%r25,30,31,%r25		;! r25 = new n1
	sub		%r0,%r22,%r21
	ds		%r0,%r21,%r0
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	comclr,>=	%r25,%r0,%r0
	addl		%r25,%r22,%r25
	sh1addl		%r25,%r20,%r25
	stws		%r25,0(%r26)
	bv		0(%r2)
	 addc		%r24,%r24,%r28

L$odd:	addib,sv,n	1,%r22,L$FF..		;! r22 = (d / 2 + 1)
	shd		%r25,%r24,1,%r24	;! r24 = new n0
	extru		%r25,30,31,%r25		;! r25 = new n1
	sub		%r0,%r22,%r21
	ds		%r0,%r21,%r0
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r28
	comclr,>=	%r25,%r0,%r0
	addl		%r25,%r22,%r25
	sh1addl		%r25,%r20,%r25
;! We have computed (n1,,n0) / (d + 1), q' = r28, r' = r25
	add,nuv		%r28,%r25,%r25
	addl		%r25,%r1,%r25
	addc		%r0,%r28,%r28
	sub,<<		%r25,%r23,%r0
	addl		%r25,%r1,%r25
	stws		%r25,0(%r26)
	bv		0(%r2)
	 addc		%r0,%r28,%r28

;! This is just a special case of the code above.
;! We come here when d == 0xFFFFFFFF
L$FF..:	add,uv		%r25,%r24,%r24
	sub,<<		%r24,%r23,%r0
	ldo		1(%r24),%r24
	stws		%r24,0(%r26)
	bv		0(%r2)
	 addc		%r0,%r25,%r28

	.exit
	.procend
