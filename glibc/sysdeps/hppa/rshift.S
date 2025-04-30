;! HP-PA  __mpn_rshift --

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
;! res_ptr	gr26
;! s_ptr	gr25
;! size		gr24
;! cnt		gr23

	.text
	.export		__mpn_rshift
__mpn_rshift:
	.proc
	.callinfo	frame=64,no_calls
	.entry

	ldws,ma		4(%r25),%r22
	mtsar		%r23
	addib,=		-1,%r24,L$0004
	vshd		%r22,%r0,%r28		;! compute carry out limb
	ldws,ma		4(%r25),%r29
	addib,=		-1,%r24,L$0002
	vshd		%r29,%r22,%r20

L$loop:	ldws,ma		4(%r25),%r22
	stws,ma		%r20,4(%r26)
	addib,=		-1,%r24,L$0003
	vshd		%r22,%r29,%r20
	ldws,ma		4(%r25),%r29
	stws,ma		%r20,4(%r26)
	addib,<>	-1,%r24,L$loop
	vshd		%r29,%r22,%r20

L$0002:	stws,ma		%r20,4(%r26)
	vshd		%r0,%r29,%r20
	bv		0(%r2)
	stw		%r20,0(%r26)
L$0003:	stws,ma		%r20,4(%r26)
L$0004:	vshd		%r0,%r22,%r20
	bv		0(%r2)
	stw		%r20,0(%r26)

	.exit
	.procend
