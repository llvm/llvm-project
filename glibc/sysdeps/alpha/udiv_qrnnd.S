 # Alpha 21064 __udiv_qrnnd

 # Copyright (C) 1992-2021 Free Software Foundation, Inc.

 # This file is part of the GNU MP Library.

 # The GNU MP Library is free software; you can redistribute it and/or modify
 # it under the terms of the GNU Lesser General Public License as published by
 # the Free Software Foundation; either version 2.1 of the License, or (at your
 # option) any later version.

 # The GNU MP Library is distributed in the hope that it will be useful, but
 # WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 # or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 # License for more details.

 # You should have received a copy of the GNU Lesser General Public License
 # along with the GNU MP Library.  If not, see <https://www.gnu.org/licenses/>.

#include <sysdep.h>

        .set noreorder
        .set noat

	.text

LEAF(__udiv_qrnnd, 0)
#ifdef PROF
	ldgp	gp, 0(pv)
	lda	AT, _mcount
	jsr	AT, (AT), _mcount
	.prologue 1
#else
        .prologue 0
#endif

#define cnt	$2
#define tmp	$3
#define rem_ptr	$16
#define n1	$17
#define n0	$18
#define d	$19
#define qb	$20

	ldiq	cnt,16
	blt	d,$largedivisor

$loop1:	cmplt	n0,0,tmp
	addq	n1,n1,n1
	bis	n1,tmp,n1
	addq	n0,n0,n0
	cmpule	d,n1,qb
	subq	n1,d,tmp
	cmovne	qb,tmp,n1
	bis	n0,qb,n0
	cmplt	n0,0,tmp
	addq	n1,n1,n1
	bis	n1,tmp,n1
	addq	n0,n0,n0
	cmpule	d,n1,qb
	subq	n1,d,tmp
	cmovne	qb,tmp,n1
	bis	n0,qb,n0
	cmplt	n0,0,tmp
	addq	n1,n1,n1
	bis	n1,tmp,n1
	addq	n0,n0,n0
	cmpule	d,n1,qb
	subq	n1,d,tmp
	cmovne	qb,tmp,n1
	bis	n0,qb,n0
	cmplt	n0,0,tmp
	addq	n1,n1,n1
	bis	n1,tmp,n1
	addq	n0,n0,n0
	cmpule	d,n1,qb
	subq	n1,d,tmp
	cmovne	qb,tmp,n1
	bis	n0,qb,n0
	subq	cnt,1,cnt
	bgt	cnt,$loop1
	stq	n1,0(rem_ptr)
	bis	$31,n0,$0
	ret	$31,($26),1

$largedivisor:
	and	n0,1,$4

	srl	n0,1,n0
	sll	n1,63,tmp
	or	tmp,n0,n0
	srl	n1,1,n1

	and	d,1,$6
	srl	d,1,$5
	addq	$5,$6,$5

$loop2:	cmplt	n0,0,tmp
	addq	n1,n1,n1
	bis	n1,tmp,n1
	addq	n0,n0,n0
	cmpule	$5,n1,qb
	subq	n1,$5,tmp
	cmovne	qb,tmp,n1
	bis	n0,qb,n0
	cmplt	n0,0,tmp
	addq	n1,n1,n1
	bis	n1,tmp,n1
	addq	n0,n0,n0
	cmpule	$5,n1,qb
	subq	n1,$5,tmp
	cmovne	qb,tmp,n1
	bis	n0,qb,n0
	cmplt	n0,0,tmp
	addq	n1,n1,n1
	bis	n1,tmp,n1
	addq	n0,n0,n0
	cmpule	$5,n1,qb
	subq	n1,$5,tmp
	cmovne	qb,tmp,n1
	bis	n0,qb,n0
	cmplt	n0,0,tmp
	addq	n1,n1,n1
	bis	n1,tmp,n1
	addq	n0,n0,n0
	cmpule	$5,n1,qb
	subq	n1,$5,tmp
	cmovne	qb,tmp,n1
	bis	n0,qb,n0
	subq	cnt,1,cnt
	bgt	cnt,$loop2

	addq	n1,n1,n1
	addq	$4,n1,n1
	bne	$6,$Odd
	stq	n1,0(rem_ptr)
	bis	$31,n0,$0
	ret	$31,($26),1

$Odd:
	/* q' in n0. r' in n1 */
	addq	n1,n0,n1

	cmpult	n1,n0,tmp	# tmp := carry from addq
	subq	n1,d,AT
	addq	n0,tmp,n0
	cmovne	tmp,AT,n1

	cmpult	n1,d,tmp
	addq	n0,1,AT
	cmoveq	tmp,AT,n0
	subq	n1,d,AT
	cmoveq	tmp,AT,n1

	stq	n1,0(rem_ptr)
	bis	$31,n0,$0
	ret	$31,($26),1

	.end	__udiv_qrnnd
