/* Optimized strcasecmp implementation for PowerPC32.
   Copyright (C) 2011-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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
#include <locale-defines.h>

/* int [r3] strcasecmp (const char *s1 [r3], const char *s2 [r4] )

   or if defined USE_IN_EXTENDED_LOCALE_MODEL:

   int [r3] strcasecmp_l (const char *s1 [r3], const char *s2 [r4],
                          locale_t loc [r5]) */

#ifndef STRCMP
# define __STRCMP __strcasecmp
# define STRCMP   strcasecmp
#endif

ENTRY (__STRCMP)

#define rRTN	r3	/* Return value */
#define rSTR1	r5	/* 1st string */
#define rSTR2	r4	/* 2nd string */
#define rLOCARG	r5	/* 3rd argument: locale_t */
#define rCHAR1	r6	/* Byte read from 1st string */
#define rCHAR2	r7	/* Byte read from 2nd string */
#define rADDR1	r8	/* Address of tolower(rCHAR1) */
#define rADDR2	r12	/* Address of tolower(rCHAR2) */
#define rLWR1	r8	/* Byte tolower(rCHAR1) */
#define rLWR2	r12	/* Byte tolower(rCHAR2) */
#define rTMP	r0
#define rGOT	r9	/* Address of the Global Offset Table */
#define rLOC	r11	/* Default locale address */

	cmpw    cr7, r3, r4
#ifndef USE_IN_EXTENDED_LOCALE_MODEL
# ifdef SHARED
	mflr	rTMP
	bcl	20,31,.L1
.L1:	mflr	rGOT
	addis	rGOT, rGOT, _GLOBAL_OFFSET_TABLE_-.L1@ha
	addi 	rGOT, rGOT, _GLOBAL_OFFSET_TABLE_-.L1@l
	lwz	rLOC, __libc_tsd_LOCALE@got@tprel(rGOT)
	add 	rLOC, rLOC, __libc_tsd_LOCALE@tls
	lwz	rLOC, 0(rLOC)
	mtlr	rTMP
# else
	lis	rTMP,_GLOBAL_OFFSET_TABLE_@ha
	la	rLOC,_GLOBAL_OFFSET_TABLE_@l(rTMP)
	lwz	rLOC, __libc_tsd_LOCALE@got@tprel(rGOT)
	add	rLOC, rLOC, __libc_tsd_LOCALE@tls
	lwz	rLOC, 0(rLOC)
# endif /* SHARED */
#else
	mr	rLOC, rLOCARG
#endif
	mr	rSTR1, rRTN
	lwz	rLOC, LOCALE_CTYPE_TOLOWER(rLOC)
	li	rRTN, 0
	beqlr	cr7

	/* Unrolling loop for POWER: loads are done with 'lbz' plus
	offset and string descriptors are only updated in the end
	of loop unrolling. */

L(loop):
	lbz	rCHAR1, 0(rSTR1)	/* Load char from s1 */
	lbz	rCHAR2, 0(rSTR2)	/* Load char from s2 */
	sldi	rADDR1, rCHAR1, 2	/* Calculate address for tolower(*s1) */
	sldi	rADDR2, rCHAR2, 2	/* Calculate address for tolower(*s2) */
	lwzx	rLWR1, rLOC, rADDR1	/* Load tolower(*s1) */
	lwzx	rLWR2, rLOC, rADDR2	/* Load tolower(*s2) */
	cmpwi	cr7, rCHAR1, 0		/* *s1 == '\0' ? */
	subf.	r3, rLWR2, rLWR1
	bnelr
	beqlr	cr7
	lbz	rCHAR1, 1(rSTR1)
	lbz	rCHAR2, 1(rSTR2)
	sldi	rADDR1, rCHAR1, 2
	sldi	rADDR2, rCHAR2, 2
	lwzx	rLWR1, rLOC, rADDR1
	lwzx	rLWR2, rLOC, rADDR2
	cmpwi	cr7, rCHAR1, 0
	subf.	r3, rLWR2, rLWR1
	bnelr
	beqlr	cr7
	lbz	rCHAR1, 2(rSTR1)
	lbz	rCHAR2, 2(rSTR2)
	sldi	rADDR1, rCHAR1, 2
	sldi	rADDR2, rCHAR2, 2
	lwzx	rLWR1, rLOC, rADDR1
	lwzx	rLWR2, rLOC, rADDR2
	cmpwi	cr7, rCHAR1, 0
	subf.	r3, rLWR2, rLWR1
	bnelr
	beqlr	cr7
	lbz	rCHAR1, 3(rSTR1)
	lbz	rCHAR2, 3(rSTR2)
	/* Increment both string descriptors */
	addi	rSTR1, rSTR1, 4
	addi	rSTR2, rSTR2, 4
	sldi	rADDR1, rCHAR1, 2
	sldi	rADDR2, rCHAR2, 2
	lwzx	rLWR1, rLOC, rADDR1
	lwzx	rLWR2, rLOC, rADDR2
	cmpwi	cr7, rCHAR1, 0
	subf.	r3, rLWR2, rLWR1
	bnelr
	bne	cr7,L(loop)
	blr
END (__STRCMP)

weak_alias (__STRCMP, STRCMP)
libc_hidden_builtin_def (__STRCMP)
