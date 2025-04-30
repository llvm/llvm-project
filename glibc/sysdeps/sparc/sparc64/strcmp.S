/* Compare two strings for differences.
   For SPARC v9.
   Copyright (C) 2011-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David S. Miller <davem@davemloft.net>

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
	.register	%g2, #scratch
	.register	%g3, #scratch
	.register	%g6, #scratch
#endif

#define rSTR1		%o0
#define rSTR2		%o1
#define r0101		%o2	/* 0x0101010101010101 */
#define r8080		%o3	/* 0x8080808080808080 */
#define rSTRXOR		%o4
#define rWORD1		%o5
#define rTMP1		%g1
#define rTMP2		%g2
#define rWORD2		%g3
#define rSLL		%g4
#define rSRL		%g5
#define rBARREL		%g6

	/* There are two cases, either the two pointers are aligned
	 * identically or they are not.  If they have the same
	 * alignment we can use the normal full speed loop.  Otherwise
	 * we have to use the barrel-shifter version.
	 */

	.text
	.align	32
ENTRY(strcmp)
	or	rSTR2, rSTR1, rTMP1
	sethi	%hi(0x80808080), r8080

	andcc	rTMP1, 0x7, %g0
	bne,pn	%icc, .Lmaybe_barrel_shift
	 or	r8080, %lo(0x80808080), r8080
	ldx	[rSTR1], rWORD1

	sub	rSTR2, rSTR1, rSTR2
	sllx	r8080, 32, rTMP1

	ldx	[rSTR1 + rSTR2], rWORD2
	or	r8080, rTMP1, r8080

	ba,pt	%xcc, .Laligned_loop_entry
	 srlx	r8080, 7, r0101

	.align	32
.Laligned_loop_entry:
.Laligned_loop:
	add	rSTR1, 8, rSTR1

	sub	rWORD1, r0101, rTMP2
	xorcc	rWORD1, rWORD2, rSTRXOR
	bne,pn	%xcc, .Lcommon_endstring

	 andn	r8080, rWORD1, rTMP1

	ldxa	[rSTR1] ASI_PNF, rWORD1
	andcc	rTMP1, rTMP2, %g0
	be,a,pt	%xcc, .Laligned_loop

	 ldxa	[rSTR1 + rSTR2] ASI_PNF, rWORD2

.Lcommon_equal:
	retl
	 mov	0, %o0

	/* All loops terminate here once they find an unequal word.
	 * If a zero byte appears in the word before the first unequal
	 * byte, we must report zero.  Otherwise we report '1' or '-1'
	 * depending upon whether the first mis-matching byte is larger
	 * in the first string or the second, respectively.
	 *
	 * First we compute a 64-bit mask value that has "0x01" in
	 * each byte where a zero exists in rWORD1.  rSTRXOR holds the
	 * value (rWORD1 ^ rWORD2).  Therefore, if considered as an
	 * unsigned quantity, our "0x01" mask value is "greater than"
	 * rSTRXOR then a zero terminating byte comes first and
	 * therefore we report '0'.
	 *
	 * The formula for this mask is:
	 *
	 *    mask_tmp1 = ~rWORD1 & 0x8080808080808080;
	 *    mask_tmp2 = ((rWORD1 & 0x7f7f7f7f7f7f7f7f) +
	 *                 0x7f7f7f7f7f7f7f7f);
	 *
	 *    mask = ((mask_tmp1 & ~mask_tmp2) >> 7);
	 */
.Lcommon_endstring:
	andn	rWORD1, r8080, rTMP2
	or	r8080, 1, %o1

	mov	1, %o0
	sub	rTMP2, %o1, rTMP2

	cmp	rWORD1, rWORD2
	andn	rTMP1, rTMP2, rTMP1

	movleu	%xcc, -1, %o0
	srlx	rTMP1, 7, rTMP1

	/* In order not to be influenced by bytes after the zero byte, we
	 * have to retain only the highest bit in the mask for the comparison
	 * with rSTRXOR to work properly.
	 */
	mov	0, rTMP2
	andcc	rTMP1, 0x0100, %g0

	movne	%xcc, 8, rTMP2
	sllx	rTMP1, 63 - 16, %o1

	movrlz	%o1, 16, rTMP2
	sllx	rTMP1, 63 - 24, %o1

	movrlz	%o1, 24, rTMP2
	sllx	rTMP1, 63 - 32, %o1

	movrlz	%o1, 32, rTMP2
	sllx	rTMP1, 63 - 40, %o1

	movrlz	%o1, 40, rTMP2
	sllx	rTMP1, 63 - 48, %o1

	movrlz	%o1, 48, rTMP2
	sllx	rTMP1, 63 - 56, %o1

	movrlz	%o1, 56, rTMP2

	srlx	rTMP1, rTMP2, rTMP1

	sllx	rTMP1, rTMP2, rTMP1

	cmp	rTMP1, rSTRXOR
	retl
	 movgu	%xcc, 0, %o0

.Lmaybe_barrel_shift:
	sub	rSTR2, rSTR1, rSTR2
	sllx	r8080, 32, rTMP1

	or	r8080, rTMP1, r8080
	and	rSTR1, 0x7, rTMP2

	srlx	r8080, 7, r0101
	andn	rSTR1, 0x7, rSTR1

	ldxa	[rSTR1] ASI_PNF, rWORD1
	andcc	rSTR2, 0x7, rSLL
	sll	rTMP2, 3, rSTRXOR

	bne,pn	%icc, .Lneed_barrel_shift
	 mov	-1, rTMP1
	ldxa	[rSTR1 + rSTR2] ASI_PNF, rBARREL

	srlx	rTMP1, rSTRXOR, rTMP2

	orn	rWORD1, rTMP2, rWORD1
	ba,pt	%xcc, .Laligned_loop_entry
	 orn	rBARREL, rTMP2, rWORD2

.Lneed_barrel_shift:
	sllx	rSLL, 3, rSLL
	andn	rSTR2, 0x7, rSTR2

	ldxa	[rSTR1 + rSTR2] ASI_PNF, rBARREL
	mov	64, rTMP2
	sub	rTMP2, rSLL, rSRL

	srlx	rTMP1, rSTRXOR, rTMP1
	add	rSTR2, 8, rSTR2

	orn	rWORD1, rTMP1, rWORD1
	sllx	rBARREL, rSLL, rWORD2
	ldxa	[rSTR1 + rSTR2] ASI_PNF, rBARREL

	add	rSTR1, 8, rSTR1
	sub	rWORD1, r0101, rTMP2

	srlx	rBARREL, rSRL, rSTRXOR

	or	rWORD2, rSTRXOR, rWORD2

	orn	rWORD2, rTMP1, rWORD2
	ba,pt	%xcc, .Lbarrel_shift_loop_entry
	 andn	r8080, rWORD1, rTMP1

.Lbarrel_shift_loop:
	sllx	rBARREL, rSLL, rWORD2
	ldxa	[rSTR1 + rSTR2] ASI_PNF, rBARREL

	add	rSTR1, 8, rSTR1
	sub	rWORD1, r0101, rTMP2

	srlx	rBARREL, rSRL, rSTRXOR
	andn	r8080, rWORD1, rTMP1

	or	rWORD2, rSTRXOR, rWORD2

.Lbarrel_shift_loop_entry:
	xorcc	rWORD1, rWORD2, rSTRXOR
	bne,pn	%xcc, .Lcommon_endstring

	 andcc	rTMP1, rTMP2, %g0
	be,a,pt	%xcc, .Lbarrel_shift_loop
	 ldxa	[rSTR1] ASI_PNF, rWORD1

	retl
	 mov	0, %o0
END(strcmp)
libc_hidden_builtin_def (strcmp)
