/* Copyright (C) 2014-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#ifdef ANDROID_CHANGES
# include "machine/asm.h"
# include "machine/regdef.h"
#elif _LIBC
# include <sysdep.h>
# include <regdef.h>
# include <sys/asm.h>
#elif defined _COMPILING_NEWLIB
# include "machine/asm.h"
# include "machine/regdef.h"
#else
# include <regdef.h>
# include <sys/asm.h>
#endif

/* Technically strcmp should not read past the end of the strings being
   compared.  We will read a full word that may contain excess bits beyond
   the NULL string terminator but unless ENABLE_READAHEAD is set, we will not
   read the next word after the end of string.  Setting ENABLE_READAHEAD will
   improve performance but is technically illegal based on the definition of
   strcmp.  */
#ifdef ENABLE_READAHEAD
# define DELAY_READ
#else
# define DELAY_READ nop
#endif

/* Testing on a little endian machine showed using CLZ was a
   performance loss, so we are not turning it on by default.  */
#if defined(ENABLE_CLZ) && (__mips_isa_rev > 1)
# define USE_CLZ
#endif

/* Some asm.h files do not have the L macro definition.  */
#ifndef L
# if _MIPS_SIM == _ABIO32
#  define L(label) $L ## label
# else
#  define L(label) .L ## label
# endif
#endif

/* Some asm.h files do not have the PTR_ADDIU macro definition.  */
#ifndef PTR_ADDIU
# ifdef USE_DOUBLE
#  define PTR_ADDIU       daddiu
# else
#  define PTR_ADDIU       addiu
# endif
#endif

/* Allow the routine to be named something else if desired.  */
#ifndef STRCMP_NAME
# define STRCMP_NAME strcmp
#endif

#ifdef ANDROID_CHANGES
LEAF(STRCMP_NAME, 0)
#else
LEAF(STRCMP_NAME)
#endif
	.set	nomips16
	.set	noreorder

	or	t0, a0, a1
	andi	t0,0x3
	bne	t0, zero, L(byteloop)

/* Both strings are 4 byte aligned at this point.  */

	lui	t8, 0x0101
	ori	t8, t8, 0x0101
	lui	t9, 0x7f7f
	ori	t9, 0x7f7f

#define STRCMP32(OFFSET) \
	lw	v0, OFFSET(a0); \
	lw	v1, OFFSET(a1); \
	subu	t0, v0, t8; \
	bne	v0, v1, L(worddiff); \
	nor	t1, v0, t9; \
	and	t0, t0, t1; \
	bne	t0, zero, L(returnzero)

L(wordloop):
	STRCMP32(0)
	DELAY_READ
	STRCMP32(4)
	DELAY_READ
	STRCMP32(8)
	DELAY_READ
	STRCMP32(12)
	DELAY_READ
	STRCMP32(16)
	DELAY_READ
	STRCMP32(20)
	DELAY_READ
	STRCMP32(24)
	DELAY_READ
	STRCMP32(28)
	PTR_ADDIU a0, a0, 32
	b	L(wordloop)
	PTR_ADDIU a1, a1, 32

L(returnzero):
	j	ra
	move	v0, zero

L(worddiff):
#ifdef USE_CLZ
	subu	t0, v0, t8
	nor	t1, v0, t9
	and	t1, t0, t1
	xor	t0, v0, v1
	or	t0, t0, t1
# if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
	wsbh	t0, t0
	rotr	t0, t0, 16
# endif
	clz	t1, t0
	and	t1, 0xf8
# if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
	neg	t1
	addu	t1, 24
# endif
	rotrv	v0, v0, t1
	rotrv	v1, v1, t1
	and	v0, v0, 0xff
	and	v1, v1, 0xff
	j	ra
	subu	v0, v0, v1
#else /* USE_CLZ */
# if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
	andi	t0, v0, 0xff
	beq	t0, zero, L(wexit01)
	andi	t1, v1, 0xff
	bne	t0, t1, L(wexit01)

	srl	t8, v0, 8
	srl	t9, v1, 8
	andi	t8, t8, 0xff
	beq	t8, zero, L(wexit89)
	andi	t9, t9, 0xff
	bne	t8, t9, L(wexit89)

	srl	t0, v0, 16
	srl	t1, v1, 16
	andi	t0, t0, 0xff
	beq	t0, zero, L(wexit01)
	andi	t1, t1, 0xff
	bne	t0, t1, L(wexit01)

	srl	t8, v0, 24
	srl	t9, v1, 24
# else /* __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ */
	srl	t0, v0, 24
	beq	t0, zero, L(wexit01)
	srl	t1, v1, 24
	bne	t0, t1, L(wexit01)

	srl	t8, v0, 16
	srl	t9, v1, 16
	andi	t8, t8, 0xff
	beq	t8, zero, L(wexit89)
	andi	t9, t9, 0xff
	bne	t8, t9, L(wexit89)

	srl	t0, v0, 8
	srl	t1, v1, 8
	andi	t0, t0, 0xff
	beq	t0, zero, L(wexit01)
	andi	t1, t1, 0xff
	bne	t0, t1, L(wexit01)

	andi	t8, v0, 0xff
	andi	t9, v1, 0xff
# endif /* __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ */

L(wexit89):
	j	ra
	subu	v0, t8, t9
L(wexit01):
	j	ra
	subu	v0, t0, t1
#endif /* USE_CLZ */

/* It might seem better to do the 'beq' instruction between the two 'lbu'
   instructions so that the nop is not needed but testing showed that this
   code is actually faster (based on glibc strcmp test).  */
#define BYTECMP01(OFFSET) \
	lbu	v0, OFFSET(a0); \
	lbu	v1, OFFSET(a1); \
	beq	v0, zero, L(bexit01); \
	nop; \
	bne	v0, v1, L(bexit01)

#define BYTECMP89(OFFSET) \
	lbu	t8, OFFSET(a0); \
	lbu	t9, OFFSET(a1); \
	beq	t8, zero, L(bexit89); \
	nop;	\
	bne	t8, t9, L(bexit89)

L(byteloop):
	BYTECMP01(0)
	BYTECMP89(1)
	BYTECMP01(2)
	BYTECMP89(3)
	BYTECMP01(4)
	BYTECMP89(5)
	BYTECMP01(6)
	BYTECMP89(7)
	PTR_ADDIU a0, a0, 8
	b	L(byteloop)
	PTR_ADDIU a1, a1, 8

L(bexit01):
	j	ra
	subu	v0, v0, v1
L(bexit89):
	j	ra
	subu	v0, t8, t9

	.set	at
	.set	reorder

END(STRCMP_NAME)
#ifndef ANDROID_CHANGES
# ifdef _LIBC
libc_hidden_builtin_def (STRCMP_NAME)
# endif
#endif
