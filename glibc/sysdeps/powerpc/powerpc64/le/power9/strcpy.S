/* Optimized strcpy implementation for PowerPC64/POWER9.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#ifdef USE_AS_STPCPY
# ifndef STPCPY
#   define FUNC_NAME __stpcpy
# else
#   define FUNC_NAME STPCPY
# endif
#else
# ifndef STRCPY
#  define FUNC_NAME strcpy
# else
#  define FUNC_NAME STRCPY
# endif
#endif  /* !USE_AS_STPCPY  */

/* Implements the function

   char * [r3] strcpy (char *dest [r3], const char *src [r4])

   or

   char * [r3] stpcpy (char *dest [r3], const char *src [r4])

   if USE_AS_STPCPY is defined.

   The implementation can load bytes past a null terminator, but only
   up to the next 16B boundary, so it never crosses a page.  */

/* Load quadword at addr+offset to vreg, check for null bytes,
   and branch to label if any are found.  */
#define CHECK16(vreg,offset,addr,label) \
	lxv	vreg+32,offset(addr);	\
	vcmpequb. v6,vreg,v18;	\
	bne	cr6,L(label);

.machine power9
ENTRY_TOCLESS (FUNC_NAME, 4)
	CALL_MCOUNT 2

	vspltisb v18,0		/* Zeroes in v18  */
	vspltisb v19,-1 	/* 0xFF bytes in v19  */

	/* Next 16B-aligned address. Prepare address for L(loop).  */
	addi	r5,r4,16
	clrrdi	r5,r5,4
	subf	r8,r4,r5
	add	r11,r3,r8

	/* Align data and fill bytes not loaded with non matching char.  */
	lvx	v0,0,r4
	lvsr	v1,0,r4
	vperm	v0,v19,v0,v1

	vcmpequb. v6,v0,v18	/* 0xff if byte is NULL, 0x00 otherwise  */
	beq	cr6,L(no_null)

	/* There's a null byte.  */
	vctzlsbb r8,v6		/* Number of trailing zeroes  */
	addi	r9,r8,1 	/* Add null byte.  */
	sldi	r10,r9,56	/* stxvl wants size in top 8 bits.  */
	stxvl	32+v0,r3,r10	/* Partial store  */

#ifdef USE_AS_STPCPY
	/* stpcpy returns the dest address plus the size not counting the
	   final '\0'.  */
	add	r3,r3,r8
#endif
	blr

L(no_null):
	sldi	r10,r8,56	/* stxvl wants size in top 8 bits  */
	stxvl	32+v0,r3,r10	/* Partial store  */

	.p2align 4
L(loop):
	CHECK16(v0,0,r5,tail1)
	CHECK16(v1,16,r5,tail2)
	CHECK16(v2,32,r5,tail3)
	CHECK16(v3,48,r5,tail4)
	CHECK16(v4,64,r5,tail5)
	CHECK16(v5,80,r5,tail6)

	stxv	32+v0,0(r11)
	stxv	32+v1,16(r11)
	stxv	32+v2,32(r11)
	stxv	32+v3,48(r11)
	stxv	32+v4,64(r11)
	stxv	32+v5,80(r11)

	addi	r5,r5,96
	addi	r11,r11,96

	b	L(loop)

	.p2align 4
L(tail1):
	vctzlsbb r8,v6		/* Number of trailing zeroes  */
	addi	r9,r8,1		/* Add null terminator  */
	sldi	r9,r9,56	/* stxvl wants size in top 8 bits  */
	stxvl	32+v0,r11,r9	/* Partial store  */
#ifdef USE_AS_STPCPY
	/* stpcpy returns the dest address plus the size not counting the
	   final '\0'.  */
	add	r3,r11,r8
#endif
	blr

	.p2align 4
L(tail2):
	stxv	32+v0,0(r11)
	vctzlsbb r8,v6
	addi	r9,r8,1
	sldi	r9,r9,56
	addi	r11,r11,16
	stxvl	32+v1,r11,r9
#ifdef USE_AS_STPCPY
	add	r3,r11,r8
#endif
	blr

	.p2align 4
L(tail3):
	stxv	32+v0,0(r11)
	stxv	32+v1,16(r11)
	vctzlsbb r8,v6
	addi	r9,r8,1
	sldi	r9,r9,56
	addi	r11,r11,32
	stxvl	32+v2,r11,r9
#ifdef USE_AS_STPCPY
	add	r3,r11,r8
#endif
	blr

	.p2align 4
L(tail4):
	stxv	32+v0,0(r11)
	stxv	32+v1,16(r11)
	stxv	32+v2,32(r11)
	vctzlsbb r8,v6
	addi	r9,r8,1
	sldi	r9,r9,56
	addi	r11,r11,48
	stxvl	32+v3,r11,r9
#ifdef USE_AS_STPCPY
	add	r3,r11,r8
#endif
	blr

	.p2align 4
L(tail5):
	stxv	32+v0,0(r11)
	stxv	32+v1,16(r11)
	stxv	32+v2,32(r11)
	stxv	32+v3,48(r11)
	vctzlsbb r8,v6
	addi	r9,r8,1
	sldi	r9,r9,56
	addi	r11,r11,64
	stxvl	32+v4,r11,r9
#ifdef USE_AS_STPCPY
	add	r3,r11,r8
#endif
	blr

	.p2align 4
L(tail6):
	stxv	32+v0,0(r11)
	stxv	32+v1,16(r11)
	stxv	32+v2,32(r11)
	stxv	32+v3,48(r11)
	stxv	32+v4,64(r11)
	vctzlsbb r8,v6
	addi	r9,r8,1
	sldi	r9,r9,56
	addi	r11,r11,80
	stxvl	32+v5,r11,r9
#ifdef USE_AS_STPCPY
	add	r3,r11,r8
#endif
	blr

END (FUNC_NAME)
#ifndef USE_AS_STPCPY
libc_hidden_builtin_def (strcpy)
#endif
