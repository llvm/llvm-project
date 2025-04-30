/* Optimized strncpy implementation for POWER9 LE.
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

#ifdef USE_AS_STPNCPY
# ifndef STPNCPY
#   define FUNC_NAME __stpncpy
# else
#   define FUNC_NAME STPNCPY
# endif
#else
# ifndef STRNCPY
#  define FUNC_NAME strncpy
# else
#  define FUNC_NAME STRNCPY
# endif
#endif  /* !USE_AS_STPNCPY  */

#ifndef MEMSET
/* For builds without IFUNC support, local calls should be made to internal
   GLIBC symbol (created by libc_hidden_builtin_def).  */
# ifdef SHARED
#  define MEMSET_is_local
#  define MEMSET   __GI_memset
# else
#  define MEMSET   memset
# endif
#endif

#define FRAMESIZE (FRAME_MIN_SIZE+8)

/* Implements the function

   char * [r3] strncpy (char *dest [r3], const char *src [r4], size_t n [r5])

   or

   char * [r3] stpncpy (char *dest [r3], const char *src [r4], size_t n [r5])

   if USE_AS_STPNCPY is defined.

   The implementation can load bytes past a null terminator, but only
   up to the next 16-byte aligned address, so it never crosses a page.  */

.machine power9
#ifdef MEMSET_is_local
ENTRY_TOCLESS (FUNC_NAME, 4)
#else
ENTRY (FUNC_NAME, 4)
#endif
	CALL_MCOUNT 2

	/* NULL string optimizations  */
	cmpdi   r5, 0
	beqlr

	lbz	r0,0(r4)
	stb	r0,0(r3)
	addi	r11,r3,1
	addi	r5,r5,-1
	vspltisb v18,0		/* Zeroes in v18  */
	cmpdi	r0,0
	beq	L(zero_padding)

	/* Empty/1-byte string optimization  */
	cmpdi	r5,0
#ifdef USE_AS_STPNCPY
	bgt	L(cont)
	/* Compute pointer to last byte copied into dest.  */
	addi	r3,r3,1
	blr
L(cont):
#else
	beqlr
#endif

	addi	r4,r4,1
	neg	r7,r4
	rldicl	r9,r7,0,60	/* How many bytes to get source 16B aligned?  */

	/* Get source 16B aligned  */
	lvx	v0,0,r4
	lvsr	v1,0,r4
	vperm	v0,v18,v0,v1

	vcmpequb v6,v0,v18	/* 0xff if byte is NULL, 0x00 otherwise  */
	vctzlsbb r7,v6		/* Number of trailing zeroes  */
	addi	r8,r7,1		/* Add null terminator  */

	/* r8 = bytes including null
	   r9 = bytes to get source 16B aligned
	   if r8 > r9
	      no null, copy r9 bytes
	   else
	      there is a null, copy r8 bytes and return.  */
	cmpld	r8,r9
	bgt	L(no_null)

	cmpld	cr6,r8,r5	/* r8 <= n?  */
	ble	cr6,L(null)

	sldi	r10,r5,56	/* stxvl wants size in top 8 bits  */
	stxvl	32+v0,r11,r10	/* Partial store  */

#ifdef USE_AS_STPNCPY
	/* Compute pointer to last byte copied into dest.  */
	add	r3,r11,r5
#endif
	blr

L(null):
	sldi	r10,r8,56	/* stxvl wants size in top 8 bits  */
	stxvl	32+v0,r11,r10	/* Partial store  */

#ifdef USE_AS_STPNCPY
	/* Compute pointer to last byte copied into dest.  */
	add	r3,r11,r7
#endif
	add	r11,r11,r8
	sub	r5,r5,r8
	b L(zero_padding)

L(no_null):
	cmpld	r9,r5		/* Check if length was reached.  */
	bge	L(n_tail1)

	sldi	r10,r9,56	/* stxvl wants size in top 8 bits  */
	stxvl	32+v0,r11,r10	/* Partial store  */

	add	r4,r4,r9
	add	r11,r11,r9
	sub	r5,r5,r9

L(loop):
	cmpldi	cr6,r5,64	/* Check if length was reached.  */
	ble	cr6,L(final_loop)

	lxv	32+v0,0(r4)
	vcmpequb. v6,v0,v18	/* Any zero bytes?  */
	bne	cr6,L(prep_tail1)

	lxv	32+v1,16(r4)
	vcmpequb. v6,v1,v18	/* Any zero bytes?  */
	bne	cr6,L(prep_tail2)

	lxv	32+v2,32(r4)
	vcmpequb. v6,v2,v18	/* Any zero bytes?  */
	bne	cr6,L(prep_tail3)

	lxv	32+v3,48(r4)
	vcmpequb. v6,v3,v18	/* Any zero bytes?  */
	bne	cr6,L(prep_tail4)

	stxv	32+v0,0(r11)
	stxv	32+v1,16(r11)
	stxv	32+v2,32(r11)
	stxv	32+v3,48(r11)

	addi	r4,r4,64
	addi	r11,r11,64
	addi	r5,r5,-64

	b	L(loop)

L(final_loop):
	cmpldi	cr5,r5,16
	lxv	32+v0,0(r4)
	vcmpequb. v6,v0,v18	/* Any zero bytes?  */
	ble	cr5,L(prep_n_tail1)
	bne	cr6,L(count_tail1)
	addi	r5,r5,-16

	cmpldi	cr5,r5,16
	lxv	32+v1,16(r4)
	vcmpequb. v6,v1,v18	/* Any zero bytes?  */
	ble	cr5,L(prep_n_tail2)
	bne	cr6,L(count_tail2)
	addi	r5,r5,-16

	cmpldi	cr5,r5,16
	lxv	32+v2,32(r4)
	vcmpequb. v6,v2,v18	/* Any zero bytes?  */
	ble	cr5,L(prep_n_tail3)
	bne	cr6,L(count_tail3)
	addi	r5,r5,-16

	lxv	32+v3,48(r4)
	vcmpequb. v6,v3,v18	/* Any zero bytes?  */
	beq	cr6,L(n_tail4)

	vctzlsbb r8,v6		/* Number of trailing zeroes  */
	cmpld	r8,r5		/* r8 < n?  */
	blt	L(tail4)

L(n_tail4):
	stxv	32+v0,0(r11)
	stxv	32+v1,16(r11)
	stxv	32+v2,32(r11)
	sldi	r10,r5,56	/* stxvl wants size in top 8 bits  */
	addi	r11,r11,48	/* Offset */
	stxvl	32+v3,r11,r10	/* Partial store  */
#ifdef USE_AS_STPNCPY
	/* Compute pointer to last byte copied into dest.  */
	add	r3,r11,r5
#endif
	blr

L(prep_n_tail1):
	beq	cr6,L(n_tail1)	/* Any zero bytes?  */
	vctzlsbb r8,v6		/* Number of trailing zeroes  */
	cmpld	r8,r5		/* r8 < n?  */
	blt	L(tail1)

L(n_tail1):
	sldi	r10,r5,56	/* stxvl wants size in top 8 bits  */
	stxvl	32+v0,r11,r10	/* Partial store  */
#ifdef USE_AS_STPNCPY
	/* Compute pointer to last byte copied into dest.  */
	add	r3,r11,r5
#endif
	blr

L(prep_n_tail2):
	beq	cr6,L(n_tail2)	/* Any zero bytes?  */
	vctzlsbb r8,v6		/* Number of trailing zeroes  */
	cmpld	r8,r5		/* r8 < n?  */
	blt	L(tail2)

L(n_tail2):
	stxv	32+v0,0(r11)
	sldi	r10,r5,56	/* stxvl wants size in top 8 bits  */
	addi	r11,r11,16	/* offset */
	stxvl	32+v1,r11,r10	/* Partial store  */
#ifdef USE_AS_STPNCPY
	/* Compute pointer to last byte copied into dest.  */
	add	r3,r11,r5
#endif
	blr

L(prep_n_tail3):
	beq	cr6,L(n_tail3)	/* Any zero bytes?  */
	vctzlsbb r8,v6		/* Number of trailing zeroes  */
	cmpld	r8,r5		/* r8 < n?  */
	blt	L(tail3)

L(n_tail3):
	stxv	32+v0,0(r11)
	stxv	32+v1,16(r11)
	sldi	r10,r5,56	/* stxvl wants size in top 8 bits  */
	addi	r11,r11,32	/* Offset */
	stxvl	32+v2,r11,r10	/* Partial store  */
#ifdef USE_AS_STPNCPY
	/* Compute pointer to last byte copied into dest.  */
	add	r3,r11,r5
#endif
	blr

L(prep_tail1):
L(count_tail1):
	vctzlsbb r8,v6		/* Number of trailing zeroes  */
L(tail1):
	addi	r9,r8,1		/* Add null terminator  */
	sldi	r10,r9,56	/* stxvl wants size in top 8 bits  */
	stxvl	32+v0,r11,r10	/* Partial store  */
#ifdef USE_AS_STPNCPY
	/* Compute pointer to last byte copied into dest.  */
	add	r3,r11,r8
#endif
	add	r11,r11,r9
	sub	r5,r5,r9
	b L(zero_padding)

L(prep_tail2):
	addi	r5,r5,-16
L(count_tail2):
	vctzlsbb r8,v6		/* Number of trailing zeroes  */
L(tail2):
	addi	r9,r8,1		/* Add null terminator  */
	stxv	32+v0,0(r11)
	sldi	r10,r9,56	/* stxvl wants size in top 8 bits  */
	addi	r11,r11,16	/* offset */
	stxvl	32+v1,r11,r10	/* Partial store  */
#ifdef USE_AS_STPNCPY
	/* Compute pointer to last byte copied into dest.  */
	add	r3,r11,r8
#endif
	add	r11,r11,r9
	sub	r5,r5,r9
	b L(zero_padding)

L(prep_tail3):
	addi	r5,r5,-32
L(count_tail3):
	vctzlsbb r8,v6		/* Number of trailing zeroes  */
L(tail3):
	addi	r9,r8,1		/* Add null terminator  */
	stxv	32+v0,0(r11)
	stxv	32+v1,16(r11)
	sldi	r10,r9,56	/* stxvl wants size in top 8 bits  */
	addi	r11,r11,32	/* offset */
	stxvl	32+v2,r11,r10	/* Partial store  */
#ifdef USE_AS_STPNCPY
	/* Compute pointer to last byte copied into dest.  */
	add	r3,r11,r8
#endif
	add	r11,r11,r9
	sub	r5,r5,r9
	b L(zero_padding)

L(prep_tail4):
	addi	r5,r5,-48
	vctzlsbb r8,v6		/* Number of trailing zeroes  */
L(tail4):
	addi	r9,r8,1		/* Add null terminator  */
	stxv	32+v0,0(r11)
	stxv	32+v1,16(r11)
	stxv	32+v2,32(r11)
	sldi	r10,r9,56	/* stxvl wants size in top 8 bits  */
	addi	r11,r11,48	/* offset */
	stxvl	32+v3,r11,r10	/* Partial store  */
#ifdef USE_AS_STPNCPY
	/* Compute pointer to last byte copied into dest.  */
	add	r3,r11,r8
#endif
	add	r11,r11,r9
	sub	r5,r5,r9

/* This code pads the remainder of dest with NULL bytes.  For large numbers
   memset gives a better performance, 255 was chosen through experimentation.
   */
L(zero_padding):
	cmpldi	r5,255
	bge	L(zero_padding_memset)

L(zero_padding_loop):
	cmpldi	cr6,r5,16	/* Check if length was reached.  */
	ble	cr6,L(zero_padding_end)

	stxv	v18,0(r11)
	addi	r11,r11,16
	addi	r5,r5,-16

	b	L(zero_padding_loop)

L(zero_padding_end):
	sldi	r10,r5,56	/* stxvl wants size in top 8 bits  */
	stxvl	v18,r11,r10	/* Partial store  */
	blr

	.align	4
L(zero_padding_memset):
	std	r30,-8(r1)   /* Save r30 on the stack.  */
	cfi_offset(r30, -8)
	mr	r30,r3       /* Save the return value of strncpy.  */
	/* Prepare the call to memset.  */
	mr	r3,r11       /* Pointer to the area to be zero-filled.  */
	li	r4,0         /* Byte to be written (zero).  */

	/* We delayed the creation of the stack frame, as well as the saving of
	   the link register, because only at this point, we are sure that
	   doing so is actually needed.  */

	/* Save the link register.  */
	mflr	r0
	std	r0,16(r1)

	/* Create the stack frame.  */
	stdu	r1,-FRAMESIZE(r1)
	cfi_adjust_cfa_offset(FRAMESIZE)
	cfi_offset(lr, 16)

	bl	MEMSET
#ifndef MEMSET_is_local
	nop
#endif

	ld	r0,FRAMESIZE+16(r1)

	mr	r3,r30       /* Restore the return value of strncpy, i.e.:
				dest.  For stpncpy, the return value is the
				same as return value of memset.  */
	ld	r30,FRAMESIZE-8(r1) /* Restore r30.  */
	/* Restore the stack frame.  */
	addi	r1,r1,FRAMESIZE
	cfi_adjust_cfa_offset(-FRAMESIZE)
	/* Restore the link register.  */
	mtlr	r0
	cfi_restore(lr)
	blr

END (FUNC_NAME)
#ifndef USE_AS_STPNCPY
libc_hidden_builtin_def (strncpy)
#endif
