/* Optimized strncpy/stpncpy implementation for PowerPC64/POWER8.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#define FRAMESIZE (FRAME_MIN_SIZE+48)

/* Implements the function

   char * [r3] strncpy (char *dest [r3], const char *src [r4], size_t n [r5])

   or

   char * [r3] stpncpy (char *dest [r3], const char *src [r4], size_t n [r5])

   if USE_AS_STPCPY is defined.

   The implementation uses unaligned doubleword access to avoid specialized
   code paths depending of data alignment.  Although recent powerpc64 uses
   64K as default, the page cross handling assumes minimum page size of
   4k.  */

	.machine  power8
#ifdef MEMSET_is_local
ENTRY_TOCLESS (FUNC_NAME, 4)
#else
ENTRY (FUNC_NAME, 4)
#endif
	CALL_MCOUNT 3

        /* Check if the [src]+15 will cross a 4K page by checking if the bit
           indicating the page size changes.  Basically:

           uint64_t srcin = (uint64_t)src;
           uint64_t ob = srcin & 4096UL;
           uint64_t nb = (srcin+15UL) & 4096UL;
           if (ob ^ nb)
             goto pagecross;  */

	addi	r10,r4,16
	rlwinm	r9,r4,0,19,19

	/* Save some non-volatile registers on the stack.  */
	std	r26,-48(r1)
	std	r27,-40(r1)

	rlwinm	r8,r10,0,19,19

	std	r28,-32(r1)
	std	r29,-24(r1)

	cmpld	cr7,r9,r8

	std	r30,-16(r1)
	std	r31,-8(r1)

	/* Update CFI.  */
	cfi_offset(r26, -48)
	cfi_offset(r27, -40)
	cfi_offset(r28, -32)
	cfi_offset(r29, -24)
	cfi_offset(r30, -16)
	cfi_offset(r31, -8)

	beq	cr7,L(unaligned_lt_16)
	rldicl	r9,r4,0,61
	subfic	r8,r9,8
	cmpld	cr7,r5,r8
	bgt 	cr7,L(pagecross)

	/* At this points there is 1 to 15 bytes to check and write.  Since it could
	   be either from first unaligned 16 bytes access or from bulk copy, the code
	   uses an unrolled byte read/write instead of trying to analyze the cmpb
	   results.  */
L(short_path):
	mr	r9,r3
L(short_path_1):
	/* Return if there are no more bytes to be written.  */
	cmpdi	cr7,r5,0
	beq	cr7,L(short_path_loop_end_1)
L(short_path_2):
	/* Copy one char from src (r4) and write it to dest (r9).  If it is the
	   end-of-string, start the null padding.  Continue, otherwise.  */
	lbz	r10,0(r4)
	cmpdi	cr7,r10,0
	stb	r10,0(r9)
	beq	cr7,L(zero_pad_start_1)
	/* If there are no more bytes to be written, return.  */
	cmpdi	cr0,r5,1
	addi	r8,r9,1
	addi	r6,r5,-1
	beq	cr0,L(short_path_loop_end_0)
	/* Copy another char from src (r4) to dest (r9).  Check again if it is
	   the end-of-string.  If so, start the null padding.  */
	lbz	r10,1(r4)
	cmpdi	cr7,r10,0
	stb	r10,1(r9)
	beq	cr7,L(zero_pad_start_prepare_1)
	/* Eagerly decrement r5 by 3, which is the number of bytes already
	   written, plus one write that will be performed later on.  */
	addi	r10,r5,-3
	b	L(short_path_loop_1)

	.align	4
L(short_path_loop):
	/* At this point, the induction variable, r5, as well as the pointers
	   to dest and src (r9 and r4, respectivelly) have been updated.

	   Note: The registers r7 and r10 are induction variables derived from
	   r5.  They are used to determine if the total number of writes has
	   been reached at every other write.

	   Copy one char from src (r4) and write it to dest (r9).  If it is the
	   end-of-string, start the null padding.  Continue, otherwise.  */
	lbz	r8,0(r4)
	addi	r7,r10,-2
	cmpdi	cr5,r8,0
	stb	r8,0(r9)
	beq	cr5,L(zero_pad_start_1)
	beq	cr7,L(short_path_loop_end_0)
	/* Copy another char from src (r4) to dest (r9).  Check again if it is
	   the end-of-string.  If so, start the null padding.  */
	lbz	r8,1(r4)
	cmpdi	cr7,r8,0
	stb	r8,1(r9)
	beq	cr7,L(zero_pad_start)
	mr	r10,r7
L(short_path_loop_1):
	/* This block is reached after two chars have been already written to
	   dest.  Nevertheless, r5 (the induction variable), r9 (the pointer to
	   dest), and r4 (the pointer to src) have not yet been updated.

	   At this point:
	     r5 holds the count of bytes yet to be written plus 2.
	     r9 points to the last two chars that were already written to dest.
	     r4 points to the last two chars that were already copied from src.

	   The algorithm continues by decrementing r5, the induction variable,
	   so that it reflects the last two writes.  The pointers to dest (r9)
	   and to src (r4) are increment by two, for the same reason.

	   Note: Register r10 is another induction variable, derived from r5,
	   which determines if the total number of writes has been reached.  */
	addic.	r5,r5,-2
	addi	r9,r9,2
	cmpdi	cr7,r10,0 /* Eagerly check if the next write is the last.  */
	addi	r4,r4,2
	addi	r6,r9,1
	bne	cr0,L(short_path_loop) /* Check if the total number of writes
					  has been reached at every other
					  write.  */
#ifdef USE_AS_STPNCPY
	mr	r3,r9
	b	L(short_path_loop_end)
#endif

L(short_path_loop_end_0):
#ifdef USE_AS_STPNCPY
	addi	r3,r9,1
	b	L(short_path_loop_end)
#endif
L(short_path_loop_end_1):
#ifdef USE_AS_STPNCPY
	mr	r3,r9
#endif
L(short_path_loop_end):
	/* Restore non-volatile registers.  */
	ld	r26,-48(r1)
	ld	r27,-40(r1)
	ld	r28,-32(r1)
	ld	r29,-24(r1)
	ld	r30,-16(r1)
	ld	r31,-8(r1)
	blr

	/* This code pads the remainder of dest with NULL bytes.  The algorithm
	   calculates the remaining size and calls memset.  */
	.align	4
L(zero_pad_start):
	mr	r5,r10
	mr	r9,r6
L(zero_pad_start_1):
	/* At this point:
	     - r5 holds the number of bytes that still have to be written to
	       dest.
	     - r9 points to the position, in dest, where the first null byte
	       will be written.
	   The above statements are true both when control reaches this label
	   from a branch or when falling through the previous lines.  */
#ifndef USE_AS_STPNCPY
	mr	r30,r3       /* Save the return value of strncpy.  */
#endif
	/* Prepare the call to memset.  */
	mr	r3,r9        /* Pointer to the area to be zero-filled.  */
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

#ifndef USE_AS_STPNCPY
	mr	r3,r30       /* Restore the return value of strncpy, i.e.:
				dest.  For stpncpy, the return value is the
				same as return value of memset.  */
#endif

	/* Restore non-volatile registers and return.  */
	ld	r26,FRAMESIZE-48(r1)
	ld	r27,FRAMESIZE-40(r1)
	ld	r28,FRAMESIZE-32(r1)
	ld	r29,FRAMESIZE-24(r1)
	ld	r30,FRAMESIZE-16(r1)
	ld	r31,FRAMESIZE-8(r1)
	/* Restore the stack frame.  */
	addi	r1,r1,FRAMESIZE
	cfi_adjust_cfa_offset(-FRAMESIZE)
	/* Restore the link register.  */
	mtlr	r0
	cfi_restore(lr)
	blr

	/* The common case where [src]+16 will not cross a 4K page boundary.
	   In this case the code fast check the first 16 bytes by using doubleword
	   read/compares and update destiny if neither total size or null byte
	   is found in destiny. */
	.align	4
L(unaligned_lt_16):
	cmpldi	cr7,r5,7
	ble	cr7,L(short_path)
	ld	r7,0(r4)
	li	r8,0
	cmpb	r8,r7,r8
	cmpdi	cr7,r8,0
	bne	cr7,L(short_path_prepare_2)
	addi	r6,r5,-8
	std	r7,0(r3)
	addi	r9,r3,8
	cmpldi	cr7,r6,7
	addi	r7,r4,8
	ble	cr7,L(short_path_prepare_1_1)
	ld	r4,8(r4)
	cmpb	r8,r4,r8
	cmpdi	cr7,r8,0
	bne	cr7,L(short_path_prepare_2_1)
	std	r4,8(r3)
	addi	r29,r3,16
	addi	r5,r5,-16
	/* Neither the null byte was found or total length was reached,
	   align to 16 bytes and issue a bulk copy/compare.  */
	b	L(align_to_16b)

	/* In the case of 4k page boundary cross, the algorithm first align
	   the address to a doubleword, calculate a mask based on alignment
	   to ignore the bytes and continue using doubleword.  */
	.align	4
L(pagecross):
	rldicr	r11,r4,0,59	/* Align the address to 8 bytes boundary.  */
	li	r6,-1		/* MASK = 0xffffffffffffffffUL.  */
	sldi	r9,r9,3		/* Calculate padding.  */
	ld	r7,0(r11)	/* Load doubleword from memory.  */
#ifdef __LITTLE_ENDIAN__
	sld	r9,r6,r9	/* MASK = MASK << padding.  */
#else
	srd	r9,r6,r9	/* MASK = MASK >> padding.  */
#endif
	orc	r9,r7,r9	/* Mask bits that are not part of the
				   string.  */
	li	r7,0
	cmpb	r9,r9,r7	/* Check for null bytes in DWORD1.  */
	cmpdi	cr7,r9,0
	bne	cr7,L(short_path_prepare_2)
	subf	r8,r8,r5	/* Adjust total length.  */
	cmpldi	cr7,r8,8	/* Check if length was reached.  */
	ble	cr7,L(short_path_prepare_2)

	/* For next checks we have aligned address, so we check for more
	   three doublewords to make sure we can read 16 unaligned bytes
	   to start the bulk copy with 16 aligned addresses.  */
	ld	r7,8(r11)
	cmpb	r9,r7,r9
	cmpdi	cr7,r9,0
	bne	cr7,L(short_path_prepare_2)
	addi	r7,r8,-8
	cmpldi	cr7,r7,8
	ble	cr7,L(short_path_prepare_2)
	ld	r7,16(r11)
	cmpb	r9,r7,r9
	cmpdi	cr7,r9,0
	bne	cr7,L(short_path_prepare_2)
	addi	r8,r8,-16
	cmpldi	cr7,r8,8
	ble	cr7,L(short_path_prepare_2)
	ld	r8,24(r11)
	cmpb	r9,r8,r9
	cmpdi	cr7,r9,0
	bne	cr7,L(short_path_prepare_2)

	/* No null byte found in the 32 bytes readed and length not reached,
	   read source again using unaligned loads and store them.  */
	ld	r9,0(r4)
	addi	r29,r3,16
	addi	r5,r5,-16
	std	r9,0(r3)
	ld	r9,8(r4)
	std	r9,8(r3)

	/* Align source to 16 bytes and adjust destiny and size.  */
L(align_to_16b):
	rldicl	r9,r10,0,60
	rldicr	r28,r10,0,59
	add	r12,r5,r9
	subf	r29,r9,r29

	/* The bulk read/compare/copy loads two doublewords, compare and merge
	   in a single register for speed.  This is an attempt to speed up the
	   null-checking process for bigger strings.  */

	cmpldi	cr7,r12,15
	ble	cr7,L(short_path_prepare_1_2)

	/* Main loop for large sizes, unrolled 2 times to get better use of
	   pipeline.  */
	ld	r8,0(28)
	ld	r10,8(28)
	li	r9,0
	cmpb	r7,r8,r9
	cmpb	r9,r10,r9
	or.	r6,r9,r7
	bne	cr0,L(short_path_prepare_2_3)
	addi	r5,r12,-16
	addi	r4,r28,16
	std	r8,0(r29)
	std	r10,8(r29)
	cmpldi	cr7,r5,15
	addi	r9,r29,16
	ble	cr7,L(short_path_1)
	mr	r11,r28
	mr	r6,r29
	li	r30,0
	subfic	r26,r4,48
	subfic	r27,r9,48

	b	L(loop_16b)

	.align	4
L(loop_start):
	ld	r31,0(r11)
	ld	r10,8(r11)
	cmpb	r0,r31,r7
	cmpb	r8,r10,r7
	or.	r7,r0,r8
	addi	r5,r5,-32
	cmpldi	cr7,r5,15
	add	r4,r4,r26
	add	r9,r9,r27
	bne	cr0,L(short_path_prepare_2_2)
	add	r4,r28,r4
	std	r31,0(r6)
	add	r9,r29,r9
	std	r10,8(r6)
	ble	cr7,L(short_path_1)

L(loop_16b):
	ld	r10,16(r11)
	ld	r0,24(r11)
	cmpb	r8,r10,r30
	cmpb	r7,r0,r30
	or.	r7,r8,r7
	addi	r12,r12,-32
	cmpldi	cr7,r12,15
	addi	r11,r11,32
	bne	cr0,L(short_path_2)
	std	r10,16(r6)
	addi	r6,r6,32
	std	r0,-8(r6)
	bgt	cr7,L(loop_start)

	mr	r5,r12
	mr	r4,r11
	mr	r9,r6
	b	L(short_path_1)

	.align	4
L(short_path_prepare_1_1):
	mr	r5,r6
	mr	r4,r7
	b	L(short_path_1)
L(short_path_prepare_1_2):
	mr	r5,r12
	mr	r4,r28
	mr	r9,r29
	b	L(short_path_1)
L(short_path_prepare_2):
	mr	r9,r3
	b	L(short_path_2)
L(short_path_prepare_2_1):
	mr	r5,r6
	mr	r4,r7
	b	L(short_path_2)
L(short_path_prepare_2_2):
	mr	r5,r12
	mr	r4,r11
	mr	r9,r6
	b	L(short_path_2)
L(short_path_prepare_2_3):
	mr	r5,r12
	mr	r4,r28
	mr	r9,r29
	b	L(short_path_2)
L(zero_pad_start_prepare_1):
	mr	r5,r6
	mr	r9,r8
	b	L(zero_pad_start_1)
END (FUNC_NAME)

#ifndef USE_AS_STPNCPY
libc_hidden_builtin_def (strncpy)
#endif
