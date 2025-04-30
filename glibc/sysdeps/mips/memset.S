/* Copyright (C) 2013-2021 Free Software Foundation, Inc.
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
# define PREFETCH_STORE_HINT PREFETCH_HINT_PREPAREFORSTORE
#elif _LIBC
# include <sysdep.h>
# include <regdef.h>
# include <sys/asm.h>
# define PREFETCH_STORE_HINT PREFETCH_HINT_PREPAREFORSTORE
#elif defined _COMPILING_NEWLIB
# include "machine/asm.h"
# include "machine/regdef.h"
# define PREFETCH_STORE_HINT PREFETCH_HINT_PREPAREFORSTORE
#else
# include <regdef.h>
# include <sys/asm.h>
#endif

/* Check to see if the MIPS architecture we are compiling for supports
   prefetching.  */

#if (__mips == 4) || (__mips == 5) || (__mips == 32) || (__mips == 64)
# ifndef DISABLE_PREFETCH
#  define USE_PREFETCH
# endif
#endif

#if defined(_MIPS_SIM) && ((_MIPS_SIM == _ABI64) || (_MIPS_SIM == _ABIN32))
# ifndef DISABLE_DOUBLE
#  define USE_DOUBLE
# endif
#endif

#ifndef USE_DOUBLE
# ifndef DISABLE_DOUBLE_ALIGN
#  define DOUBLE_ALIGN
# endif
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
#  define PTR_ADDIU	daddiu
# else
#  define PTR_ADDIU	addiu
# endif
#endif

/* New R6 instructions that may not be in asm.h.  */
#ifndef PTR_LSA
# if _MIPS_SIM == _ABI64
#  define PTR_LSA        dlsa
# else
#  define PTR_LSA        lsa
# endif
#endif

/* Using PREFETCH_HINT_PREPAREFORSTORE instead of PREFETCH_STORE
   or PREFETCH_STORE_STREAMED offers a large performance advantage
   but PREPAREFORSTORE has some special restrictions to consider.

   Prefetch with the 'prepare for store' hint does not copy a memory
   location into the cache, it just allocates a cache line and zeros
   it out.  This means that if you do not write to the entire cache
   line before writing it out to memory some data will get zero'ed out
   when the cache line is written back to memory and data will be lost.

   There are ifdef'ed sections of this memcpy to make sure that it does not
   do prefetches on cache lines that are not going to be completely written.
   This code is only needed and only used when PREFETCH_STORE_HINT is set to
   PREFETCH_HINT_PREPAREFORSTORE.  This code assumes that cache lines are
   less than MAX_PREFETCH_SIZE bytes and if the cache line is larger it will
   not work correctly.  */

#ifdef USE_PREFETCH
# define PREFETCH_HINT_STORE		1
# define PREFETCH_HINT_STORE_STREAMED	5
# define PREFETCH_HINT_STORE_RETAINED	7
# define PREFETCH_HINT_PREPAREFORSTORE	30

/* If we have not picked out what hints to use at this point use the
   standard load and store prefetch hints.  */
# ifndef PREFETCH_STORE_HINT
#  define PREFETCH_STORE_HINT PREFETCH_HINT_STORE
# endif

/* We double everything when USE_DOUBLE is true so we do 2 prefetches to
   get 64 bytes in that case.  The assumption is that each individual
   prefetch brings in 32 bytes.  */
# ifdef USE_DOUBLE
#  define PREFETCH_CHUNK 64
#  define PREFETCH_FOR_STORE(chunk, reg) \
    pref PREFETCH_STORE_HINT, (chunk)*64(reg); \
    pref PREFETCH_STORE_HINT, ((chunk)*64)+32(reg)
# else
#  define PREFETCH_CHUNK 32
#  define PREFETCH_FOR_STORE(chunk, reg) \
    pref PREFETCH_STORE_HINT, (chunk)*32(reg)
# endif

/* MAX_PREFETCH_SIZE is the maximum size of a prefetch, it must not be less
   than PREFETCH_CHUNK, the assumed size of each prefetch.  If the real size
   of a prefetch is greater than MAX_PREFETCH_SIZE and the PREPAREFORSTORE
   hint is used, the code will not work correctly.  If PREPAREFORSTORE is not
   used than MAX_PREFETCH_SIZE does not matter.  */
# define MAX_PREFETCH_SIZE 128
/* PREFETCH_LIMIT is set based on the fact that we never use an offset greater
   than 5 on a STORE prefetch and that a single prefetch can never be larger
   than MAX_PREFETCH_SIZE.  We add the extra 32 when USE_DOUBLE is set because
   we actually do two prefetches in that case, one 32 bytes after the other.  */
# ifdef USE_DOUBLE
#  define PREFETCH_LIMIT (5 * PREFETCH_CHUNK) + 32 + MAX_PREFETCH_SIZE
# else
#  define PREFETCH_LIMIT (5 * PREFETCH_CHUNK) + MAX_PREFETCH_SIZE
# endif

# if (PREFETCH_STORE_HINT == PREFETCH_HINT_PREPAREFORSTORE) \
    && ((PREFETCH_CHUNK * 4) < MAX_PREFETCH_SIZE)
/* We cannot handle this because the initial prefetches may fetch bytes that
   are before the buffer being copied.  We start copies with an offset
   of 4 so avoid this situation when using PREPAREFORSTORE.  */
#  error "PREFETCH_CHUNK is too large and/or MAX_PREFETCH_SIZE is too small."
# endif
#else /* USE_PREFETCH not defined */
# define PREFETCH_FOR_STORE(offset, reg)
#endif

#if __mips_isa_rev > 5
# if (PREFETCH_STORE_HINT == PREFETCH_HINT_PREPAREFORSTORE)
#  undef PREFETCH_STORE_HINT
#  define PREFETCH_STORE_HINT PREFETCH_HINT_STORE_STREAMED
# endif
# define R6_CODE
#endif

/* Allow the routine to be named something else if desired.  */
#ifndef MEMSET_NAME
# define MEMSET_NAME memset
#endif

/* We load/store 64 bits at a time when USE_DOUBLE is true.
   The C_ prefix stands for CHUNK and is used to avoid macro name
   conflicts with system header files.  */

#ifdef USE_DOUBLE
# define C_ST	sd
# ifdef __MIPSEB
#  define C_STHI	sdl	/* high part is left in big-endian	*/
# else
#  define C_STHI	sdr	/* high part is right in little-endian	*/
# endif
#else
# define C_ST	sw
# ifdef __MIPSEB
#  define C_STHI	swl	/* high part is left in big-endian	*/
# else
#  define C_STHI	swr	/* high part is right in little-endian	*/
# endif
#endif

/* Bookkeeping values for 32 vs. 64 bit mode.  */
#ifdef USE_DOUBLE
# define NSIZE 8
# define NSIZEMASK 0x3f
# define NSIZEDMASK 0x7f
#else
# define NSIZE 4
# define NSIZEMASK 0x1f
# define NSIZEDMASK 0x3f
#endif
#define UNIT(unit) ((unit)*NSIZE)
#define UNITM1(unit) (((unit)*NSIZE)-1)

#ifdef ANDROID_CHANGES
LEAF(MEMSET_NAME,0)
#else
LEAF(MEMSET_NAME)
#endif

	.set	nomips16
	.set	noreorder
/* If the size is less than 2*NSIZE (8 or 16), go to L(lastb).  Regardless of
   size, copy dst pointer to v0 for the return value.  */
	slti	t2,a2,(2 * NSIZE)
	bne	t2,zero,L(lastb)
	move	v0,a0

/* If memset value is not zero, we copy it to all the bytes in a 32 or 64
   bit word.  */
	beq	a1,zero,L(set0)		/* If memset value is zero no smear  */
	PTR_SUBU a3,zero,a0
	nop

	/* smear byte into 32 or 64 bit word */
#if ((__mips == 64) || (__mips == 32)) && (__mips_isa_rev >= 2)
# ifdef USE_DOUBLE
	dins	a1, a1, 8, 8        /* Replicate fill byte into half-word.  */
	dins	a1, a1, 16, 16      /* Replicate fill byte into word.       */
	dins	a1, a1, 32, 32      /* Replicate fill byte into dbl word.   */
# else
	ins	a1, a1, 8, 8        /* Replicate fill byte into half-word.  */
	ins	a1, a1, 16, 16      /* Replicate fill byte into word.       */
# endif
#else
# ifdef USE_DOUBLE
        and     a1,0xff
	dsll	t2,a1,8
	or	a1,t2
	dsll	t2,a1,16
	or	a1,t2
	dsll	t2,a1,32
	or	a1,t2
# else
        and     a1,0xff
	sll	t2,a1,8
	or	a1,t2
	sll	t2,a1,16
	or	a1,t2
# endif
#endif

/* If the destination address is not aligned do a partial store to get it
   aligned.  If it is already aligned just jump to L(aligned).  */
L(set0):
#ifndef R6_CODE
	andi	t2,a3,(NSIZE-1)		/* word-unaligned address?          */
	beq	t2,zero,L(aligned)	/* t2 is the unalignment count      */
	PTR_SUBU a2,a2,t2
	C_STHI	a1,0(a0)
	PTR_ADDU a0,a0,t2
#else /* R6_CODE */
	andi	t2,a0,(NSIZE-1)
	lapc	t9,L(atable)
	PTR_LSA	t9,t2,t9,2
	jrc	t9
L(atable):
	bc	L(aligned)
# ifdef USE_DOUBLE
	bc	L(lb7)
	bc	L(lb6)
	bc	L(lb5)
	bc	L(lb4)
# endif
	bc	L(lb3)
	bc	L(lb2)
	bc	L(lb1)
L(lb7):
	sb	a1,6(a0)
L(lb6):
	sb	a1,5(a0)
L(lb5):
	sb	a1,4(a0)
L(lb4):
	sb	a1,3(a0)
L(lb3):
	sb	a1,2(a0)
L(lb2):
	sb	a1,1(a0)
L(lb1):
	sb	a1,0(a0)

	li	t9,NSIZE
	subu	t2,t9,t2
	PTR_SUBU a2,a2,t2
	PTR_ADDU a0,a0,t2
#endif /* R6_CODE */

L(aligned):
/* If USE_DOUBLE is not set we may still want to align the data on a 16
   byte boundry instead of an 8 byte boundry to maximize the opportunity
   of proAptiv chips to do memory bonding (combining two sequential 4
   byte stores into one 8 byte store).  We know there are at least 4 bytes
   left to store or we would have jumped to L(lastb) earlier in the code.  */
#ifdef DOUBLE_ALIGN
	andi	t2,a3,4
	beq	t2,zero,L(double_aligned)
	PTR_SUBU a2,a2,t2
	sw	a1,0(a0)
	PTR_ADDU a0,a0,t2
L(double_aligned):
#endif

/* Now the destination is aligned to (word or double word) aligned address
   Set a2 to count how many bytes we have to copy after all the 64/128 byte
   chunks are copied and a3 to the dest pointer after all the 64/128 byte
   chunks have been copied.  We will loop, incrementing a0 until it equals
   a3.  */
	andi	t8,a2,NSIZEDMASK /* any whole 64-byte/128-byte chunks? */
	beq	a2,t8,L(chkw)	 /* if a2==t8, no 64-byte/128-byte chunks */
	PTR_SUBU a3,a2,t8	 /* subtract from a2 the reminder */
	PTR_ADDU a3,a0,a3	 /* Now a3 is the final dst after loop */

/* When in the loop we may prefetch with the 'prepare to store' hint,
   in this case the a0+x should not be past the "t0-32" address.  This
   means: for x=128 the last "safe" a0 address is "t0-160".  Alternatively,
   for x=64 the last "safe" a0 address is "t0-96" In the current version we
   will use "prefetch hint,128(a0)", so "t0-160" is the limit.  */
#if defined(USE_PREFETCH) \
    && (PREFETCH_STORE_HINT == PREFETCH_HINT_PREPAREFORSTORE)
	PTR_ADDU t0,a0,a2		/* t0 is the "past the end" address */
	PTR_SUBU t9,t0,PREFETCH_LIMIT	/* t9 is the "last safe pref" address */
#endif
#if defined(USE_PREFETCH) \
    && (PREFETCH_STORE_HINT != PREFETCH_HINT_PREPAREFORSTORE)
	PREFETCH_FOR_STORE (1, a0)
	PREFETCH_FOR_STORE (2, a0)
	PREFETCH_FOR_STORE (3, a0)
#endif

L(loop16w):
#if defined(USE_PREFETCH) \
    && (PREFETCH_STORE_HINT == PREFETCH_HINT_PREPAREFORSTORE)
	sltu	v1,t9,a0		/* If a0 > t9 don't use next prefetch */
	bgtz	v1,L(skip_pref)
	nop
#endif
#ifdef R6_CODE
	PREFETCH_FOR_STORE (2, a0)
#else
	PREFETCH_FOR_STORE (4, a0)
	PREFETCH_FOR_STORE (5, a0)
#endif
L(skip_pref):
	C_ST	a1,UNIT(0)(a0)
	C_ST	a1,UNIT(1)(a0)
	C_ST	a1,UNIT(2)(a0)
	C_ST	a1,UNIT(3)(a0)
	C_ST	a1,UNIT(4)(a0)
	C_ST	a1,UNIT(5)(a0)
	C_ST	a1,UNIT(6)(a0)
	C_ST	a1,UNIT(7)(a0)
	C_ST	a1,UNIT(8)(a0)
	C_ST	a1,UNIT(9)(a0)
	C_ST	a1,UNIT(10)(a0)
	C_ST	a1,UNIT(11)(a0)
	C_ST	a1,UNIT(12)(a0)
	C_ST	a1,UNIT(13)(a0)
	C_ST	a1,UNIT(14)(a0)
	C_ST	a1,UNIT(15)(a0)
	PTR_ADDIU a0,a0,UNIT(16)	/* adding 64/128 to dest */
	bne	a0,a3,L(loop16w)
	nop
	move	a2,t8

/* Here we have dest word-aligned but less than 64-bytes or 128 bytes to go.
   Check for a 32(64) byte chunk and copy if there is one.  Otherwise
   jump down to L(chk1w) to handle the tail end of the copy.  */
L(chkw):
	andi	t8,a2,NSIZEMASK	/* is there a 32-byte/64-byte chunk.  */
				/* the t8 is the reminder count past 32-bytes */
	beq	a2,t8,L(chk1w)/* when a2==t8, no 32-byte chunk */
	nop
	C_ST	a1,UNIT(0)(a0)
	C_ST	a1,UNIT(1)(a0)
	C_ST	a1,UNIT(2)(a0)
	C_ST	a1,UNIT(3)(a0)
	C_ST	a1,UNIT(4)(a0)
	C_ST	a1,UNIT(5)(a0)
	C_ST	a1,UNIT(6)(a0)
	C_ST	a1,UNIT(7)(a0)
	PTR_ADDIU a0,a0,UNIT(8)

/* Here we have less than 32(64) bytes to set.  Set up for a loop to
   copy one word (or double word) at a time.  Set a2 to count how many
   bytes we have to copy after all the word (or double word) chunks are
   copied and a3 to the dest pointer after all the (d)word chunks have
   been copied.  We will loop, incrementing a0 until a0 equals a3.  */
L(chk1w):
	andi	a2,t8,(NSIZE-1)	/* a2 is the reminder past one (d)word chunks */
	beq	a2,t8,L(lastb)
	PTR_SUBU a3,t8,a2	/* a3 is count of bytes in one (d)word chunks */
	PTR_ADDU a3,a0,a3	/* a3 is the dst address after loop */

/* copying in words (4-byte or 8 byte chunks) */
L(wordCopy_loop):
	PTR_ADDIU a0,a0,UNIT(1)
	bne	a0,a3,L(wordCopy_loop)
	C_ST	a1,UNIT(-1)(a0)

/* Copy the last 8 (or 16) bytes */
L(lastb):
	blez	a2,L(leave)
	PTR_ADDU a3,a0,a2       /* a3 is the last dst address */
L(lastbloop):
	PTR_ADDIU a0,a0,1
	bne	a0,a3,L(lastbloop)
	sb	a1,-1(a0)
L(leave):
	j	ra
	nop

	.set	at
	.set	reorder
END(MEMSET_NAME)
#ifndef ANDROID_CHANGES
# ifdef _LIBC
libc_hidden_builtin_def (MEMSET_NAME)
# endif
#endif
