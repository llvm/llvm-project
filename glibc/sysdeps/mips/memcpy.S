/* Copyright (C) 2012-2021 Free Software Foundation, Inc.
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
# define USE_MEMMOVE_FOR_OVERLAP
# define PREFETCH_LOAD_HINT PREFETCH_HINT_LOAD_STREAMED
# define PREFETCH_STORE_HINT PREFETCH_HINT_PREPAREFORSTORE
#elif _LIBC
# include <sysdep.h>
# include <regdef.h>
# include <sys/asm.h>
# define PREFETCH_LOAD_HINT PREFETCH_HINT_LOAD_STREAMED
# define PREFETCH_STORE_HINT PREFETCH_HINT_PREPAREFORSTORE
#elif defined _COMPILING_NEWLIB
# include "machine/asm.h"
# include "machine/regdef.h"
# define PREFETCH_LOAD_HINT PREFETCH_HINT_LOAD_STREAMED
# define PREFETCH_STORE_HINT PREFETCH_HINT_PREPAREFORSTORE
#else
# include <regdef.h>
# include <sys/asm.h>
#endif

#if (_MIPS_ISA == _MIPS_ISA_MIPS4) || (_MIPS_ISA == _MIPS_ISA_MIPS5) || \
    (_MIPS_ISA == _MIPS_ISA_MIPS32) || (_MIPS_ISA == _MIPS_ISA_MIPS64)
# ifndef DISABLE_PREFETCH
#  define USE_PREFETCH
# endif
#endif

#if defined(_MIPS_SIM) && ((_MIPS_SIM == _ABI64) || (_MIPS_SIM == _ABIN32))
# ifndef DISABLE_DOUBLE
#  define USE_DOUBLE
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

/* Some asm.h files do not have the PTR_SRA macro definition.  */
#ifndef PTR_SRA
# ifdef USE_DOUBLE
#  define PTR_SRA		dsra
# else
#  define PTR_SRA		sra
# endif
#endif

/* New R6 instructions that may not be in asm.h.  */
#ifndef PTR_LSA
# if _MIPS_SIM == _ABI64
#  define PTR_LSA	dlsa
# else
#  define PTR_LSA	lsa
# endif
#endif

/*
 * Using PREFETCH_HINT_LOAD_STREAMED instead of PREFETCH_LOAD on load
 * prefetches appears to offer a slight preformance advantage.
 *
 * Using PREFETCH_HINT_PREPAREFORSTORE instead of PREFETCH_STORE
 * or PREFETCH_STORE_STREAMED offers a large performance advantage
 * but PREPAREFORSTORE has some special restrictions to consider.
 *
 * Prefetch with the 'prepare for store' hint does not copy a memory
 * location into the cache, it just allocates a cache line and zeros
 * it out.  This means that if you do not write to the entire cache
 * line before writing it out to memory some data will get zero'ed out
 * when the cache line is written back to memory and data will be lost.
 *
 * Also if you are using this memcpy to copy overlapping buffers it may
 * not behave correctly when using the 'prepare for store' hint.  If you
 * use the 'prepare for store' prefetch on a memory area that is in the
 * memcpy source (as well as the memcpy destination), then you will get
 * some data zero'ed out before you have a chance to read it and data will
 * be lost.
 *
 * If you are going to use this memcpy routine with the 'prepare for store'
 * prefetch you may want to set USE_MEMMOVE_FOR_OVERLAP in order to avoid
 * the problem of running memcpy on overlapping buffers.
 *
 * There are ifdef'ed sections of this memcpy to make sure that it does not
 * do prefetches on cache lines that are not going to be completely written.
 * This code is only needed and only used when PREFETCH_STORE_HINT is set to
 * PREFETCH_HINT_PREPAREFORSTORE.  This code assumes that cache lines are
 * 32 bytes and if the cache line is larger it will not work correctly.
 */

#ifdef USE_PREFETCH
# define PREFETCH_HINT_LOAD		0
# define PREFETCH_HINT_STORE		1
# define PREFETCH_HINT_LOAD_STREAMED	4
# define PREFETCH_HINT_STORE_STREAMED	5
# define PREFETCH_HINT_LOAD_RETAINED	6
# define PREFETCH_HINT_STORE_RETAINED	7
# define PREFETCH_HINT_WRITEBACK_INVAL	25
# define PREFETCH_HINT_PREPAREFORSTORE	30

/*
 * If we have not picked out what hints to use at this point use the
 * standard load and store prefetch hints.
 */
# ifndef PREFETCH_STORE_HINT
#  define PREFETCH_STORE_HINT PREFETCH_HINT_STORE
# endif
# ifndef PREFETCH_LOAD_HINT
#  define PREFETCH_LOAD_HINT PREFETCH_HINT_LOAD
# endif

/*
 * We double everything when USE_DOUBLE is true so we do 2 prefetches to
 * get 64 bytes in that case.  The assumption is that each individual
 * prefetch brings in 32 bytes.
 */

# ifdef USE_DOUBLE
#  define PREFETCH_CHUNK 64
#  define PREFETCH_FOR_LOAD(chunk, reg) \
 pref PREFETCH_LOAD_HINT, (chunk)*64(reg); \
 pref PREFETCH_LOAD_HINT, ((chunk)*64)+32(reg)
#  define PREFETCH_FOR_STORE(chunk, reg) \
 pref PREFETCH_STORE_HINT, (chunk)*64(reg); \
 pref PREFETCH_STORE_HINT, ((chunk)*64)+32(reg)
# else
#  define PREFETCH_CHUNK 32
#  define PREFETCH_FOR_LOAD(chunk, reg) \
 pref PREFETCH_LOAD_HINT, (chunk)*32(reg)
#  define PREFETCH_FOR_STORE(chunk, reg) \
 pref PREFETCH_STORE_HINT, (chunk)*32(reg)
# endif
/* MAX_PREFETCH_SIZE is the maximum size of a prefetch, it must not be less
 * than PREFETCH_CHUNK, the assumed size of each prefetch.  If the real size
 * of a prefetch is greater than MAX_PREFETCH_SIZE and the PREPAREFORSTORE
 * hint is used, the code will not work correctly.  If PREPAREFORSTORE is not
 * used then MAX_PREFETCH_SIZE does not matter.  */
# define MAX_PREFETCH_SIZE 128
/* PREFETCH_LIMIT is set based on the fact that we never use an offset greater
 * than 5 on a STORE prefetch and that a single prefetch can never be larger
 * than MAX_PREFETCH_SIZE.  We add the extra 32 when USE_DOUBLE is set because
 * we actually do two prefetches in that case, one 32 bytes after the other.  */
# ifdef USE_DOUBLE
#  define PREFETCH_LIMIT (5 * PREFETCH_CHUNK) + 32 + MAX_PREFETCH_SIZE
# else
#  define PREFETCH_LIMIT (5 * PREFETCH_CHUNK) + MAX_PREFETCH_SIZE
# endif
# if (PREFETCH_STORE_HINT == PREFETCH_HINT_PREPAREFORSTORE) \
    && ((PREFETCH_CHUNK * 4) < MAX_PREFETCH_SIZE)
/* We cannot handle this because the initial prefetches may fetch bytes that
 * are before the buffer being copied.  We start copies with an offset
 * of 4 so avoid this situation when using PREPAREFORSTORE.  */
#error "PREFETCH_CHUNK is too large and/or MAX_PREFETCH_SIZE is too small."
# endif
#else /* USE_PREFETCH not defined */
# define PREFETCH_FOR_LOAD(offset, reg)
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
#ifndef MEMCPY_NAME
# define MEMCPY_NAME memcpy
#endif

/* We use these 32/64 bit registers as temporaries to do the copying.  */
#define REG0 t0
#define REG1 t1
#define REG2 t2
#define REG3 t3
#if defined(_MIPS_SIM) && ((_MIPS_SIM == _ABIO32) || (_MIPS_SIM == _ABIO64))
# define REG4 t4
# define REG5 t5
# define REG6 t6
# define REG7 t7
#else
# define REG4 ta0
# define REG5 ta1
# define REG6 ta2
# define REG7 ta3
#endif

/* We load/store 64 bits at a time when USE_DOUBLE is true.
 * The C_ prefix stands for CHUNK and is used to avoid macro name
 * conflicts with system header files.  */

#ifdef USE_DOUBLE
# define C_ST	sd
# define C_LD	ld
# ifdef __MIPSEB
#  define C_LDHI	ldl	/* high part is left in big-endian	*/
#  define C_STHI	sdl	/* high part is left in big-endian	*/
#  define C_LDLO	ldr	/* low part is right in big-endian	*/
#  define C_STLO	sdr	/* low part is right in big-endian	*/
# else
#  define C_LDHI	ldr	/* high part is right in little-endian	*/
#  define C_STHI	sdr	/* high part is right in little-endian	*/
#  define C_LDLO	ldl	/* low part is left in little-endian	*/
#  define C_STLO	sdl	/* low part is left in little-endian	*/
# endif
# define C_ALIGN	dalign	/* r6 align instruction			*/
#else
# define C_ST	sw
# define C_LD	lw
# ifdef __MIPSEB
#  define C_LDHI	lwl	/* high part is left in big-endian	*/
#  define C_STHI	swl	/* high part is left in big-endian	*/
#  define C_LDLO	lwr	/* low part is right in big-endian	*/
#  define C_STLO	swr	/* low part is right in big-endian	*/
# else
#  define C_LDHI	lwr	/* high part is right in little-endian	*/
#  define C_STHI	swr	/* high part is right in little-endian	*/
#  define C_LDLO	lwl	/* low part is left in little-endian	*/
#  define C_STLO	swl	/* low part is left in little-endian	*/
# endif
# define C_ALIGN	align	/* r6 align instruction			*/
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
LEAF(MEMCPY_NAME, 0)
#else
LEAF(MEMCPY_NAME)
#endif
	.set	nomips16
	.set	noreorder
/*
 * Below we handle the case where memcpy is called with overlapping src and dst.
 * Although memcpy is not required to handle this case, some parts of Android
 * like Skia rely on such usage. We call memmove to handle such cases.
 */
#ifdef USE_MEMMOVE_FOR_OVERLAP
	PTR_SUBU t0,a0,a1
	PTR_SRA	t2,t0,31
	xor	t1,t0,t2
	PTR_SUBU t0,t1,t2
	sltu	t2,t0,a2
	beq	t2,zero,L(memcpy)
	la	t9,memmove
	jr	t9
	 nop
L(memcpy):
#endif
/*
 * If the size is less than 2*NSIZE (8 or 16), go to L(lastb).  Regardless of
 * size, copy dst pointer to v0 for the return value.
 */
	slti	t2,a2,(2 * NSIZE)
	bne	t2,zero,L(lasts)
#if defined(RETURN_FIRST_PREFETCH) || defined(RETURN_LAST_PREFETCH)
	move	v0,zero
#else
	move	v0,a0
#endif

#ifndef R6_CODE

/*
 * If src and dst have different alignments, go to L(unaligned), if they
 * have the same alignment (but are not actually aligned) do a partial
 * load/store to make them aligned.  If they are both already aligned
 * we can start copying at L(aligned).
 */
	xor	t8,a1,a0
	andi	t8,t8,(NSIZE-1)		/* t8 is a0/a1 word-displacement */
	bne	t8,zero,L(unaligned)
	PTR_SUBU a3, zero, a0

	andi	a3,a3,(NSIZE-1)		/* copy a3 bytes to align a0/a1	  */
	beq	a3,zero,L(aligned)	/* if a3=0, it is already aligned */
	PTR_SUBU a2,a2,a3		/* a2 is the remining bytes count */

	C_LDHI	t8,0(a1)
	PTR_ADDU a1,a1,a3
	C_STHI	t8,0(a0)
	PTR_ADDU a0,a0,a3

#else /* R6_CODE */

/*
 * Align the destination and hope that the source gets aligned too.  If it
 * doesn't we jump to L(r6_unaligned*) to do unaligned copies using the r6
 * align instruction.
 */
	andi	t8,a0,7
	lapc	t9,L(atable)
	PTR_LSA	t9,t8,t9,2
	jrc	t9
L(atable):
	bc	L(lb0)
	bc	L(lb7)
	bc	L(lb6)
	bc	L(lb5)
	bc	L(lb4)
	bc	L(lb3)
	bc	L(lb2)
	bc	L(lb1)
L(lb7):
	lb	a3, 6(a1)
	sb	a3, 6(a0)
L(lb6):
	lb	a3, 5(a1)
	sb	a3, 5(a0)
L(lb5):
	lb	a3, 4(a1)
	sb	a3, 4(a0)
L(lb4):
	lb	a3, 3(a1)
	sb	a3, 3(a0)
L(lb3):
	lb	a3, 2(a1)
	sb	a3, 2(a0)
L(lb2):
	lb	a3, 1(a1)
	sb	a3, 1(a0)
L(lb1):
	lb	a3, 0(a1)
	sb	a3, 0(a0)

	li	t9,8
	subu	t8,t9,t8
	PTR_SUBU a2,a2,t8
	PTR_ADDU a0,a0,t8
	PTR_ADDU a1,a1,t8
L(lb0):

	andi	t8,a1,(NSIZE-1)
	lapc	t9,L(jtable)
	PTR_LSA	t9,t8,t9,2
	jrc	t9
L(jtable):
        bc      L(aligned)
        bc      L(r6_unaligned1)
        bc      L(r6_unaligned2)
        bc      L(r6_unaligned3)
# ifdef USE_DOUBLE
        bc      L(r6_unaligned4)
        bc      L(r6_unaligned5)
        bc      L(r6_unaligned6)
        bc      L(r6_unaligned7)
# endif
#endif /* R6_CODE */

L(aligned):

/*
 * Now dst/src are both aligned to (word or double word) aligned addresses
 * Set a2 to count how many bytes we have to copy after all the 64/128 byte
 * chunks are copied and a3 to the dst pointer after all the 64/128 byte
 * chunks have been copied.  We will loop, incrementing a0 and a1 until a0
 * equals a3.
 */

	andi	t8,a2,NSIZEDMASK /* any whole 64-byte/128-byte chunks? */
	beq	a2,t8,L(chkw)	 /* if a2==t8, no 64-byte/128-byte chunks */
	PTR_SUBU a3,a2,t8	 /* subtract from a2 the reminder */
	PTR_ADDU a3,a0,a3	 /* Now a3 is the final dst after loop */

/* When in the loop we may prefetch with the 'prepare to store' hint,
 * in this case the a0+x should not be past the "t0-32" address.  This
 * means: for x=128 the last "safe" a0 address is "t0-160".  Alternatively,
 * for x=64 the last "safe" a0 address is "t0-96" In the current version we
 * will use "prefetch hint,128(a0)", so "t0-160" is the limit.
 */
#if defined(USE_PREFETCH) && (PREFETCH_STORE_HINT == PREFETCH_HINT_PREPAREFORSTORE)
	PTR_ADDU t0,a0,a2		/* t0 is the "past the end" address */
	PTR_SUBU t9,t0,PREFETCH_LIMIT	/* t9 is the "last safe pref" address */
#endif
	PREFETCH_FOR_LOAD  (0, a1)
	PREFETCH_FOR_LOAD  (1, a1)
	PREFETCH_FOR_LOAD  (2, a1)
	PREFETCH_FOR_LOAD  (3, a1)
#if defined(USE_PREFETCH) && (PREFETCH_STORE_HINT != PREFETCH_HINT_PREPAREFORSTORE)
	PREFETCH_FOR_STORE (1, a0)
	PREFETCH_FOR_STORE (2, a0)
	PREFETCH_FOR_STORE (3, a0)
#endif
#if defined(RETURN_FIRST_PREFETCH) && defined(USE_PREFETCH)
# if PREFETCH_STORE_HINT == PREFETCH_HINT_PREPAREFORSTORE
	sltu    v1,t9,a0
	bgtz    v1,L(skip_set)
	nop
	PTR_ADDIU v0,a0,(PREFETCH_CHUNK*4)
L(skip_set):
# else
	PTR_ADDIU v0,a0,(PREFETCH_CHUNK*1)
# endif
#endif
#if defined(RETURN_LAST_PREFETCH) && defined(USE_PREFETCH) \
    && (PREFETCH_STORE_HINT != PREFETCH_HINT_PREPAREFORSTORE)
	PTR_ADDIU v0,a0,(PREFETCH_CHUNK*3)
# ifdef USE_DOUBLE
	PTR_ADDIU v0,v0,32
# endif
#endif
L(loop16w):
	C_LD	t0,UNIT(0)(a1)
#if defined(USE_PREFETCH) && (PREFETCH_STORE_HINT == PREFETCH_HINT_PREPAREFORSTORE)
	sltu	v1,t9,a0		/* If a0 > t9 don't use next prefetch */
	bgtz	v1,L(skip_pref)
#endif
	C_LD	t1,UNIT(1)(a1)
#ifdef R6_CODE
	PREFETCH_FOR_STORE (2, a0)
#else
	PREFETCH_FOR_STORE (4, a0)
	PREFETCH_FOR_STORE (5, a0)
#endif
#if defined(RETURN_LAST_PREFETCH) && defined(USE_PREFETCH)
	PTR_ADDIU v0,a0,(PREFETCH_CHUNK*5)
# ifdef USE_DOUBLE
	PTR_ADDIU v0,v0,32
# endif
#endif
L(skip_pref):
	C_LD	REG2,UNIT(2)(a1)
	C_LD	REG3,UNIT(3)(a1)
	C_LD	REG4,UNIT(4)(a1)
	C_LD	REG5,UNIT(5)(a1)
	C_LD	REG6,UNIT(6)(a1)
	C_LD	REG7,UNIT(7)(a1)
#ifdef R6_CODE
	PREFETCH_FOR_LOAD (3, a1)
#else
	PREFETCH_FOR_LOAD (4, a1)
#endif
	C_ST	t0,UNIT(0)(a0)
	C_ST	t1,UNIT(1)(a0)
	C_ST	REG2,UNIT(2)(a0)
	C_ST	REG3,UNIT(3)(a0)
	C_ST	REG4,UNIT(4)(a0)
	C_ST	REG5,UNIT(5)(a0)
	C_ST	REG6,UNIT(6)(a0)
	C_ST	REG7,UNIT(7)(a0)

	C_LD	t0,UNIT(8)(a1)
	C_LD	t1,UNIT(9)(a1)
	C_LD	REG2,UNIT(10)(a1)
	C_LD	REG3,UNIT(11)(a1)
	C_LD	REG4,UNIT(12)(a1)
	C_LD	REG5,UNIT(13)(a1)
	C_LD	REG6,UNIT(14)(a1)
	C_LD	REG7,UNIT(15)(a1)
#ifndef R6_CODE
        PREFETCH_FOR_LOAD (5, a1)
#endif
	C_ST	t0,UNIT(8)(a0)
	C_ST	t1,UNIT(9)(a0)
	C_ST	REG2,UNIT(10)(a0)
	C_ST	REG3,UNIT(11)(a0)
	C_ST	REG4,UNIT(12)(a0)
	C_ST	REG5,UNIT(13)(a0)
	C_ST	REG6,UNIT(14)(a0)
	C_ST	REG7,UNIT(15)(a0)
	PTR_ADDIU a0,a0,UNIT(16)	/* adding 64/128 to dest */
	bne	a0,a3,L(loop16w)
	PTR_ADDIU a1,a1,UNIT(16)	/* adding 64/128 to src */
	move	a2,t8

/* Here we have src and dest word-aligned but less than 64-bytes or
 * 128 bytes to go.  Check for a 32(64) byte chunk and copy if there
 * is one.  Otherwise jump down to L(chk1w) to handle the tail end of
 * the copy.
 */

L(chkw):
	PREFETCH_FOR_LOAD (0, a1)
	andi	t8,a2,NSIZEMASK	/* Is there a 32-byte/64-byte chunk.  */
				/* The t8 is the reminder count past 32-bytes */
	beq	a2,t8,L(chk1w)	/* When a2=t8, no 32-byte chunk  */
	nop
	C_LD	t0,UNIT(0)(a1)
	C_LD	t1,UNIT(1)(a1)
	C_LD	REG2,UNIT(2)(a1)
	C_LD	REG3,UNIT(3)(a1)
	C_LD	REG4,UNIT(4)(a1)
	C_LD	REG5,UNIT(5)(a1)
	C_LD	REG6,UNIT(6)(a1)
	C_LD	REG7,UNIT(7)(a1)
	PTR_ADDIU a1,a1,UNIT(8)
	C_ST	t0,UNIT(0)(a0)
	C_ST	t1,UNIT(1)(a0)
	C_ST	REG2,UNIT(2)(a0)
	C_ST	REG3,UNIT(3)(a0)
	C_ST	REG4,UNIT(4)(a0)
	C_ST	REG5,UNIT(5)(a0)
	C_ST	REG6,UNIT(6)(a0)
	C_ST	REG7,UNIT(7)(a0)
	PTR_ADDIU a0,a0,UNIT(8)

/*
 * Here we have less than 32(64) bytes to copy.  Set up for a loop to
 * copy one word (or double word) at a time.  Set a2 to count how many
 * bytes we have to copy after all the word (or double word) chunks are
 * copied and a3 to the dst pointer after all the (d)word chunks have
 * been copied.  We will loop, incrementing a0 and a1 until a0 equals a3.
 */
L(chk1w):
	andi	a2,t8,(NSIZE-1)	/* a2 is the reminder past one (d)word chunks */
	beq	a2,t8,L(lastw)
	PTR_SUBU a3,t8,a2	/* a3 is count of bytes in one (d)word chunks */
	PTR_ADDU a3,a0,a3	/* a3 is the dst address after loop */

/* copying in words (4-byte or 8-byte chunks) */
L(wordCopy_loop):
	C_LD	REG3,UNIT(0)(a1)
	PTR_ADDIU a0,a0,UNIT(1)
	PTR_ADDIU a1,a1,UNIT(1)
	bne	a0,a3,L(wordCopy_loop)
	C_ST	REG3,UNIT(-1)(a0)

/* If we have been copying double words, see if we can copy a single word
   before doing byte copies.  We can have, at most, one word to copy.  */

L(lastw):
#ifdef USE_DOUBLE
	andi    t8,a2,3		/* a2 is the remainder past 4 byte chunks.  */
	beq	t8,a2,L(lastb)
	move	a2,t8
	lw	REG3,0(a1)
	sw	REG3,0(a0)
	PTR_ADDIU a0,a0,4
	PTR_ADDIU a1,a1,4
#endif

/* Copy the last 8 (or 16) bytes */
L(lastb):
	blez	a2,L(leave)
	PTR_ADDU a3,a0,a2	/* a3 is the last dst address */
L(lastbloop):
	lb	v1,0(a1)
	PTR_ADDIU a0,a0,1
	PTR_ADDIU a1,a1,1
	bne	a0,a3,L(lastbloop)
	sb	v1,-1(a0)
L(leave):
	j	ra
	nop

/* We jump here with a memcpy of less than 8 or 16 bytes, depending on
   whether or not USE_DOUBLE is defined.  Instead of just doing byte
   copies, check the alignment and size and use lw/sw if possible.
   Otherwise, do byte copies.  */

L(lasts):
	andi	t8,a2,3
	beq	t8,a2,L(lastb)

	andi	t9,a0,3
	bne	t9,zero,L(lastb)
	andi	t9,a1,3
	bne	t9,zero,L(lastb)

	PTR_SUBU a3,a2,t8
	PTR_ADDU a3,a0,a3

L(wcopy_loop):
	lw	REG3,0(a1)
	PTR_ADDIU a0,a0,4
	PTR_ADDIU a1,a1,4
	bne	a0,a3,L(wcopy_loop)
	sw	REG3,-4(a0)

	b	L(lastb)
	move	a2,t8

#ifndef R6_CODE
/*
 * UNALIGNED case, got here with a3 = "negu a0"
 * This code is nearly identical to the aligned code above
 * but only the destination (not the source) gets aligned
 * so we need to do partial loads of the source followed
 * by normal stores to the destination (once we have aligned
 * the destination).
 */

L(unaligned):
	andi	a3,a3,(NSIZE-1)	/* copy a3 bytes to align a0/a1 */
	beqz	a3,L(ua_chk16w) /* if a3=0, it is already aligned */
	PTR_SUBU a2,a2,a3	/* a2 is the remining bytes count */

	C_LDHI	v1,UNIT(0)(a1)
	C_LDLO	v1,UNITM1(1)(a1)
	PTR_ADDU a1,a1,a3
	C_STHI	v1,UNIT(0)(a0)
	PTR_ADDU a0,a0,a3

/*
 *  Now the destination (but not the source) is aligned
 * Set a2 to count how many bytes we have to copy after all the 64/128 byte
 * chunks are copied and a3 to the dst pointer after all the 64/128 byte
 * chunks have been copied.  We will loop, incrementing a0 and a1 until a0
 * equals a3.
 */

L(ua_chk16w):
	andi	t8,a2,NSIZEDMASK /* any whole 64-byte/128-byte chunks? */
	beq	a2,t8,L(ua_chkw) /* if a2==t8, no 64-byte/128-byte chunks */
	PTR_SUBU a3,a2,t8	 /* subtract from a2 the reminder */
	PTR_ADDU a3,a0,a3	 /* Now a3 is the final dst after loop */

# if defined(USE_PREFETCH) && (PREFETCH_STORE_HINT == PREFETCH_HINT_PREPAREFORSTORE)
	PTR_ADDU t0,a0,a2	  /* t0 is the "past the end" address */
	PTR_SUBU t9,t0,PREFETCH_LIMIT /* t9 is the "last safe pref" address */
# endif
	PREFETCH_FOR_LOAD  (0, a1)
	PREFETCH_FOR_LOAD  (1, a1)
	PREFETCH_FOR_LOAD  (2, a1)
# if defined(USE_PREFETCH) && (PREFETCH_STORE_HINT != PREFETCH_HINT_PREPAREFORSTORE)
	PREFETCH_FOR_STORE (1, a0)
	PREFETCH_FOR_STORE (2, a0)
	PREFETCH_FOR_STORE (3, a0)
# endif
# if defined(RETURN_FIRST_PREFETCH) && defined(USE_PREFETCH)
#  if (PREFETCH_STORE_HINT == PREFETCH_HINT_PREPAREFORSTORE)
	sltu    v1,t9,a0
	bgtz    v1,L(ua_skip_set)
	nop
	PTR_ADDIU v0,a0,(PREFETCH_CHUNK*4)
L(ua_skip_set):
#  else
	PTR_ADDIU v0,a0,(PREFETCH_CHUNK*1)
#  endif
# endif
L(ua_loop16w):
	PREFETCH_FOR_LOAD  (3, a1)
	C_LDHI	t0,UNIT(0)(a1)
	C_LDHI	t1,UNIT(1)(a1)
	C_LDHI	REG2,UNIT(2)(a1)
# if defined(USE_PREFETCH) && (PREFETCH_STORE_HINT == PREFETCH_HINT_PREPAREFORSTORE)
	sltu	v1,t9,a0
	bgtz	v1,L(ua_skip_pref)
# endif
	C_LDHI	REG3,UNIT(3)(a1)
	PREFETCH_FOR_STORE (4, a0)
	PREFETCH_FOR_STORE (5, a0)
L(ua_skip_pref):
	C_LDHI	REG4,UNIT(4)(a1)
	C_LDHI	REG5,UNIT(5)(a1)
	C_LDHI	REG6,UNIT(6)(a1)
	C_LDHI	REG7,UNIT(7)(a1)
	C_LDLO	t0,UNITM1(1)(a1)
	C_LDLO	t1,UNITM1(2)(a1)
	C_LDLO	REG2,UNITM1(3)(a1)
	C_LDLO	REG3,UNITM1(4)(a1)
	C_LDLO	REG4,UNITM1(5)(a1)
	C_LDLO	REG5,UNITM1(6)(a1)
	C_LDLO	REG6,UNITM1(7)(a1)
	C_LDLO	REG7,UNITM1(8)(a1)
        PREFETCH_FOR_LOAD (4, a1)
	C_ST	t0,UNIT(0)(a0)
	C_ST	t1,UNIT(1)(a0)
	C_ST	REG2,UNIT(2)(a0)
	C_ST	REG3,UNIT(3)(a0)
	C_ST	REG4,UNIT(4)(a0)
	C_ST	REG5,UNIT(5)(a0)
	C_ST	REG6,UNIT(6)(a0)
	C_ST	REG7,UNIT(7)(a0)
	C_LDHI	t0,UNIT(8)(a1)
	C_LDHI	t1,UNIT(9)(a1)
	C_LDHI	REG2,UNIT(10)(a1)
	C_LDHI	REG3,UNIT(11)(a1)
	C_LDHI	REG4,UNIT(12)(a1)
	C_LDHI	REG5,UNIT(13)(a1)
	C_LDHI	REG6,UNIT(14)(a1)
	C_LDHI	REG7,UNIT(15)(a1)
	C_LDLO	t0,UNITM1(9)(a1)
	C_LDLO	t1,UNITM1(10)(a1)
	C_LDLO	REG2,UNITM1(11)(a1)
	C_LDLO	REG3,UNITM1(12)(a1)
	C_LDLO	REG4,UNITM1(13)(a1)
	C_LDLO	REG5,UNITM1(14)(a1)
	C_LDLO	REG6,UNITM1(15)(a1)
	C_LDLO	REG7,UNITM1(16)(a1)
        PREFETCH_FOR_LOAD (5, a1)
	C_ST	t0,UNIT(8)(a0)
	C_ST	t1,UNIT(9)(a0)
	C_ST	REG2,UNIT(10)(a0)
	C_ST	REG3,UNIT(11)(a0)
	C_ST	REG4,UNIT(12)(a0)
	C_ST	REG5,UNIT(13)(a0)
	C_ST	REG6,UNIT(14)(a0)
	C_ST	REG7,UNIT(15)(a0)
	PTR_ADDIU a0,a0,UNIT(16)	/* adding 64/128 to dest */
	bne	a0,a3,L(ua_loop16w)
	PTR_ADDIU a1,a1,UNIT(16)	/* adding 64/128 to src */
	move	a2,t8

/* Here we have src and dest word-aligned but less than 64-bytes or
 * 128 bytes to go.  Check for a 32(64) byte chunk and copy if there
 * is one.  Otherwise jump down to L(ua_chk1w) to handle the tail end of
 * the copy.  */

L(ua_chkw):
	PREFETCH_FOR_LOAD (0, a1)
	andi	t8,a2,NSIZEMASK	  /* Is there a 32-byte/64-byte chunk.  */
				  /* t8 is the reminder count past 32-bytes */
	beq	a2,t8,L(ua_chk1w) /* When a2=t8, no 32-byte chunk */
	nop
	C_LDHI	t0,UNIT(0)(a1)
	C_LDHI	t1,UNIT(1)(a1)
	C_LDHI	REG2,UNIT(2)(a1)
	C_LDHI	REG3,UNIT(3)(a1)
	C_LDHI	REG4,UNIT(4)(a1)
	C_LDHI	REG5,UNIT(5)(a1)
	C_LDHI	REG6,UNIT(6)(a1)
	C_LDHI	REG7,UNIT(7)(a1)
	C_LDLO	t0,UNITM1(1)(a1)
	C_LDLO	t1,UNITM1(2)(a1)
	C_LDLO	REG2,UNITM1(3)(a1)
	C_LDLO	REG3,UNITM1(4)(a1)
	C_LDLO	REG4,UNITM1(5)(a1)
	C_LDLO	REG5,UNITM1(6)(a1)
	C_LDLO	REG6,UNITM1(7)(a1)
	C_LDLO	REG7,UNITM1(8)(a1)
	PTR_ADDIU a1,a1,UNIT(8)
	C_ST	t0,UNIT(0)(a0)
	C_ST	t1,UNIT(1)(a0)
	C_ST	REG2,UNIT(2)(a0)
	C_ST	REG3,UNIT(3)(a0)
	C_ST	REG4,UNIT(4)(a0)
	C_ST	REG5,UNIT(5)(a0)
	C_ST	REG6,UNIT(6)(a0)
	C_ST	REG7,UNIT(7)(a0)
	PTR_ADDIU a0,a0,UNIT(8)
/*
 * Here we have less than 32(64) bytes to copy.  Set up for a loop to
 * copy one word (or double word) at a time.
 */
L(ua_chk1w):
	andi	a2,t8,(NSIZE-1)	/* a2 is the reminder past one (d)word chunks */
	beq	a2,t8,L(ua_smallCopy)
	PTR_SUBU a3,t8,a2	/* a3 is count of bytes in one (d)word chunks */
	PTR_ADDU a3,a0,a3	/* a3 is the dst address after loop */

/* copying in words (4-byte or 8-byte chunks) */
L(ua_wordCopy_loop):
	C_LDHI	v1,UNIT(0)(a1)
	C_LDLO	v1,UNITM1(1)(a1)
	PTR_ADDIU a0,a0,UNIT(1)
	PTR_ADDIU a1,a1,UNIT(1)
	bne	a0,a3,L(ua_wordCopy_loop)
	C_ST	v1,UNIT(-1)(a0)

/* Copy the last 8 (or 16) bytes */
L(ua_smallCopy):
	beqz	a2,L(leave)
	PTR_ADDU a3,a0,a2	/* a3 is the last dst address */
L(ua_smallCopy_loop):
	lb	v1,0(a1)
	PTR_ADDIU a0,a0,1
	PTR_ADDIU a1,a1,1
	bne	a0,a3,L(ua_smallCopy_loop)
	sb	v1,-1(a0)

	j	ra
	nop

#else /* R6_CODE */

# ifdef __MIPSEB
#  define SWAP_REGS(X,Y) X, Y
#  define ALIGN_OFFSET(N) (N)
# else
#  define SWAP_REGS(X,Y) Y, X
#  define ALIGN_OFFSET(N) (NSIZE-N)
# endif
# define R6_UNALIGNED_WORD_COPY(BYTEOFFSET) \
	andi	REG7, a2, (NSIZE-1);/* REG7 is # of bytes to by bytes.     */ \
	beq	REG7, a2, L(lastb); /* Check for bytes to copy by word	   */ \
	PTR_SUBU a3, a2, REG7;	/* a3 is number of bytes to be copied in   */ \
				/* (d)word chunks.			   */ \
	move	a2, REG7;	/* a2 is # of bytes to copy byte by byte   */ \
				/* after word loop is finished.		   */ \
	PTR_ADDU REG6, a0, a3;	/* REG6 is the dst address after loop.	   */ \
	PTR_SUBU REG2, a1, t8;	/* REG2 is the aligned src address.	   */ \
	PTR_ADDU a1, a1, a3;	/* a1 is addr of source after word loop.   */ \
	C_LD	t0, UNIT(0)(REG2);  /* Load first part of source.	   */ \
L(r6_ua_wordcopy##BYTEOFFSET):						      \
	C_LD	t1, UNIT(1)(REG2);  /* Load second part of source.	   */ \
	C_ALIGN	REG3, SWAP_REGS(t1,t0), ALIGN_OFFSET(BYTEOFFSET);	      \
	PTR_ADDIU a0, a0, UNIT(1);  /* Increment destination pointer.	   */ \
	PTR_ADDIU REG2, REG2, UNIT(1); /* Increment aligned source pointer.*/ \
	move	t0, t1;		/* Move second part of source to first.	   */ \
	bne	a0, REG6,L(r6_ua_wordcopy##BYTEOFFSET);			      \
	C_ST	REG3, UNIT(-1)(a0);					      \
	j	L(lastb);						      \
	nop

	/* We are generating R6 code, the destination is 4 byte aligned and
	   the source is not 4 byte aligned. t8 is 1, 2, or 3 depending on the
           alignment of the source.  */

L(r6_unaligned1):
	R6_UNALIGNED_WORD_COPY(1)
L(r6_unaligned2):
	R6_UNALIGNED_WORD_COPY(2)
L(r6_unaligned3):
	R6_UNALIGNED_WORD_COPY(3)
# ifdef USE_DOUBLE
L(r6_unaligned4):
	R6_UNALIGNED_WORD_COPY(4)
L(r6_unaligned5):
	R6_UNALIGNED_WORD_COPY(5)
L(r6_unaligned6):
	R6_UNALIGNED_WORD_COPY(6)
L(r6_unaligned7):
	R6_UNALIGNED_WORD_COPY(7)
# endif
#endif /* R6_CODE */

	.set	at
	.set	reorder
END(MEMCPY_NAME)
#ifndef ANDROID_CHANGES
# ifdef _LIBC
libc_hidden_builtin_def (MEMCPY_NAME)
# endif
#endif
