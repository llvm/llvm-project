/* Atomic operations used inside libc.  Linux/SH version.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
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

#include <stdint.h>


typedef int8_t atomic8_t;
typedef uint8_t uatomic8_t;
typedef int_fast8_t atomic_fast8_t;
typedef uint_fast8_t uatomic_fast8_t;

typedef int16_t atomic16_t;
typedef uint16_t uatomic16_t;
typedef int_fast16_t atomic_fast16_t;
typedef uint_fast16_t uatomic_fast16_t;

typedef int32_t atomic32_t;
typedef uint32_t uatomic32_t;
typedef int_fast32_t atomic_fast32_t;
typedef uint_fast32_t uatomic_fast32_t;

typedef int64_t atomic64_t;
typedef uint64_t uatomic64_t;
typedef int_fast64_t atomic_fast64_t;
typedef uint_fast64_t uatomic_fast64_t;

typedef intptr_t atomicptr_t;
typedef uintptr_t uatomicptr_t;
typedef intmax_t atomic_max_t;
typedef uintmax_t uatomic_max_t;

#define __HAVE_64B_ATOMICS 0
#define USE_ATOMIC_COMPILER_BUILTINS 0

/* XXX Is this actually correct?  */
#define ATOMIC_EXCHANGE_USES_CAS 1

/* SH kernel has implemented a gUSA ("g" User Space Atomicity) support
   for the user space atomicity. The atomicity macros use this scheme.

  Reference:
    Niibe Yutaka, "gUSA: Simple and Efficient User Space Atomicity
    Emulation with Little Kernel Modification", Linux Conference 2002,
    Japan. http://lc.linux.or.jp/lc2002/papers/niibe0919h.pdf (in
    Japanese).

    B.N. Bershad, D. Redell, and J. Ellis, "Fast Mutual Exclusion for
    Uniprocessors",  Proceedings of the Fifth Architectural Support for
    Programming Languages and Operating Systems (ASPLOS), pp. 223-233,
    October 1992. http://www.cs.washington.edu/homes/bershad/Papers/Rcs.ps

  SuperH ABI:
      r15:    -(size of atomic instruction sequence) < 0
      r0:     end point
      r1:     saved stack pointer
*/

#define __arch_compare_and_exchange_val_8_acq(mem, newval, oldval) \
  ({ __typeof (*(mem)) __result; \
     __asm __volatile ("\
	mova 1f,r0\n\
	.align 2\n\
	mov r15,r1\n\
	mov #(0f-1f),r15\n\
     0: mov.b @%1,%0\n\
	cmp/eq %0,%3\n\
	bf 1f\n\
	mov.b %2,@%1\n\
     1: mov r1,r15"\
	: "=&r" (__result) : "u" (mem), "u" (newval), "u" (oldval) \
	: "r0", "r1", "t", "memory"); \
     __result; })

#define __arch_compare_and_exchange_val_16_acq(mem, newval, oldval) \
  ({ __typeof (*(mem)) __result; \
     __asm __volatile ("\
	mova 1f,r0\n\
	mov r15,r1\n\
	.align 2\n\
	mov #(0f-1f),r15\n\
	mov #-8,r15\n\
     0: mov.w @%1,%0\n\
	cmp/eq %0,%3\n\
	bf 1f\n\
	mov.w %2,@%1\n\
     1: mov r1,r15"\
	: "=&r" (__result) : "u" (mem), "u" (newval), "u" (oldval) \
	: "r0", "r1", "t", "memory"); \
     __result; })

#define __arch_compare_and_exchange_val_32_acq(mem, newval, oldval) \
  ({ __typeof (*(mem)) __result; \
     __asm __volatile ("\
	mova 1f,r0\n\
	.align 2\n\
	mov r15,r1\n\
	mov #(0f-1f),r15\n\
     0: mov.l @%1,%0\n\
	cmp/eq %0,%3\n\
	bf 1f\n\
	mov.l %2,@%1\n\
     1: mov r1,r15"\
	: "=&r" (__result) : "u" (mem), "u" (newval), "u" (oldval) \
	: "r0", "r1", "t", "memory"); \
     __result; })

/* XXX We do not really need 64-bit compare-and-exchange.  At least
   not in the moment.  Using it would mean causing portability
   problems since not many other 32-bit architectures have support for
   such an operation.  So don't define any code for now.  */

# define __arch_compare_and_exchange_val_64_acq(mem, newval, oldval) \
  (abort (), (__typeof (*mem)) 0)

#define atomic_exchange_and_add(mem, value) \
  ({ __typeof (*(mem)) __result, __tmp, __value = (value); \
     if (sizeof (*(mem)) == 1) \
       __asm __volatile ("\
	  mova 1f,r0\n\
	  .align 2\n\
	  mov r15,r1\n\
	  mov #(0f-1f),r15\n\
       0: mov.b @%2,%0\n\
	  mov %1,r2\n\
	  add %0,r2\n\
	  mov.b r2,@%2\n\
       1: mov r1,r15"\
	: "=&r" (__result), "=&r" (__tmp) : "u" (mem), "1" (__value) \
	: "r0", "r1", "r2", "memory");		       \
     else if (sizeof (*(mem)) == 2) \
       __asm __volatile ("\
	  mova 1f,r0\n\
	  .align 2\n\
	  mov r15,r1\n\
	  mov #(0f-1f),r15\n\
       0: mov.w @%2,%0\n\
	  mov %1,r2\n\
	  add %0,r2\n\
	  mov.w r2,@%2\n\
       1: mov r1,r15"\
	: "=&r" (__result), "=&r" (__tmp) : "u" (mem), "1" (__value) \
	: "r0", "r1", "r2", "memory"); \
     else if (sizeof (*(mem)) == 4) \
       __asm __volatile ("\
	  mova 1f,r0\n\
	  .align 2\n\
	  mov r15,r1\n\
	  mov #(0f-1f),r15\n\
       0: mov.l @%2,%0\n\
	  mov %1,r2\n\
	  add %0,r2\n\
	  mov.l r2,@%2\n\
       1: mov r1,r15"\
	: "=&r" (__result), "=&r" (__tmp) : "u" (mem), "1" (__value) \
	: "r0", "r1", "r2", "memory"); \
     else \
       { \
	 __typeof (mem) memp = (mem); \
	 do \
	   __result = *memp; \
	 while (__arch_compare_and_exchange_val_64_acq \
		 (memp,	__result + __value, __result) == __result); \
	 (void) __value; \
       } \
     __result; })

#define atomic_add(mem, value) \
  (void) ({ __typeof (*(mem)) __tmp, __value = (value); \
	    if (sizeof (*(mem)) == 1) \
	      __asm __volatile ("\
		mova 1f,r0\n\
		mov r15,r1\n\
		.align 2\n\
		mov #(0f-1f),r15\n\
	     0: mov.b @%1,r2\n\
		add %0,r2\n\
		mov.b r2,@%1\n\
	     1: mov r1,r15"\
		: "=&r" (__tmp) : "u" (mem), "0" (__value) \
		: "r0", "r1", "r2", "memory"); \
	    else if (sizeof (*(mem)) == 2) \
	      __asm __volatile ("\
		mova 1f,r0\n\
		mov r15,r1\n\
		.align 2\n\
		mov #(0f-1f),r15\n\
	     0: mov.w @%1,r2\n\
		add %0,r2\n\
		mov.w r2,@%1\n\
	     1: mov r1,r15"\
		: "=&r" (__tmp) : "u" (mem), "0" (__value) \
		: "r0", "r1", "r2", "memory"); \
	    else if (sizeof (*(mem)) == 4) \
	      __asm __volatile ("\
		mova 1f,r0\n\
		mov r15,r1\n\
		.align 2\n\
		mov #(0f-1f),r15\n\
	     0: mov.l @%1,r2\n\
		add %0,r2\n\
		mov.l r2,@%1\n\
	     1: mov r1,r15"\
		: "=&r" (__tmp) : "u" (mem), "0" (__value) \
		: "r0", "r1", "r2", "memory"); \
	    else \
	      { \
		__typeof (*(mem)) oldval; \
		__typeof (mem) memp = (mem); \
		do \
		  oldval = *memp; \
		while (__arch_compare_and_exchange_val_64_acq \
			(memp, oldval + __value, oldval) == oldval); \
		(void) __value; \
	      } \
	    })

#define atomic_add_negative(mem, value) \
  ({ unsigned char __result; \
     __typeof (*(mem)) __tmp, __value = (value); \
     if (sizeof (*(mem)) == 1) \
       __asm __volatile ("\
	  mova 1f,r0\n\
	  mov r15,r1\n\
	  .align 2\n\
	  mov #(0f-1f),r15\n\
       0: mov.b @%2,r2\n\
	  add %1,r2\n\
	  mov.b r2,@%2\n\
       1: mov r1,r15\n\
	  shal r2\n\
	  movt %0"\
	: "=r" (__result), "=&r" (__tmp) : "u" (mem), "1" (__value) \
	: "r0", "r1", "r2", "t", "memory"); \
     else if (sizeof (*(mem)) == 2) \
       __asm __volatile ("\
	  mova 1f,r0\n\
	  mov r15,r1\n\
	  .align 2\n\
	  mov #(0f-1f),r15\n\
       0: mov.w @%2,r2\n\
	  add %1,r2\n\
	  mov.w r2,@%2\n\
       1: mov r1,r15\n\
	  shal r2\n\
	  movt %0"\
	: "=r" (__result), "=&r" (__tmp) : "u" (mem), "1" (__value) \
	: "r0", "r1", "r2", "t", "memory"); \
     else if (sizeof (*(mem)) == 4) \
       __asm __volatile ("\
	  mova 1f,r0\n\
	  mov r15,r1\n\
	  .align 2\n\
	  mov #(0f-1f),r15\n\
       0: mov.l @%2,r2\n\
	  add %1,r2\n\
	  mov.l r2,@%2\n\
       1: mov r1,r15\n\
	  shal r2\n\
	  movt %0"\
	: "=r" (__result), "=&r" (__tmp) : "u" (mem), "1" (__value) \
	: "r0", "r1", "r2", "t", "memory"); \
     else \
       abort (); \
     __result; })

#define atomic_add_zero(mem, value) \
  ({ unsigned char __result; \
     __typeof (*(mem)) __tmp, __value = (value); \
     if (sizeof (*(mem)) == 1) \
       __asm __volatile ("\
	  mova 1f,r0\n\
	  mov r15,r1\n\
	  .align 2\n\
	  mov #(0f-1f),r15\n\
       0: mov.b @%2,r2\n\
	  add %1,r2\n\
	  mov.b r2,@%2\n\
       1: mov r1,r15\n\
	  tst r2,r2\n\
	  movt %0"\
	: "=r" (__result), "=&r" (__tmp) : "u" (mem), "1" (__value) \
	: "r0", "r1", "r2", "t", "memory"); \
     else if (sizeof (*(mem)) == 2) \
       __asm __volatile ("\
	  mova 1f,r0\n\
	  mov r15,r1\n\
	  .align 2\n\
	  mov #(0f-1f),r15\n\
       0: mov.w @%2,r2\n\
	  add %1,r2\n\
	  mov.w r2,@%2\n\
       1: mov r1,r15\n\
	  tst r2,r2\n\
	  movt %0"\
	: "=r" (__result), "=&r" (__tmp) : "u" (mem), "1" (__value) \
	: "r0", "r1", "r2", "t", "memory"); \
     else if (sizeof (*(mem)) == 4) \
       __asm __volatile ("\
	  mova 1f,r0\n\
	  mov r15,r1\n\
	  .align 2\n\
	  mov #(0f-1f),r15\n\
       0: mov.l @%2,r2\n\
	  add %1,r2\n\
	  mov.l r2,@%2\n\
       1: mov r1,r15\n\
	  tst r2,r2\n\
	  movt %0"\
	: "=r" (__result), "=&r" (__tmp) : "u" (mem), "1" (__value) \
	: "r0", "r1", "r2", "t", "memory"); \
     else \
       abort (); \
     __result; })

#define atomic_increment_and_test(mem) atomic_add_zero((mem), 1)
#define atomic_decrement_and_test(mem) atomic_add_zero((mem), -1)

#define atomic_bit_set(mem, bit) \
  (void) ({ unsigned int __mask = 1 << (bit); \
	    if (sizeof (*(mem)) == 1) \
	      __asm __volatile ("\
		mova 1f,r0\n\
		mov r15,r1\n\
		.align 2\n\
		mov #(0f-1f),r15\n\
	     0: mov.b @%0,r2\n\
		or %1,r2\n\
		mov.b r2,@%0\n\
	     1: mov r1,r15"\
		: : "u" (mem), "u" (__mask) \
		: "r0", "r1", "r2", "memory"); \
	    else if (sizeof (*(mem)) == 2) \
	      __asm __volatile ("\
		mova 1f,r0\n\
		mov r15,r1\n\
		.align 2\n\
		mov #(0f-1f),r15\n\
	     0: mov.w @%0,r2\n\
		or %1,r2\n\
		mov.w r2,@%0\n\
	     1: mov r1,r15"\
		: : "u" (mem), "u" (__mask) \
		: "r0", "r1", "r2", "memory"); \
	    else if (sizeof (*(mem)) == 4) \
	      __asm __volatile ("\
		mova 1f,r0\n\
		mov r15,r1\n\
		.align 2\n\
		mov #(0f-1f),r15\n\
	     0: mov.l @%0,r2\n\
		or %1,r2\n\
		mov.l r2,@%0\n\
	     1: mov r1,r15"\
		: : "u" (mem), "u" (__mask) \
		: "r0", "r1", "r2", "memory"); \
	    else \
	      abort (); \
	    })

#define atomic_bit_test_set(mem, bit) \
  ({ unsigned int __mask = 1 << (bit); \
     unsigned int __result = __mask; \
     if (sizeof (*(mem)) == 1) \
       __asm __volatile ("\
	  mova 1f,r0\n\
	  .align 2\n\
	  mov r15,r1\n\
	  mov #(0f-1f),r15\n\
       0: mov.b @%2,r2\n\
	  mov r2,r3\n\
	  or %1,r2\n\
	  mov.b r2,@%2\n\
       1: mov r1,r15\n\
	  and r3,%0"\
	: "=&r" (__result), "=&r" (__mask) \
	: "u" (mem), "0" (__result), "1" (__mask) \
	: "r0", "r1", "r2", "r3", "memory");	\
     else if (sizeof (*(mem)) == 2) \
       __asm __volatile ("\
	  mova 1f,r0\n\
	  .align 2\n\
	  mov r15,r1\n\
	  mov #(0f-1f),r15\n\
       0: mov.w @%2,r2\n\
	  mov r2,r3\n\
	  or %1,r2\n\
	  mov.w %1,@%2\n\
       1: mov r1,r15\n\
	  and r3,%0"\
	: "=&r" (__result), "=&r" (__mask) \
	: "u" (mem), "0" (__result), "1" (__mask) \
	: "r0", "r1", "r2", "r3", "memory"); \
     else if (sizeof (*(mem)) == 4) \
       __asm __volatile ("\
	  mova 1f,r0\n\
	  .align 2\n\
	  mov r15,r1\n\
	  mov #(0f-1f),r15\n\
       0: mov.l @%2,r2\n\
	  mov r2,r3\n\
	  or r2,%1\n\
	  mov.l %1,@%2\n\
       1: mov r1,r15\n\
	  and r3,%0"\
	: "=&r" (__result), "=&r" (__mask) \
	: "u" (mem), "0" (__result), "1" (__mask) \
	: "r0", "r1", "r2", "r3", "memory"); \
     else \
       abort (); \
     __result; })
