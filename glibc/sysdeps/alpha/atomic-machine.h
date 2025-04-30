/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
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

#define __HAVE_64B_ATOMICS 1
#define USE_ATOMIC_COMPILER_BUILTINS 0

/* XXX Is this actually correct?  */
#define ATOMIC_EXCHANGE_USES_CAS 1


#define __MB		"	mb\n"


/* Compare and exchange.  For all of the "xxx" routines, we expect a
   "__prev" and a "__cmp" variable to be provided by the enclosing scope,
   in which values are returned.  */

#define __arch_compare_and_exchange_xxx_8_int(mem, new, old, mb1, mb2)	\
({									\
  unsigned long __tmp, __snew, __addr64;				\
  __asm__ __volatile__ (						\
		mb1							\
	"	andnot	%[__addr8],7,%[__addr64]\n"			\
	"	insbl	%[__new],%[__addr8],%[__snew]\n"		\
	"1:	ldq_l	%[__tmp],0(%[__addr64])\n"			\
	"	extbl	%[__tmp],%[__addr8],%[__prev]\n"		\
	"	cmpeq	%[__prev],%[__old],%[__cmp]\n"			\
	"	beq	%[__cmp],2f\n"					\
	"	mskbl	%[__tmp],%[__addr8],%[__tmp]\n"			\
	"	or	%[__snew],%[__tmp],%[__tmp]\n"			\
	"	stq_c	%[__tmp],0(%[__addr64])\n"			\
	"	beq	%[__tmp],1b\n"					\
		mb2							\
	"2:"								\
	: [__prev] "=&r" (__prev),					\
	  [__snew] "=&r" (__snew),					\
	  [__tmp] "=&r" (__tmp),					\
	  [__cmp] "=&r" (__cmp),					\
	  [__addr64] "=&r" (__addr64)					\
	: [__addr8] "r" (mem),						\
	  [__old] "Ir" ((uint64_t)(uint8_t)(uint64_t)(old)),		\
	  [__new] "r" (new)						\
	: "memory");							\
})

#define __arch_compare_and_exchange_xxx_16_int(mem, new, old, mb1, mb2) \
({									\
  unsigned long __tmp, __snew, __addr64;				\
  __asm__ __volatile__ (						\
		mb1							\
	"	andnot	%[__addr16],7,%[__addr64]\n"			\
	"	inswl	%[__new],%[__addr16],%[__snew]\n"		\
	"1:	ldq_l	%[__tmp],0(%[__addr64])\n"			\
	"	extwl	%[__tmp],%[__addr16],%[__prev]\n"		\
	"	cmpeq	%[__prev],%[__old],%[__cmp]\n"			\
	"	beq	%[__cmp],2f\n"					\
	"	mskwl	%[__tmp],%[__addr16],%[__tmp]\n"		\
	"	or	%[__snew],%[__tmp],%[__tmp]\n"			\
	"	stq_c	%[__tmp],0(%[__addr64])\n"			\
	"	beq	%[__tmp],1b\n"					\
		mb2							\
	"2:"								\
	: [__prev] "=&r" (__prev),					\
	  [__snew] "=&r" (__snew),					\
	  [__tmp] "=&r" (__tmp),					\
	  [__cmp] "=&r" (__cmp),					\
	  [__addr64] "=&r" (__addr64)					\
	: [__addr16] "r" (mem),						\
	  [__old] "Ir" ((uint64_t)(uint16_t)(uint64_t)(old)),		\
	  [__new] "r" (new)						\
	: "memory");							\
})

#define __arch_compare_and_exchange_xxx_32_int(mem, new, old, mb1, mb2) \
({									\
  __asm__ __volatile__ (						\
		mb1							\
	"1:	ldl_l	%[__prev],%[__mem]\n"				\
	"	cmpeq	%[__prev],%[__old],%[__cmp]\n"			\
	"	beq	%[__cmp],2f\n"					\
	"	mov	%[__new],%[__cmp]\n"				\
	"	stl_c	%[__cmp],%[__mem]\n"				\
	"	beq	%[__cmp],1b\n"					\
		mb2							\
	"2:"								\
	: [__prev] "=&r" (__prev),					\
	  [__cmp] "=&r" (__cmp)						\
	: [__mem] "m" (*(mem)),						\
	  [__old] "Ir" ((uint64_t)(atomic32_t)(uint64_t)(old)),		\
	  [__new] "Ir" (new)						\
	: "memory");							\
})

#define __arch_compare_and_exchange_xxx_64_int(mem, new, old, mb1, mb2) \
({									\
  __asm__ __volatile__ (						\
		mb1							\
	"1:	ldq_l	%[__prev],%[__mem]\n"				\
	"	cmpeq	%[__prev],%[__old],%[__cmp]\n"			\
	"	beq	%[__cmp],2f\n"					\
	"	mov	%[__new],%[__cmp]\n"				\
	"	stq_c	%[__cmp],%[__mem]\n"				\
	"	beq	%[__cmp],1b\n"					\
		mb2							\
	"2:"								\
	: [__prev] "=&r" (__prev),					\
	  [__cmp] "=&r" (__cmp)						\
	: [__mem] "m" (*(mem)),						\
	  [__old] "Ir" ((uint64_t)(old)),				\
	  [__new] "Ir" (new)						\
	: "memory");							\
})

/* For all "bool" routines, we return FALSE if exchange succesful.  */

#define __arch_compare_and_exchange_bool_8_int(mem, new, old, mb1, mb2)	\
({ unsigned long __prev; int __cmp;					\
   __arch_compare_and_exchange_xxx_8_int(mem, new, old, mb1, mb2);	\
   !__cmp; })

#define __arch_compare_and_exchange_bool_16_int(mem, new, old, mb1, mb2) \
({ unsigned long __prev; int __cmp;					\
   __arch_compare_and_exchange_xxx_16_int(mem, new, old, mb1, mb2);	\
   !__cmp; })

#define __arch_compare_and_exchange_bool_32_int(mem, new, old, mb1, mb2) \
({ unsigned long __prev; int __cmp;					\
   __arch_compare_and_exchange_xxx_32_int(mem, new, old, mb1, mb2);	\
   !__cmp; })

#define __arch_compare_and_exchange_bool_64_int(mem, new, old, mb1, mb2) \
({ unsigned long __prev; int __cmp;					\
   __arch_compare_and_exchange_xxx_64_int(mem, new, old, mb1, mb2);	\
   !__cmp; })

/* For all "val" routines, return the old value whether exchange
   successful or not.  */

#define __arch_compare_and_exchange_val_8_int(mem, new, old, mb1, mb2)	\
({ unsigned long __prev; int __cmp;					\
   __arch_compare_and_exchange_xxx_8_int(mem, new, old, mb1, mb2);	\
   (typeof (*mem))__prev; })

#define __arch_compare_and_exchange_val_16_int(mem, new, old, mb1, mb2) \
({ unsigned long __prev; int __cmp;					\
   __arch_compare_and_exchange_xxx_16_int(mem, new, old, mb1, mb2);	\
   (typeof (*mem))__prev; })

#define __arch_compare_and_exchange_val_32_int(mem, new, old, mb1, mb2) \
({ unsigned long __prev; int __cmp;					\
   __arch_compare_and_exchange_xxx_32_int(mem, new, old, mb1, mb2);	\
   (typeof (*mem))__prev; })

#define __arch_compare_and_exchange_val_64_int(mem, new, old, mb1, mb2) \
({ unsigned long __prev; int __cmp;					\
   __arch_compare_and_exchange_xxx_64_int(mem, new, old, mb1, mb2);	\
   (typeof (*mem))__prev; })

/* Compare and exchange with "acquire" semantics, ie barrier after.  */

#define atomic_compare_and_exchange_bool_acq(mem, new, old)	\
  __atomic_bool_bysize (__arch_compare_and_exchange_bool, int,	\
		        mem, new, old, "", __MB)

#define atomic_compare_and_exchange_val_acq(mem, new, old)	\
  __atomic_val_bysize (__arch_compare_and_exchange_val, int,	\
		       mem, new, old, "", __MB)

/* Compare and exchange with "release" semantics, ie barrier before.  */

#define atomic_compare_and_exchange_val_rel(mem, new, old)	\
  __atomic_val_bysize (__arch_compare_and_exchange_val, int,	\
		       mem, new, old, __MB, "")


/* Atomically store value and return the previous value.  */

#define __arch_exchange_8_int(mem, value, mb1, mb2)			\
({									\
  unsigned long __tmp, __addr64, __sval; __typeof(*mem) __ret;		\
  __asm__ __volatile__ (						\
		mb1							\
	"	andnot	%[__addr8],7,%[__addr64]\n"			\
	"	insbl	%[__value],%[__addr8],%[__sval]\n"		\
	"1:	ldq_l	%[__tmp],0(%[__addr64])\n"			\
	"	extbl	%[__tmp],%[__addr8],%[__ret]\n"			\
	"	mskbl	%[__tmp],%[__addr8],%[__tmp]\n"			\
	"	or	%[__sval],%[__tmp],%[__tmp]\n"			\
	"	stq_c	%[__tmp],0(%[__addr64])\n"			\
	"	beq	%[__tmp],1b\n"					\
		mb2							\
	: [__ret] "=&r" (__ret),					\
	  [__sval] "=&r" (__sval),					\
	  [__tmp] "=&r" (__tmp),					\
	  [__addr64] "=&r" (__addr64)					\
	: [__addr8] "r" (mem),						\
	  [__value] "r" (value)						\
	: "memory");							\
  __ret; })

#define __arch_exchange_16_int(mem, value, mb1, mb2)			\
({									\
  unsigned long __tmp, __addr64, __sval; __typeof(*mem) __ret;		\
  __asm__ __volatile__ (						\
		mb1							\
	"	andnot	%[__addr16],7,%[__addr64]\n"			\
	"	inswl	%[__value],%[__addr16],%[__sval]\n"		\
	"1:	ldq_l	%[__tmp],0(%[__addr64])\n"			\
	"	extwl	%[__tmp],%[__addr16],%[__ret]\n"		\
	"	mskwl	%[__tmp],%[__addr16],%[__tmp]\n"		\
	"	or	%[__sval],%[__tmp],%[__tmp]\n"			\
	"	stq_c	%[__tmp],0(%[__addr64])\n"			\
	"	beq	%[__tmp],1b\n"					\
		mb2							\
	: [__ret] "=&r" (__ret),					\
	  [__sval] "=&r" (__sval),					\
	  [__tmp] "=&r" (__tmp),					\
	  [__addr64] "=&r" (__addr64)					\
	: [__addr16] "r" (mem),						\
	  [__value] "r" (value)						\
	: "memory");							\
  __ret; })

#define __arch_exchange_32_int(mem, value, mb1, mb2)			\
({									\
  signed int __tmp; __typeof(*mem) __ret;				\
  __asm__ __volatile__ (						\
		mb1							\
	"1:	ldl_l	%[__ret],%[__mem]\n"				\
	"	mov	%[__val],%[__tmp]\n"				\
	"	stl_c	%[__tmp],%[__mem]\n"				\
	"	beq	%[__tmp],1b\n"					\
		mb2							\
	: [__ret] "=&r" (__ret),					\
	  [__tmp] "=&r" (__tmp)						\
	: [__mem] "m" (*(mem)),						\
	  [__val] "Ir" (value)						\
	: "memory");							\
  __ret; })

#define __arch_exchange_64_int(mem, value, mb1, mb2)			\
({									\
  unsigned long __tmp; __typeof(*mem) __ret;				\
  __asm__ __volatile__ (						\
		mb1							\
	"1:	ldq_l	%[__ret],%[__mem]\n"				\
	"	mov	%[__val],%[__tmp]\n"				\
	"	stq_c	%[__tmp],%[__mem]\n"				\
	"	beq	%[__tmp],1b\n"					\
		mb2							\
	: [__ret] "=&r" (__ret),					\
	  [__tmp] "=&r" (__tmp)						\
	: [__mem] "m" (*(mem)),						\
	  [__val] "Ir" (value)						\
	: "memory");							\
  __ret; })

#define atomic_exchange_acq(mem, value) \
  __atomic_val_bysize (__arch_exchange, int, mem, value, "", __MB)

#define atomic_exchange_rel(mem, value) \
  __atomic_val_bysize (__arch_exchange, int, mem, value, __MB, "")


/* Atomically add value and return the previous (unincremented) value.  */

#define __arch_exchange_and_add_8_int(mem, value, mb1, mb2) \
  ({ __builtin_trap (); 0; })

#define __arch_exchange_and_add_16_int(mem, value, mb1, mb2) \
  ({ __builtin_trap (); 0; })

#define __arch_exchange_and_add_32_int(mem, value, mb1, mb2)		\
({									\
  signed int __tmp; __typeof(*mem) __ret;				\
  __asm__ __volatile__ (						\
		mb1							\
	"1:	ldl_l	%[__ret],%[__mem]\n"				\
	"	addl	%[__ret],%[__val],%[__tmp]\n"			\
	"	stl_c	%[__tmp],%[__mem]\n"				\
	"	beq	%[__tmp],1b\n"					\
		mb2							\
	: [__ret] "=&r" (__ret),					\
	  [__tmp] "=&r" (__tmp)						\
	: [__mem] "m" (*(mem)),						\
	  [__val] "Ir" ((signed int)(value))				\
	: "memory");							\
  __ret; })

#define __arch_exchange_and_add_64_int(mem, value, mb1, mb2)		\
({									\
  unsigned long __tmp; __typeof(*mem) __ret;				\
  __asm__ __volatile__ (						\
		mb1							\
	"1:	ldq_l	%[__ret],%[__mem]\n"				\
	"	addq	%[__ret],%[__val],%[__tmp]\n"			\
	"	stq_c	%[__tmp],%[__mem]\n"				\
	"	beq	%[__tmp],1b\n"					\
		mb2							\
	: [__ret] "=&r" (__ret),					\
	  [__tmp] "=&r" (__tmp)						\
	: [__mem] "m" (*(mem)),						\
	  [__val] "Ir" ((unsigned long)(value))				\
	: "memory");							\
  __ret; })

/* ??? Barrier semantics for atomic_exchange_and_add appear to be
   undefined.  Use full barrier for now, as that's safe.  */
#define atomic_exchange_and_add(mem, value) \
  __atomic_val_bysize (__arch_exchange_and_add, int, mem, value, __MB, __MB)


/* ??? Blah, I'm lazy.  Implement these later.  Can do better than the
   compare-and-exchange loop provided by generic code.

#define atomic_decrement_if_positive(mem)
#define atomic_bit_test_set(mem, bit)

*/

#define atomic_full_barrier()	__asm ("mb" : : : "memory");
#define atomic_read_barrier()	__asm ("mb" : : : "memory");
#define atomic_write_barrier()	__asm ("wmb" : : : "memory");
