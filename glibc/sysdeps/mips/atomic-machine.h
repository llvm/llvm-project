/* Low-level functions for atomic operations. Mips version.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
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

#ifndef _MIPS_ATOMIC_MACHINE_H
#define _MIPS_ATOMIC_MACHINE_H 1

#include <stdint.h>
#include <inttypes.h>
#include <sgidefs.h>

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

#if _MIPS_SIM == _ABIO32 && __mips < 2
#define MIPS_PUSH_MIPS2 ".set	mips2\n\t"
#else
#define MIPS_PUSH_MIPS2
#endif

#if _MIPS_SIM == _ABIO32 || _MIPS_SIM == _ABIN32
#define __HAVE_64B_ATOMICS 0
#else
#define __HAVE_64B_ATOMICS 1
#endif

/* See the comments in <sys/asm.h> about the use of the sync instruction.  */
#ifndef MIPS_SYNC
# define MIPS_SYNC	sync
#endif

#define MIPS_SYNC_STR_2(X) #X
#define MIPS_SYNC_STR_1(X) MIPS_SYNC_STR_2(X)
#define MIPS_SYNC_STR MIPS_SYNC_STR_1(MIPS_SYNC)

#define USE_ATOMIC_COMPILER_BUILTINS 1

/* MIPS is an LL/SC machine.  However, XLP has a direct atomic exchange
   instruction which will be used by __atomic_exchange_n.  */
#ifdef _MIPS_ARCH_XLP
# define ATOMIC_EXCHANGE_USES_CAS 0
#else
# define ATOMIC_EXCHANGE_USES_CAS 1
#endif

/* Compare and exchange.
   For all "bool" routines, we return FALSE if exchange succesful.  */

#define __arch_compare_and_exchange_bool_8_int(mem, newval, oldval, model) \
  (abort (), 0)

#define __arch_compare_and_exchange_bool_16_int(mem, newval, oldval, model) \
  (abort (), 0)

#define __arch_compare_and_exchange_bool_32_int(mem, newval, oldval, model) \
  ({									\
    typeof (*mem) __oldval = (oldval);					\
    !__atomic_compare_exchange_n (mem, (void *) &__oldval, newval, 0,	\
				  model, __ATOMIC_RELAXED);		\
  })

#define __arch_compare_and_exchange_val_8_int(mem, newval, oldval, model) \
  (abort (), (typeof(*mem)) 0)

#define __arch_compare_and_exchange_val_16_int(mem, newval, oldval, model) \
  (abort (), (typeof(*mem)) 0)

#define __arch_compare_and_exchange_val_32_int(mem, newval, oldval, model) \
  ({									\
    typeof (*mem) __oldval = (oldval);					\
    __atomic_compare_exchange_n (mem, (void *) &__oldval, newval, 0,	\
				 model, __ATOMIC_RELAXED);		\
    __oldval;								\
  })

#if _MIPS_SIM == _ABIO32
  /* We can't do an atomic 64-bit operation in O32.  */
# define __arch_compare_and_exchange_bool_64_int(mem, newval, oldval, model) \
  (abort (), 0)
# define __arch_compare_and_exchange_val_64_int(mem, newval, oldval, model) \
  (abort (), (typeof(*mem)) 0)
#else
# define __arch_compare_and_exchange_bool_64_int(mem, newval, oldval, model) \
  __arch_compare_and_exchange_bool_32_int (mem, newval, oldval, model)
# define __arch_compare_and_exchange_val_64_int(mem, newval, oldval, model) \
  __arch_compare_and_exchange_val_32_int (mem, newval, oldval, model)
#endif

/* Compare and exchange with "acquire" semantics, ie barrier after.  */

#define atomic_compare_and_exchange_bool_acq(mem, new, old)	\
  __atomic_bool_bysize (__arch_compare_and_exchange_bool, int,	\
			mem, new, old, __ATOMIC_ACQUIRE)

#define atomic_compare_and_exchange_val_acq(mem, new, old)	\
  __atomic_val_bysize (__arch_compare_and_exchange_val, int,	\
		       mem, new, old, __ATOMIC_ACQUIRE)

/* Compare and exchange with "release" semantics, ie barrier before.  */

#define atomic_compare_and_exchange_val_rel(mem, new, old)	 \
  __atomic_val_bysize (__arch_compare_and_exchange_val, int,    \
                       mem, new, old, __ATOMIC_RELEASE)


/* Atomic exchange (without compare).  */

#define __arch_exchange_8_int(mem, newval, model)	\
  (abort (), (typeof(*mem)) 0)

#define __arch_exchange_16_int(mem, newval, model)	\
  (abort (), (typeof(*mem)) 0)

#define __arch_exchange_32_int(mem, newval, model)	\
  __atomic_exchange_n (mem, newval, model)

#if _MIPS_SIM == _ABIO32
/* We can't do an atomic 64-bit operation in O32.  */
# define __arch_exchange_64_int(mem, newval, model)	\
  (abort (), (typeof(*mem)) 0)
#else
# define __arch_exchange_64_int(mem, newval, model)	\
  __atomic_exchange_n (mem, newval, model)
#endif

#define atomic_exchange_acq(mem, value)				\
  __atomic_val_bysize (__arch_exchange, int, mem, value, __ATOMIC_ACQUIRE)

#define atomic_exchange_rel(mem, value)				\
  __atomic_val_bysize (__arch_exchange, int, mem, value, __ATOMIC_RELEASE)


/* Atomically add value and return the previous (unincremented) value.  */

#define __arch_exchange_and_add_8_int(mem, value, model)	\
  (abort (), (typeof(*mem)) 0)

#define __arch_exchange_and_add_16_int(mem, value, model)	\
  (abort (), (typeof(*mem)) 0)

#define __arch_exchange_and_add_32_int(mem, value, model)	\
  __atomic_fetch_add (mem, value, model)

#if _MIPS_SIM == _ABIO32
/* We can't do an atomic 64-bit operation in O32.  */
# define __arch_exchange_and_add_64_int(mem, value, model)	\
  (abort (), (typeof(*mem)) 0)
#else
# define __arch_exchange_and_add_64_int(mem, value, model)	\
  __atomic_fetch_add (mem, value, model)
#endif

#define atomic_exchange_and_add_acq(mem, value)			\
  __atomic_val_bysize (__arch_exchange_and_add, int, mem, value,	\
		       __ATOMIC_ACQUIRE)

#define atomic_exchange_and_add_rel(mem, value)			\
  __atomic_val_bysize (__arch_exchange_and_add, int, mem, value,	\
		       __ATOMIC_RELEASE)

/* TODO: More atomic operations could be implemented efficiently; only the
   basic requirements are done.  */

#ifdef __mips16
# define atomic_full_barrier() __sync_synchronize ()

#else /* !__mips16 */
# define atomic_full_barrier() \
  __asm__ __volatile__ (".set push\n\t"					      \
			MIPS_PUSH_MIPS2					      \
			MIPS_SYNC_STR "\n\t"				      \
			".set pop" : : : "memory")
#endif /* !__mips16 */

#endif /* atomic-machine.h */
