/* Low-level functions for atomic operations. RISC-V version.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#ifndef _LINUX_RISCV_BITS_ATOMIC_H
#define _LINUX_RISCV_BITS_ATOMIC_H 1

#include <stdint.h>

typedef int32_t atomic32_t;
typedef uint32_t uatomic32_t;

typedef int64_t atomic64_t;
typedef uint64_t uatomic64_t;

typedef intptr_t atomicptr_t;
typedef uintptr_t uatomicptr_t;
typedef intmax_t atomic_max_t;
typedef uintmax_t uatomic_max_t;

#define atomic_full_barrier() __sync_synchronize ()

#ifdef __riscv_atomic

# define __HAVE_64B_ATOMICS (__riscv_xlen >= 64)
# define USE_ATOMIC_COMPILER_BUILTINS 1
# define ATOMIC_EXCHANGE_USES_CAS 0

/* Compare and exchange.
   For all "bool" routines, we return FALSE if exchange succesful.  */

# define __arch_compare_and_exchange_bool_8_int(mem, newval, oldval, model) \
  ({									\
    typeof (*mem) __oldval = (oldval);					\
    !__atomic_compare_exchange_n (mem, (void *) &__oldval, newval, 0,	\
				  model, __ATOMIC_RELAXED);		\
  })

# define __arch_compare_and_exchange_bool_16_int(mem, newval, oldval, model) \
  ({									\
    typeof (*mem) __oldval = (oldval);					\
    !__atomic_compare_exchange_n (mem, (void *) &__oldval, newval, 0,	\
				  model, __ATOMIC_RELAXED);		\
  })

# define __arch_compare_and_exchange_bool_32_int(mem, newval, oldval, model) \
  ({									\
    typeof (*mem) __oldval = (oldval);					\
    !__atomic_compare_exchange_n (mem, (void *) &__oldval, newval, 0,	\
				  model, __ATOMIC_RELAXED);		\
  })

#  define __arch_compare_and_exchange_bool_64_int(mem, newval, oldval, model) \
  ({									\
    typeof (*mem) __oldval = (oldval);					\
    !__atomic_compare_exchange_n (mem, (void *) &__oldval, newval, 0,	\
				  model, __ATOMIC_RELAXED);		\
  })

# define __arch_compare_and_exchange_val_8_int(mem, newval, oldval, model) \
  ({									\
    typeof (*mem) __oldval = (oldval);					\
    __atomic_compare_exchange_n (mem, (void *) &__oldval, newval, 0,	\
				 model, __ATOMIC_RELAXED);		\
    __oldval;								\
  })

# define __arch_compare_and_exchange_val_16_int(mem, newval, oldval, model) \
  ({									\
    typeof (*mem) __oldval = (oldval);					\
    __atomic_compare_exchange_n (mem, (void *) &__oldval, newval, 0,	\
				 model, __ATOMIC_RELAXED);		\
    __oldval;								\
  })

# define __arch_compare_and_exchange_val_32_int(mem, newval, oldval, model) \
  ({									\
    typeof (*mem) __oldval = (oldval);					\
    __atomic_compare_exchange_n (mem, (void *) &__oldval, newval, 0,	\
				 model, __ATOMIC_RELAXED);		\
    __oldval;								\
  })

# define __arch_compare_and_exchange_val_64_int(mem, newval, oldval, model) \
  ({									\
    typeof (*mem) __oldval = (oldval);					\
    __atomic_compare_exchange_n (mem, (void *) &__oldval, newval, 0,	\
				 model, __ATOMIC_RELAXED);		\
    __oldval;								\
  })

/* Atomic compare and exchange.  */

# define atomic_compare_and_exchange_bool_acq(mem, new, old)	\
  __atomic_bool_bysize (__arch_compare_and_exchange_bool, int,	\
			mem, new, old, __ATOMIC_ACQUIRE)

# define atomic_compare_and_exchange_val_acq(mem, new, old)	\
  __atomic_val_bysize (__arch_compare_and_exchange_val, int,	\
		       mem, new, old, __ATOMIC_ACQUIRE)

# define atomic_compare_and_exchange_val_rel(mem, new, old)	 \
  __atomic_val_bysize (__arch_compare_and_exchange_val, int,    \
                       mem, new, old, __ATOMIC_RELEASE)

/* Atomic exchange (without compare).  */

# define __arch_exchange_8_int(mem, newval, model)	\
  __atomic_exchange_n (mem, newval, model)

# define __arch_exchange_16_int(mem, newval, model)	\
  __atomic_exchange_n (mem, newval, model)

# define __arch_exchange_32_int(mem, newval, model)	\
  __atomic_exchange_n (mem, newval, model)

#  define __arch_exchange_64_int(mem, newval, model)	\
  __atomic_exchange_n (mem, newval, model)

# define atomic_exchange_acq(mem, value)				\
  __atomic_val_bysize (__arch_exchange, int, mem, value, __ATOMIC_ACQUIRE)

# define atomic_exchange_rel(mem, value)				\
  __atomic_val_bysize (__arch_exchange, int, mem, value, __ATOMIC_RELEASE)

/* Atomically add value and return the previous (unincremented) value.  */

# define __arch_exchange_and_add_8_int(mem, value, model)	\
  __atomic_fetch_add (mem, value, model)

# define __arch_exchange_and_add_16_int(mem, value, model)	\
  __atomic_fetch_add (mem, value, model)

# define __arch_exchange_and_add_32_int(mem, value, model)	\
  __atomic_fetch_add (mem, value, model)

#  define __arch_exchange_and_add_64_int(mem, value, model)	\
  __atomic_fetch_add (mem, value, model)

# define atomic_exchange_and_add_acq(mem, value)			\
  __atomic_val_bysize (__arch_exchange_and_add, int, mem, value,	\
		       __ATOMIC_ACQUIRE)

# define atomic_exchange_and_add_rel(mem, value)			\
  __atomic_val_bysize (__arch_exchange_and_add, int, mem, value,	\
		       __ATOMIC_RELEASE)

/* Miscellaneous.  */

# define asm_amo(which, ordering, mem, value) ({ 		\
  __atomic_check_size (mem);					\
  typeof (*mem) __tmp; 						\
  if (sizeof (__tmp) == 4)					\
    asm volatile (which ".w" ordering "\t%0, %z2, %1"		\
		  : "=r" (__tmp), "+A" (* (mem))		\
		  : "rJ" (value));				\
  else if (sizeof (__tmp) == 8)					\
    asm volatile (which ".d" ordering "\t%0, %z2, %1"		\
		  : "=r" (__tmp), "+A" (* (mem))		\
		  : "rJ" (value));				\
  else								\
    abort ();							\
  __tmp; })

# define atomic_max(mem, value) asm_amo ("amomaxu", ".aq", mem, value)
# define atomic_min(mem, value) asm_amo ("amominu", ".aq", mem, value)

# define atomic_bit_test_set(mem, bit)                   \
  ({ typeof (*mem) __mask = (typeof (*mem))1 << (bit);    \
     asm_amo ("amoor", ".aq", mem, __mask) & __mask; })

# define catomic_exchange_and_add(mem, value)		\
  atomic_exchange_and_add (mem, value)
# define catomic_max(mem, value) atomic_max (mem, value)

#else /* __riscv_atomic */
# error "ISAs that do not subsume the A extension are not supported"
#endif /* !__riscv_atomic */

#endif /* bits/atomic.h */
