/* Atomic operations.  Pure ARM version.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

typedef int32_t atomic32_t;
typedef uint32_t uatomic32_t;
typedef int_fast32_t atomic_fast32_t;
typedef uint_fast32_t uatomic_fast32_t;

typedef intptr_t atomicptr_t;
typedef uintptr_t uatomicptr_t;
typedef intmax_t atomic_max_t;
typedef uintmax_t uatomic_max_t;

#define __HAVE_64B_ATOMICS 0
#define USE_ATOMIC_COMPILER_BUILTINS 0
#define ATOMIC_EXCHANGE_USES_CAS 1

void __arm_link_error (void);

#ifdef __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4
# define atomic_full_barrier() __sync_synchronize ()
#else
# define atomic_full_barrier() __arm_assisted_full_barrier ()
#endif

/* An OS-specific atomic-machine.h file will define this macro if
   the OS can provide something.  If not, we'll fail to build
   with a compiler that doesn't supply the operation.  */
#ifndef __arm_assisted_full_barrier
# define __arm_assisted_full_barrier()  __arm_link_error()
#endif

/* Use the atomic builtins provided by GCC in case the backend provides
   a pattern to do this efficiently.  */
#ifdef __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4

# define atomic_exchange_acq(mem, value)                                \
  __atomic_val_bysize (__arch_exchange, int, mem, value, __ATOMIC_ACQUIRE)

# define atomic_exchange_rel(mem, value)                                \
  __atomic_val_bysize (__arch_exchange, int, mem, value, __ATOMIC_RELEASE)

/* Atomic exchange (without compare).  */

# define __arch_exchange_8_int(mem, newval, model)      \
  (__arm_link_error (), (typeof (*mem)) 0)

# define __arch_exchange_16_int(mem, newval, model)     \
  (__arm_link_error (), (typeof (*mem)) 0)

# define __arch_exchange_32_int(mem, newval, model)     \
  __atomic_exchange_n (mem, newval, model)

# define __arch_exchange_64_int(mem, newval, model)     \
  (__arm_link_error (), (typeof (*mem)) 0)

/* Compare and exchange with "acquire" semantics, ie barrier after.  */

# define atomic_compare_and_exchange_bool_acq(mem, new, old)    \
  __atomic_bool_bysize (__arch_compare_and_exchange_bool, int,  \
                        mem, new, old, __ATOMIC_ACQUIRE)

# define atomic_compare_and_exchange_val_acq(mem, new, old)     \
  __atomic_val_bysize (__arch_compare_and_exchange_val, int,    \
                       mem, new, old, __ATOMIC_ACQUIRE)

/* Compare and exchange with "release" semantics, ie barrier before.  */

# define atomic_compare_and_exchange_val_rel(mem, new, old)      \
  __atomic_val_bysize (__arch_compare_and_exchange_val, int,    \
                       mem, new, old, __ATOMIC_RELEASE)

/* Compare and exchange.
   For all "bool" routines, we return FALSE if exchange succesful.  */

# define __arch_compare_and_exchange_bool_8_int(mem, newval, oldval, model) \
  ({__arm_link_error (); 0; })

# define __arch_compare_and_exchange_bool_16_int(mem, newval, oldval, model) \
  ({__arm_link_error (); 0; })

# define __arch_compare_and_exchange_bool_32_int(mem, newval, oldval, model) \
  ({                                                                    \
    typeof (*mem) __oldval = (oldval);                                  \
    !__atomic_compare_exchange_n (mem, (void *) &__oldval, newval, 0,   \
                                  model, __ATOMIC_RELAXED);             \
  })

# define __arch_compare_and_exchange_bool_64_int(mem, newval, oldval, model) \
  ({__arm_link_error (); 0; })

# define __arch_compare_and_exchange_val_8_int(mem, newval, oldval, model) \
  ({__arm_link_error (); oldval; })

# define __arch_compare_and_exchange_val_16_int(mem, newval, oldval, model) \
  ({__arm_link_error (); oldval; })

# define __arch_compare_and_exchange_val_32_int(mem, newval, oldval, model) \
  ({                                                                    \
    typeof (*mem) __oldval = (oldval);                                  \
    __atomic_compare_exchange_n (mem, (void *) &__oldval, newval, 0,    \
                                 model, __ATOMIC_RELAXED);              \
    __oldval;                                                           \
  })

# define __arch_compare_and_exchange_val_64_int(mem, newval, oldval, model) \
  ({__arm_link_error (); oldval; })

#else
# define __arch_compare_and_exchange_val_32_acq(mem, newval, oldval) \
  __arm_assisted_compare_and_exchange_val_32_acq ((mem), (newval), (oldval))
#endif

#ifndef __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4
/* We don't support atomic operations on any non-word types.
   So make them link errors.  */
# define __arch_compare_and_exchange_val_8_acq(mem, newval, oldval) \
  ({ __arm_link_error (); oldval; })

# define __arch_compare_and_exchange_val_16_acq(mem, newval, oldval) \
  ({ __arm_link_error (); oldval; })

# define __arch_compare_and_exchange_val_64_acq(mem, newval, oldval) \
  ({ __arm_link_error (); oldval; })
#endif

/* An OS-specific atomic-machine.h file will define this macro if
   the OS can provide something.  If not, we'll fail to build
   with a compiler that doesn't supply the operation.  */
#ifndef __arm_assisted_compare_and_exchange_val_32_acq
# define __arm_assisted_compare_and_exchange_val_32_acq(mem, newval, oldval) \
  ({ __arm_link_error (); oldval; })
#endif
