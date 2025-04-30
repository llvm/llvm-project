/* Atomic operations.  C-SKY version.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#ifndef __CSKY_ATOMIC_H_
#define __CSKY_ATOMIC_H_

#include <stdint.h>

typedef int32_t atomic32_t;
typedef uint32_t uatomic32_t;

typedef intptr_t atomicptr_t;
typedef uintptr_t uatomicptr_t;
typedef intmax_t atomic_max_t;
typedef uintmax_t uatomic_max_t;

#define __HAVE_64B_ATOMICS 0
#define USE_ATOMIC_COMPILER_BUILTINS 1
#define ATOMIC_EXCHANGE_USES_CAS 1

#define __arch_compare_and_exchange_bool_8_int(mem, newval, oldval, model) \
  (abort (), 0)

#define __arch_compare_and_exchange_bool_16_int(mem, newval, oldval, model) \
  (abort (), 0)

#define __arch_compare_and_exchange_bool_32_int(mem, newval, oldval, model) \
  ({                                                                    \
    typeof (*mem) __oldval = (oldval);                                  \
    !__atomic_compare_exchange_n (mem, (void *) &__oldval, newval, 0,   \
                                  model, __ATOMIC_RELAXED);             \
  })

#define __arch_compare_and_exchange_bool_64_int(mem, newval, oldval, model) \
  (abort (), 0)

#define __arch_compare_and_exchange_val_8_int(mem, newval, oldval, model) \
  (abort (), (__typeof (*mem)) 0)

#define __arch_compare_and_exchange_val_16_int(mem, newval, oldval, model) \
  (abort (), (__typeof (*mem)) 0)

#define __arch_compare_and_exchange_val_32_int(mem, newval, oldval, model) \
  ({                                                                    \
    typeof (*mem) __oldval = (oldval);                                  \
    __atomic_compare_exchange_n (mem, (void *) &__oldval, newval, 0,    \
                                 model, __ATOMIC_RELAXED);              \
    __oldval;                                                           \
  })

#define __arch_compare_and_exchange_val_64_int(mem, newval, oldval, model) \
  (abort (), (__typeof (*mem)) 0)

#define atomic_compare_and_exchange_bool_acq(mem, new, old)		\
  __atomic_bool_bysize (__arch_compare_and_exchange_bool, int,		\
			mem, new, old, __ATOMIC_ACQUIRE)

#define atomic_compare_and_exchange_val_acq(mem, new, old)		\
  __atomic_val_bysize (__arch_compare_and_exchange_val, int,		\
		       mem, new, old, __ATOMIC_ACQUIRE)

#endif /* atomic-machine.h */
