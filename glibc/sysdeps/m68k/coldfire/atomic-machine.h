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

#ifndef _ATOMIC_MACHINE_H
#define _ATOMIC_MACHINE_H	1

#include <stdint.h>

/* Coldfire has no atomic compare-and-exchange operation, and the
   kernel provides no userspace atomicity operations.  Here we just
   use generic non-atomic implementations instead.  */

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

/* If we have just non-atomic operations, we can as well make them wide.  */
#define __HAVE_64B_ATOMICS 1
#define USE_ATOMIC_COMPILER_BUILTINS 0

/* XXX Is this actually correct?  */
#define ATOMIC_EXCHANGE_USES_CAS 1

/* The only basic operation needed is compare and exchange.  */
#define atomic_compare_and_exchange_val_acq(mem, newval, oldval) \
  ({ __typeof (mem) __gmemp = (mem);				      \
     __typeof (*mem) __gret = *__gmemp;				      \
     __typeof (*mem) __gnewval = (newval);			      \
								      \
     if (__gret == (oldval))					      \
       *__gmemp = __gnewval;					      \
     __gret; })

#define atomic_compare_and_exchange_bool_acq(mem, newval, oldval) \
  ({ __typeof (mem) __gmemp = (mem);				      \
     __typeof (*mem) __gnewval = (newval);			      \
								      \
     *__gmemp == (oldval) ? (*__gmemp = __gnewval, 0) : 1; })

#endif
