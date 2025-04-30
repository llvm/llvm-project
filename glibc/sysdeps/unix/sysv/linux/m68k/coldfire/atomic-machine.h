/* Copyright (C) 2010-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Maxim Kuvyrkov <maxim@codesourcery.com>, 2010.

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
#include <sysdep.h>

/* Coldfire has no atomic compare-and-exchange operation, but the
   kernel provides userspace atomicity operations.  Use them.  */

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

/* XXX Is this actually correct?  */
#define ATOMIC_EXCHANGE_USES_CAS 1

/* The only basic operation needed is compare and exchange.  */
#define atomic_compare_and_exchange_val_acq(mem, newval, oldval)	\
  ({									\
    /* Use temporary variables to workaround call-clobberness of 	\
       the registers.  */						\
    __typeof (mem) _mem = mem;						\
    __typeof (oldval) _oldval = oldval;					\
    __typeof (newval) _newval = newval;					\
    register uint32_t _d0 asm ("d0") = SYS_ify (atomic_cmpxchg_32);	\
    register uint32_t *_a0 asm ("a0") = (uint32_t *) _mem;		\
    register uint32_t _d2 asm ("d2") = (uint32_t) _oldval;		\
    register uint32_t _d1 asm ("d1") = (uint32_t) _newval;		\
									\
    asm ("trap #0"							\
	 : "+d" (_d0), "+m" (*_a0)					\
	 : "a" (_a0), "d" (_d2), "d" (_d1));				\
    (__typeof (oldval)) _d0;						\
  })

# define atomic_full_barrier()				\
  (INTERNAL_SYSCALL_CALL (atomic_barrier), (void) 0)

#endif
