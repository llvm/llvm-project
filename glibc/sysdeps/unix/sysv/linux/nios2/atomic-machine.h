/* Low-level functions for atomic operations. Nios II version.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

#ifndef _NIOS2_ATOMIC_MACHINE_H
#define _NIOS2_ATOMIC_MACHINE_H 1

#include <stdint.h>

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

#define __arch_compare_and_exchange_val_8_acq(mem, newval, oldval)	\
  (abort (), (__typeof (*mem)) 0)
#define __arch_compare_and_exchange_val_16_acq(mem, newval, oldval)	\
  (abort (), (__typeof (*mem)) 0)
#define __arch_compare_and_exchange_val_64_acq(mem, newval, oldval)	\
  (abort (), (__typeof (*mem)) 0)

#define __arch_compare_and_exchange_bool_8_acq(mem, newval, oldval)	\
  (abort (), 0)
#define __arch_compare_and_exchange_bool_16_acq(mem, newval, oldval)	\
  (abort (), 0)
#define __arch_compare_and_exchange_bool_64_acq(mem, newval, oldval)	\
  (abort (), 0)

#define __arch_compare_and_exchange_val_32_acq(mem, newval, oldval)	\
  ({									\
     register int r2 asm ("r2");					\
     register int* r4 asm ("r4") = (int*)(mem);				\
     register int r5 asm ("r5");					\
     register int r6 asm ("r6") = (int)(newval);			\
     int retval, orig_oldval = (int)(oldval);				\
     long kernel_cmpxchg = 0x1004;					\
     while (1)								\
       {								\
         r5 = *r4;							\
	 if (r5 != orig_oldval)						\
	   {								\
	     retval = r5;						\
	     break;							\
	   }								\
	 asm volatile ("callr %1\n"					\
		       : "=r" (r2)					\
		       : "r" (kernel_cmpxchg), "r" (r4), "r" (r5), "r" (r6) \
		       : "ra", "memory");				\
	 if (!r2) { retval = orig_oldval; break; }			\
       }								\
     (__typeof (*(mem))) retval;					\
  })

#define __arch_compare_and_exchange_bool_32_acq(mem, newval, oldval)	\
  ({									\
     register int r2 asm ("r2");					\
     register int *r4 asm ("r4") = (int*)(mem);				\
     register int r5 asm ("r5") = (int)(oldval);			\
     register int r6 asm ("r6") = (int)(newval);			\
     long kernel_cmpxchg = 0x1004;					\
     asm volatile ("callr %1\n"						\
		   : "=r" (r2)						\
		   : "r" (kernel_cmpxchg), "r" (r4), "r" (r5), "r" (r6) \
		   : "ra", "memory");					\
     r2;								\
  })

#define atomic_full_barrier()  ({ asm volatile ("sync"); })

#endif /* _NIOS2_ATOMIC_MACHINE_H */
