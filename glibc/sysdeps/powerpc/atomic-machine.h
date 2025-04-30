/* Atomic operations.  PowerPC Common version.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Paul Mackerras <paulus@au.ibm.com>, 2003.

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

/*
 * Never include sysdeps/powerpc/atomic-machine.h directly.
 * Alway use include/atomic.h which will include either
 * sysdeps/powerpc/powerpc32/atomic-machine.h
 * or
 * sysdeps/powerpc/powerpc64/atomic-machine.h
 * as appropriate and which in turn include this file.
 */

#include <stdint.h>

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

/*
 * Powerpc does not have byte and halfword forms of load and reserve and
 * store conditional. So for powerpc we stub out the 8- and 16-bit forms.
 */
#define __arch_compare_and_exchange_bool_8_acq(mem, newval, oldval) \
  (abort (), 0)

#define __arch_compare_and_exchange_bool_16_acq(mem, newval, oldval) \
  (abort (), 0)

#define __ARCH_ACQ_INSTR	"isync"
#ifndef __ARCH_REL_INSTR
# define __ARCH_REL_INSTR	"sync"
#endif

#ifndef MUTEX_HINT_ACQ
# define MUTEX_HINT_ACQ
#endif
#ifndef MUTEX_HINT_REL
# define MUTEX_HINT_REL
#endif

#define atomic_full_barrier()	__asm ("sync" ::: "memory")

#define __arch_compare_and_exchange_val_32_acq(mem, newval, oldval)	      \
  ({									      \
      __typeof (*(mem)) __tmp;						      \
      __typeof (mem)  __memp = (mem);					      \
      __asm __volatile (						      \
		        "1:	lwarx	%0,0,%1" MUTEX_HINT_ACQ "\n"	      \
		        "	cmpw	%0,%2\n"			      \
		        "	bne	2f\n"				      \
		        "	stwcx.	%3,0,%1\n"			      \
		        "	bne-	1b\n"				      \
		        "2:	" __ARCH_ACQ_INSTR			      \
		        : "=&r" (__tmp)					      \
		        : "b" (__memp), "r" (oldval), "r" (newval)	      \
		        : "cr0", "memory");				      \
      __tmp;								      \
  })

#define __arch_compare_and_exchange_val_32_rel(mem, newval, oldval)	      \
  ({									      \
      __typeof (*(mem)) __tmp;						      \
      __typeof (mem)  __memp = (mem);					      \
      __asm __volatile (__ARCH_REL_INSTR "\n"				      \
		        "1:	lwarx	%0,0,%1" MUTEX_HINT_REL "\n"	      \
		        "	cmpw	%0,%2\n"			      \
		        "	bne	2f\n"				      \
		        "	stwcx.	%3,0,%1\n"			      \
		        "	bne-	1b\n"				      \
		        "2:	"					      \
		        : "=&r" (__tmp)					      \
		        : "b" (__memp), "r" (oldval), "r" (newval)	      \
		        : "cr0", "memory");				      \
      __tmp;								      \
  })

#define __arch_atomic_exchange_32_acq(mem, value)			      \
  ({									      \
    __typeof (*mem) __val;						      \
    __asm __volatile (							      \
		      "1:	lwarx	%0,0,%2" MUTEX_HINT_ACQ "\n"	      \
		      "		stwcx.	%3,0,%2\n"			      \
		      "		bne-	1b\n"				      \
		      "   " __ARCH_ACQ_INSTR				      \
		      : "=&r" (__val), "=m" (*mem)			      \
		      : "b" (mem), "r" (value), "m" (*mem)		      \
		      : "cr0", "memory");				      \
    __val;								      \
  })

#define __arch_atomic_exchange_32_rel(mem, value) \
  ({									      \
    __typeof (*mem) __val;						      \
    __asm __volatile (__ARCH_REL_INSTR "\n"				      \
		      "1:	lwarx	%0,0,%2" MUTEX_HINT_REL "\n"	      \
		      "		stwcx.	%3,0,%2\n"			      \
		      "		bne-	1b"				      \
		      : "=&r" (__val), "=m" (*mem)			      \
		      : "b" (mem), "r" (value), "m" (*mem)		      \
		      : "cr0", "memory");				      \
    __val;								      \
  })

#define __arch_atomic_exchange_and_add_32(mem, value) \
  ({									      \
    __typeof (*mem) __val, __tmp;					      \
    __asm __volatile ("1:	lwarx	%0,0,%3\n"			      \
		      "		add	%1,%0,%4\n"			      \
		      "		stwcx.	%1,0,%3\n"			      \
		      "		bne-	1b"				      \
		      : "=&b" (__val), "=&r" (__tmp), "=m" (*mem)	      \
		      : "b" (mem), "r" (value), "m" (*mem)		      \
		      : "cr0", "memory");				      \
    __val;								      \
  })

#define __arch_atomic_exchange_and_add_32_acq(mem, value) \
  ({									      \
    __typeof (*mem) __val, __tmp;					      \
    __asm __volatile ("1:	lwarx	%0,0,%3" MUTEX_HINT_ACQ "\n"	      \
		      "		add	%1,%0,%4\n"			      \
		      "		stwcx.	%1,0,%3\n"			      \
		      "		bne-	1b\n"				      \
		      __ARCH_ACQ_INSTR					      \
		      : "=&b" (__val), "=&r" (__tmp), "=m" (*mem)	      \
		      : "b" (mem), "r" (value), "m" (*mem)		      \
		      : "cr0", "memory");				      \
    __val;								      \
  })

#define __arch_atomic_exchange_and_add_32_rel(mem, value) \
  ({									      \
    __typeof (*mem) __val, __tmp;					      \
    __asm __volatile (__ARCH_REL_INSTR "\n"				      \
		      "1:	lwarx	%0,0,%3" MUTEX_HINT_REL "\n"	      \
		      "		add	%1,%0,%4\n"			      \
		      "		stwcx.	%1,0,%3\n"			      \
		      "		bne-	1b"				      \
		      : "=&b" (__val), "=&r" (__tmp), "=m" (*mem)	      \
		      : "b" (mem), "r" (value), "m" (*mem)		      \
		      : "cr0", "memory");				      \
    __val;								      \
  })

#define __arch_atomic_increment_val_32(mem) \
  ({									      \
    __typeof (*(mem)) __val;						      \
    __asm __volatile ("1:	lwarx	%0,0,%2\n"			      \
		      "		addi	%0,%0,1\n"			      \
		      "		stwcx.	%0,0,%2\n"			      \
		      "		bne-	1b"				      \
		      : "=&b" (__val), "=m" (*mem)			      \
		      : "b" (mem), "m" (*mem)				      \
		      : "cr0", "memory");				      \
    __val;								      \
  })

#define __arch_atomic_decrement_val_32(mem) \
  ({									      \
    __typeof (*(mem)) __val;						      \
    __asm __volatile ("1:	lwarx	%0,0,%2\n"			      \
		      "		subi	%0,%0,1\n"			      \
		      "		stwcx.	%0,0,%2\n"			      \
		      "		bne-	1b"				      \
		      : "=&b" (__val), "=m" (*mem)			      \
		      : "b" (mem), "m" (*mem)				      \
		      : "cr0", "memory");				      \
    __val;								      \
  })

#define __arch_atomic_decrement_if_positive_32(mem) \
  ({ int __val, __tmp;							      \
     __asm __volatile ("1:	lwarx	%0,0,%3\n"			      \
		       "	cmpwi	0,%0,0\n"			      \
		       "	addi	%1,%0,-1\n"			      \
		       "	ble	2f\n"				      \
		       "	stwcx.	%1,0,%3\n"			      \
		       "	bne-	1b\n"				      \
		       "2:	" __ARCH_ACQ_INSTR			      \
		       : "=&b" (__val), "=&r" (__tmp), "=m" (*mem)	      \
		       : "b" (mem), "m" (*mem)				      \
		       : "cr0", "memory");				      \
     __val;								      \
  })

#define atomic_compare_and_exchange_val_acq(mem, newval, oldval) \
  ({									      \
    __typeof (*(mem)) __result;						      \
    if (sizeof (*mem) == 4)						      \
      __result = __arch_compare_and_exchange_val_32_acq(mem, newval, oldval); \
    else if (sizeof (*mem) == 8)					      \
      __result = __arch_compare_and_exchange_val_64_acq(mem, newval, oldval); \
    else 								      \
       abort ();							      \
    __result;								      \
  })

#define atomic_compare_and_exchange_val_rel(mem, newval, oldval) \
  ({									      \
    __typeof (*(mem)) __result;						      \
    if (sizeof (*mem) == 4)						      \
      __result = __arch_compare_and_exchange_val_32_rel(mem, newval, oldval); \
    else if (sizeof (*mem) == 8)					      \
      __result = __arch_compare_and_exchange_val_64_rel(mem, newval, oldval); \
    else 								      \
       abort ();							      \
    __result;								      \
  })

#define atomic_exchange_acq(mem, value) \
  ({									      \
    __typeof (*(mem)) __result;						      \
    if (sizeof (*mem) == 4)						      \
      __result = __arch_atomic_exchange_32_acq (mem, value);		      \
    else if (sizeof (*mem) == 8)					      \
      __result = __arch_atomic_exchange_64_acq (mem, value);		      \
    else 								      \
       abort ();							      \
    __result;								      \
  })

#define atomic_exchange_rel(mem, value) \
  ({									      \
    __typeof (*(mem)) __result;						      \
    if (sizeof (*mem) == 4)						      \
      __result = __arch_atomic_exchange_32_rel (mem, value);		      \
    else if (sizeof (*mem) == 8)					      \
      __result = __arch_atomic_exchange_64_rel (mem, value);		      \
    else 								      \
       abort ();							      \
    __result;								      \
  })

#define atomic_exchange_and_add(mem, value) \
  ({									      \
    __typeof (*(mem)) __result;						      \
    if (sizeof (*mem) == 4)						      \
      __result = __arch_atomic_exchange_and_add_32 (mem, value);	      \
    else if (sizeof (*mem) == 8)					      \
      __result = __arch_atomic_exchange_and_add_64 (mem, value);	      \
    else 								      \
       abort ();							      \
    __result;								      \
  })
#define atomic_exchange_and_add_acq(mem, value) \
  ({									      \
    __typeof (*(mem)) __result;						      \
    if (sizeof (*mem) == 4)						      \
      __result = __arch_atomic_exchange_and_add_32_acq (mem, value);	      \
    else if (sizeof (*mem) == 8)					      \
      __result = __arch_atomic_exchange_and_add_64_acq (mem, value);	      \
    else 								      \
       abort ();							      \
    __result;								      \
  })
#define atomic_exchange_and_add_rel(mem, value) \
  ({									      \
    __typeof (*(mem)) __result;						      \
    if (sizeof (*mem) == 4)						      \
      __result = __arch_atomic_exchange_and_add_32_rel (mem, value);	      \
    else if (sizeof (*mem) == 8)					      \
      __result = __arch_atomic_exchange_and_add_64_rel (mem, value);	      \
    else 								      \
       abort ();							      \
    __result;								      \
  })

#define atomic_increment_val(mem) \
  ({									      \
    __typeof (*(mem)) __result;						      \
    if (sizeof (*(mem)) == 4)						      \
      __result = __arch_atomic_increment_val_32 (mem);			      \
    else if (sizeof (*(mem)) == 8)					      \
      __result = __arch_atomic_increment_val_64 (mem);			      \
    else 								      \
       abort ();							      \
    __result;								      \
  })

#define atomic_increment(mem) ({ atomic_increment_val (mem); (void) 0; })

#define atomic_decrement_val(mem) \
  ({									      \
    __typeof (*(mem)) __result;						      \
    if (sizeof (*(mem)) == 4)						      \
      __result = __arch_atomic_decrement_val_32 (mem);			      \
    else if (sizeof (*(mem)) == 8)					      \
      __result = __arch_atomic_decrement_val_64 (mem);			      \
    else 								      \
       abort ();							      \
    __result;								      \
  })

#define atomic_decrement(mem) ({ atomic_decrement_val (mem); (void) 0; })


/* Decrement *MEM if it is > 0, and return the old value.  */
#define atomic_decrement_if_positive(mem) \
  ({ __typeof (*(mem)) __result;					      \
    if (sizeof (*mem) == 4)						      \
      __result = __arch_atomic_decrement_if_positive_32 (mem);		      \
    else if (sizeof (*mem) == 8)					      \
      __result = __arch_atomic_decrement_if_positive_64 (mem);		      \
    else								      \
       abort ();							      \
    __result;								      \
  })
