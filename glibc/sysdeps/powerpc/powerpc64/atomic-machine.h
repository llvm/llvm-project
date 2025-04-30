/* Atomic operations.  PowerPC64 version.
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

/*  POWER6 adds a "Mutex Hint" to the Load and Reserve instruction.
    This is a hint to the hardware to expect additional updates adjacent
    to the lock word or not.  If we are acquiring a Mutex, the hint
    should be true. Otherwise we releasing a Mutex or doing a simple
    atomic operation.  In that case we don't expect additional updates
    adjacent to the lock word after the Store Conditional and the hint
    should be false.  */

#if defined _ARCH_PWR6 || defined _ARCH_PWR6X
# define MUTEX_HINT_ACQ	",1"
# define MUTEX_HINT_REL	",0"
#else
# define MUTEX_HINT_ACQ
# define MUTEX_HINT_REL
#endif

#define __HAVE_64B_ATOMICS 1
#define USE_ATOMIC_COMPILER_BUILTINS 0
#define ATOMIC_EXCHANGE_USES_CAS 1

/* The 32-bit exchange_bool is different on powerpc64 because the subf
   does signed 64-bit arithmetic while the lwarx is 32-bit unsigned
   (a load word and zero (high 32) form) load.
   In powerpc64 register values are 64-bit by default,  including oldval.
   The value in old val unknown sign extension, lwarx loads the 32-bit
   value as unsigned.  So we explicitly clear the high 32 bits in oldval.  */
#define __arch_compare_and_exchange_bool_32_acq(mem, newval, oldval) \
({									      \
  unsigned int __tmp, __tmp2;						      \
  __asm __volatile ("   clrldi  %1,%1,32\n"				      \
		    "1:	lwarx	%0,0,%2" MUTEX_HINT_ACQ "\n"	 	      \
		    "	subf.	%0,%1,%0\n"				      \
		    "	bne	2f\n"					      \
		    "	stwcx.	%4,0,%2\n"				      \
		    "	bne-	1b\n"					      \
		    "2:	" __ARCH_ACQ_INSTR				      \
		    : "=&r" (__tmp), "=r" (__tmp2)			      \
		    : "b" (mem), "1" (oldval), "r" (newval)		      \
		    : "cr0", "memory");					      \
  __tmp != 0;								      \
})

/*
 * Only powerpc64 processors support Load doubleword and reserve index (ldarx)
 * and Store doubleword conditional indexed (stdcx) instructions.  So here
 * we define the 64-bit forms.
 */
#define __arch_compare_and_exchange_bool_64_acq(mem, newval, oldval) \
({									      \
  unsigned long	__tmp;							      \
  __asm __volatile (							      \
		    "1:	ldarx	%0,0,%1" MUTEX_HINT_ACQ "\n"		      \
		    "	subf.	%0,%2,%0\n"				      \
		    "	bne	2f\n"					      \
		    "	stdcx.	%3,0,%1\n"				      \
		    "	bne-	1b\n"					      \
		    "2:	" __ARCH_ACQ_INSTR				      \
		    : "=&r" (__tmp)					      \
		    : "b" (mem), "r" (oldval), "r" (newval)		      \
		    : "cr0", "memory");					      \
  __tmp != 0;								      \
})

#define __arch_compare_and_exchange_val_64_acq(mem, newval, oldval) \
  ({									      \
      __typeof (*(mem)) __tmp;						      \
      __typeof (mem)  __memp = (mem);					      \
      __asm __volatile (						      \
		        "1:	ldarx	%0,0,%1" MUTEX_HINT_ACQ "\n"	      \
		        "	cmpd	%0,%2\n"			      \
		        "	bne	2f\n"				      \
		        "	stdcx.	%3,0,%1\n"			      \
		        "	bne-	1b\n"				      \
		        "2:	" __ARCH_ACQ_INSTR			      \
		        : "=&r" (__tmp)					      \
		        : "b" (__memp), "r" (oldval), "r" (newval)	      \
		        : "cr0", "memory");				      \
      __tmp;								      \
  })

#define __arch_compare_and_exchange_val_64_rel(mem, newval, oldval) \
  ({									      \
      __typeof (*(mem)) __tmp;						      \
      __typeof (mem)  __memp = (mem);					      \
      __asm __volatile (__ARCH_REL_INSTR "\n"				      \
		        "1:	ldarx	%0,0,%1" MUTEX_HINT_REL "\n"	      \
		        "	cmpd	%0,%2\n"			      \
		        "	bne	2f\n"				      \
		        "	stdcx.	%3,0,%1\n"			      \
		        "	bne-	1b\n"				      \
		        "2:	"					      \
		        : "=&r" (__tmp)					      \
		        : "b" (__memp), "r" (oldval), "r" (newval)	      \
		        : "cr0", "memory");				      \
      __tmp;								      \
  })

#define __arch_atomic_exchange_64_acq(mem, value) \
    ({									      \
      __typeof (*mem) __val;						      \
      __asm __volatile (__ARCH_REL_INSTR "\n"				      \
			"1:	ldarx	%0,0,%2" MUTEX_HINT_ACQ "\n"	      \
			"	stdcx.	%3,0,%2\n"			      \
			"	bne-	1b\n"				      \
		  " " __ARCH_ACQ_INSTR					      \
			: "=&r" (__val), "=m" (*mem)			      \
			: "b" (mem), "r" (value), "m" (*mem)		      \
			: "cr0", "memory");				      \
      __val;								      \
    })

#define __arch_atomic_exchange_64_rel(mem, value) \
    ({									      \
      __typeof (*mem) __val;						      \
      __asm __volatile (__ARCH_REL_INSTR "\n"				      \
			"1:	ldarx	%0,0,%2" MUTEX_HINT_REL "\n"	      \
			"	stdcx.	%3,0,%2\n"			      \
			"	bne-	1b"				      \
			: "=&r" (__val), "=m" (*mem)			      \
			: "b" (mem), "r" (value), "m" (*mem)		      \
			: "cr0", "memory");				      \
      __val;								      \
    })

#define __arch_atomic_exchange_and_add_64(mem, value) \
    ({									      \
      __typeof (*mem) __val, __tmp;					      \
      __asm __volatile ("1:	ldarx	%0,0,%3\n"			      \
			"	add	%1,%0,%4\n"			      \
			"	stdcx.	%1,0,%3\n"			      \
			"	bne-	1b"				      \
			: "=&b" (__val), "=&r" (__tmp), "=m" (*mem)	      \
			: "b" (mem), "r" (value), "m" (*mem)		      \
			: "cr0", "memory");				      \
      __val;								      \
    })

#define __arch_atomic_exchange_and_add_64_acq(mem, value) \
    ({									      \
      __typeof (*mem) __val, __tmp;					      \
      __asm __volatile ("1:	ldarx	%0,0,%3" MUTEX_HINT_ACQ "\n"	      \
			"	add	%1,%0,%4\n"			      \
			"	stdcx.	%1,0,%3\n"			      \
			"	bne-	1b\n"				      \
			__ARCH_ACQ_INSTR				      \
			: "=&b" (__val), "=&r" (__tmp), "=m" (*mem)	      \
			: "b" (mem), "r" (value), "m" (*mem)		      \
			: "cr0", "memory");				      \
      __val;								      \
    })

#define __arch_atomic_exchange_and_add_64_rel(mem, value) \
    ({									      \
      __typeof (*mem) __val, __tmp;					      \
      __asm __volatile (__ARCH_REL_INSTR "\n"				      \
			"1:	ldarx	%0,0,%3" MUTEX_HINT_REL "\n"	      \
			"	add	%1,%0,%4\n"			      \
			"	stdcx.	%1,0,%3\n"			      \
			"	bne-	1b"				      \
			: "=&b" (__val), "=&r" (__tmp), "=m" (*mem)	      \
			: "b" (mem), "r" (value), "m" (*mem)		      \
			: "cr0", "memory");				      \
      __val;								      \
    })

#define __arch_atomic_increment_val_64(mem) \
    ({									      \
      __typeof (*(mem)) __val;						      \
      __asm __volatile ("1:	ldarx	%0,0,%2\n"			      \
			"	addi	%0,%0,1\n"			      \
			"	stdcx.	%0,0,%2\n"			      \
			"	bne-	1b"				      \
			: "=&b" (__val), "=m" (*mem)			      \
			: "b" (mem), "m" (*mem)				      \
			: "cr0", "memory");				      \
      __val;								      \
    })

#define __arch_atomic_decrement_val_64(mem) \
    ({									      \
      __typeof (*(mem)) __val;						      \
      __asm __volatile ("1:	ldarx	%0,0,%2\n"			      \
			"	subi	%0,%0,1\n"			      \
			"	stdcx.	%0,0,%2\n"			      \
			"	bne-	1b"				      \
			: "=&b" (__val), "=m" (*mem)			      \
			: "b" (mem), "m" (*mem)				      \
			: "cr0", "memory");				      \
      __val;								      \
    })

#define __arch_atomic_decrement_if_positive_64(mem) \
  ({ int __val, __tmp;							      \
     __asm __volatile ("1:	ldarx	%0,0,%3\n"			      \
		       "	cmpdi	0,%0,0\n"			      \
		       "	addi	%1,%0,-1\n"			      \
		       "	ble	2f\n"				      \
		       "	stdcx.	%1,0,%3\n"			      \
		       "	bne-	1b\n"				      \
		       "2:	" __ARCH_ACQ_INSTR			      \
		       : "=&b" (__val), "=&r" (__tmp), "=m" (*mem)	      \
		       : "b" (mem), "m" (*mem)				      \
		       : "cr0", "memory");				      \
     __val;								      \
  })

/*
 * All powerpc64 processors support the new "light weight"  sync (lwsync).
 */
#define atomic_read_barrier()	__asm ("lwsync" ::: "memory")
/*
 * "light weight" sync can also be used for the release barrier.
 */
#define __ARCH_REL_INSTR	"lwsync"
#define atomic_write_barrier()	__asm ("lwsync" ::: "memory")

/*
 * Include the rest of the atomic ops macros which are common to both
 * powerpc32 and powerpc64.
 */
#include_next <atomic-machine.h>
