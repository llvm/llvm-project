/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Schwab <schwab@suse.de>, 2003.

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

#define __arch_compare_and_exchange_val_8_acq(mem, newval, oldval) \
  ({ __typeof (*(mem)) __ret;						      \
     __asm __volatile ("cas%.b %0,%2,%1"				      \
		       : "=d" (__ret), "+m" (*(mem))			      \
		       : "d" (newval), "0" (oldval));			      \
     __ret; })

#define __arch_compare_and_exchange_val_16_acq(mem, newval, oldval) \
  ({ __typeof (*(mem)) __ret;						      \
     __asm __volatile ("cas%.w %0,%2,%1"				      \
		       : "=d" (__ret), "+m" (*(mem))			      \
		       : "d" (newval), "0" (oldval));			      \
     __ret; })

#define __arch_compare_and_exchange_val_32_acq(mem, newval, oldval) \
  ({ __typeof (*(mem)) __ret;						      \
     __asm __volatile ("cas%.l %0,%2,%1"				      \
		       : "=d" (__ret), "+m" (*(mem))			      \
		       : "d" (newval), "0" (oldval));			      \
     __ret; })

# define __arch_compare_and_exchange_val_64_acq(mem, newval, oldval) \
  ({ __typeof (*(mem)) __ret;						      \
     __typeof (mem) __memp = (mem);					      \
     __asm __volatile ("cas2%.l %0:%R0,%1:%R1,(%2):(%3)"		      \
		       : "=d" (__ret)					      \
		       : "d" ((__typeof (*(mem))) (newval)), "r" (__memp),    \
			 "r" ((char *) __memp + 4), "0" (oldval)	      \
		       : "memory");					      \
     __ret; })

#define atomic_exchange_acq(mem, newvalue) \
  ({ __typeof (*(mem)) __result = *(mem);				      \
     if (sizeof (*(mem)) == 1)						      \
       __asm __volatile ("1: cas%.b %0,%2,%1;"				      \
			 "   jbne 1b"					      \
			 : "=d" (__result), "+m" (*(mem))		      \
			 : "d" (newvalue), "0" (__result));		      \
     else if (sizeof (*(mem)) == 2)					      \
       __asm __volatile ("1: cas%.w %0,%2,%1;"				      \
			 "   jbne 1b"					      \
			 : "=d" (__result), "+m" (*(mem))		      \
			 : "d" (newvalue), "0" (__result));		      \
     else if (sizeof (*(mem)) == 4)					      \
       __asm __volatile ("1: cas%.l %0,%2,%1;"				      \
			 "   jbne 1b"					      \
			 : "=d" (__result), "+m" (*(mem))		      \
			 : "d" (newvalue), "0" (__result));		      \
     else								      \
       {								      \
	 __typeof (mem) __memp = (mem);					      \
	 __asm __volatile ("1: cas2%.l %0:%R0,%1:%R1,(%2):(%3);"	      \
			   "   jbne 1b"					      \
			   : "=d" (__result)				      \
			   : "d" ((__typeof (*(mem))) (newvalue)),	      \
			     "r" (__memp), "r" ((char *) __memp + 4),	      \
			     "0" (__result)				      \
			   : "memory");					      \
       }								      \
     __result; })

#define atomic_exchange_and_add(mem, value) \
  ({ __typeof (*(mem)) __result = *(mem);				      \
     __typeof (*(mem)) __temp;						      \
     if (sizeof (*(mem)) == 1)						      \
       __asm __volatile ("1: move%.b %0,%2;"				      \
			 "   add%.b %3,%2;"				      \
			 "   cas%.b %0,%2,%1;"				      \
			 "   jbne 1b"					      \
			 : "=d" (__result), "+m" (*(mem)),		      \
			   "=&d" (__temp)				      \
			 : "d" (value), "0" (__result));		      \
     else if (sizeof (*(mem)) == 2)					      \
       __asm __volatile ("1: move%.w %0,%2;"				      \
			 "   add%.w %3,%2;"				      \
			 "   cas%.w %0,%2,%1;"				      \
			 "   jbne 1b"					      \
			 : "=d" (__result), "+m" (*(mem)),		      \
			   "=&d" (__temp)				      \
			 : "d" (value), "0" (__result));		      \
     else if (sizeof (*(mem)) == 4)					      \
       __asm __volatile ("1: move%.l %0,%2;"				      \
			 "   add%.l %3,%2;"				      \
			 "   cas%.l %0,%2,%1;"				      \
			 "   jbne 1b"					      \
			 : "=d" (__result), "+m" (*(mem)),		      \
			   "=&d" (__temp)				      \
			 : "d" (value), "0" (__result));		      \
     else								      \
       {								      \
	 __typeof (mem) __memp = (mem);					      \
	 __asm __volatile ("1: move%.l %0,%1;"				      \
			   "   move%.l %R0,%R1;"			      \
			   "   add%.l %R2,%R1;"				      \
			   "   addx%.l %2,%1;"				      \
			   "   cas2%.l %0:%R0,%1:%R1,(%3):(%4);"	      \
			   "   jbne 1b"					      \
			   : "=d" (__result), "=&d" (__temp)		      \
			   : "d" ((__typeof (*(mem))) (value)), "r" (__memp), \
			     "r" ((char *) __memp + 4), "0" (__result)	      \
			   : "memory");					      \
       }								      \
     __result; })

#define atomic_add(mem, value) \
  (void) ({ if (sizeof (*(mem)) == 1)					      \
	      __asm __volatile ("add%.b %1,%0"				      \
				: "+m" (*(mem))				      \
				: "id" (value));			      \
	    else if (sizeof (*(mem)) == 2)				      \
	      __asm __volatile ("add%.w %1,%0"				      \
				: "+m" (*(mem))				      \
				: "id" (value));			      \
	    else if (sizeof (*(mem)) == 4)				      \
	      __asm __volatile ("add%.l %1,%0"				      \
				: "+m" (*(mem))				      \
				: "id" (value));			      \
	    else							      \
	      {								      \
		__typeof (mem) __memp = (mem);				      \
		__typeof (*(mem)) __oldval = *__memp;			      \
		__typeof (*(mem)) __temp;				      \
		__asm __volatile ("1: move%.l %0,%1;"			      \
				  "   move%.l %R0,%R1;"			      \
				  "   add%.l %R2,%R1;"			      \
				  "   addx%.l %2,%1;"			      \
				  "   cas2%.l %0:%R0,%1:%R1,(%3):(%4);"	      \
				  "   jbne 1b"				      \
				  : "=d" (__oldval), "=&d" (__temp)	      \
				  : "d" ((__typeof (*(mem))) (value)),	      \
				    "r" (__memp), "r" ((char *) __memp + 4),  \
				    "0" (__oldval)			      \
				  : "memory");				      \
	      }								      \
	    })

#define atomic_increment_and_test(mem) \
  ({ char __result;							      \
     if (sizeof (*(mem)) == 1)						      \
       __asm __volatile ("addq%.b %#1,%1; seq %0"			      \
			 : "=dm" (__result), "+m" (*(mem)));		      \
     else if (sizeof (*(mem)) == 2)					      \
       __asm __volatile ("addq%.w %#1,%1; seq %0"			      \
			 : "=dm" (__result), "+m" (*(mem)));		      \
     else if (sizeof (*(mem)) == 4)					      \
       __asm __volatile ("addq%.l %#1,%1; seq %0"			      \
			 : "=dm" (__result), "+m" (*(mem)));		      \
     else								      \
       {								      \
	 __typeof (mem) __memp = (mem);					      \
	 __typeof (*(mem)) __oldval = *__memp;				      \
	 __typeof (*(mem)) __temp;					      \
	 __asm __volatile ("1: move%.l %1,%2;"				      \
			   "   move%.l %R1,%R2;"			      \
			   "   addq%.l %#1,%R2;"			      \
			   "   addx%.l %5,%2;"				      \
			   "   seq %0;"					      \
			   "   cas2%.l %1:%R1,%2:%R2,(%3):(%4);"	      \
			   "   jbne 1b"					      \
			   : "=&dm" (__result), "=d" (__oldval),	      \
			     "=&d" (__temp)				      \
			   : "r" (__memp), "r" ((char *) __memp + 4),	      \
			     "d" (0), "1" (__oldval)			      \
			   : "memory");					      \
       }								      \
     __result; })

#define atomic_decrement_and_test(mem) \
  ({ char __result;							      \
     if (sizeof (*(mem)) == 1)						      \
       __asm __volatile ("subq%.b %#1,%1; seq %0"			      \
			 : "=dm" (__result), "+m" (*(mem)));		      \
     else if (sizeof (*(mem)) == 2)					      \
       __asm __volatile ("subq%.w %#1,%1; seq %0"			      \
			 : "=dm" (__result), "+m" (*(mem)));		      \
     else if (sizeof (*(mem)) == 4)					      \
       __asm __volatile ("subq%.l %#1,%1; seq %0"			      \
			 : "=dm" (__result), "+m" (*(mem)));		      \
     else								      \
       {								      \
	 __typeof (mem) __memp = (mem);					      \
	 __typeof (*(mem)) __oldval = *__memp;				      \
	 __typeof (*(mem)) __temp;					      \
	 __asm __volatile ("1: move%.l %1,%2;"				      \
			   "   move%.l %R1,%R2;"			      \
			   "   subq%.l %#1,%R2;"			      \
			   "   subx%.l %5,%2;"				      \
			   "   seq %0;"					      \
			   "   cas2%.l %1:%R1,%2:%R2,(%3):(%4);"	      \
			   "   jbne 1b"					      \
			   : "=&dm" (__result), "=d" (__oldval),	      \
			     "=&d" (__temp)				      \
			   : "r" (__memp), "r" ((char *) __memp + 4),	      \
			     "d" (0), "1" (__oldval)			      \
			   : "memory");					      \
       }								      \
     __result; })

#define atomic_bit_set(mem, bit) \
  __asm __volatile ("bfset %0{%1,#1}"					      \
		    : "+m" (*(mem))					      \
		    : "di" (sizeof (*(mem)) * 8 - (bit) - 1))

#define atomic_bit_test_set(mem, bit) \
  ({ char __result;							      \
     __asm __volatile ("bfset %1{%2,#1}; sne %0"			      \
		       : "=dm" (__result), "+m" (*(mem))		      \
		       : "di" (sizeof (*(mem)) * 8 - (bit) - 1));	      \
     __result; })
