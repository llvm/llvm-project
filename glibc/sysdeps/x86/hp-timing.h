/* High precision, low overhead timing functions.  x86 version.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _HP_TIMING_H
#define _HP_TIMING_H	1

#include <isa.h>

#if MINIMUM_ISA == 686 || MINIMUM_ISA == 8664
/* We indeed have inlined functions.  */
# define HP_TIMING_INLINE	(1)

/* We use 64bit values for the times.  */
typedef unsigned long long int hp_timing_t;

/* That's quite simple.  Use the `rdtsc' instruction.  Note that the value
   might not be 100% accurate since there might be some more instructions
   running in this moment.  This could be changed by using a barrier like
   'cpuid' right before the `rdtsc' instruciton.  But we are not interested
   in accurate clock cycles here so we don't do this.

   NB: Use __builtin_ia32_rdtsc directly since including <x86intrin.h>
   makes building glibc very slow.  */
# ifdef USE_RDTSCP
/* RDTSCP waits until all previous instructions have executed and all
   previous loads are globally visible before reading the counter.
   RDTSC doesn't wait until all previous instructions have been executed
   before reading the counter.  */
#  define HP_TIMING_NOW(Var) \
  (__extension__ ({				\
    unsigned int __aux;				\
    (Var) = __builtin_ia32_rdtscp (&__aux);	\
  }))
# else
#  define HP_TIMING_NOW(Var) ((Var) = __builtin_ia32_rdtsc ())
# endif

# include <hp-timing-common.h>
#else
/* NB: Undefine _HP_TIMING_H so that <sysdeps/generic/hp-timing.h> will
   be included.  */
# undef _HP_TIMING_H
# include <sysdeps/generic/hp-timing.h>
#endif

#endif /* hp-timing.h */
