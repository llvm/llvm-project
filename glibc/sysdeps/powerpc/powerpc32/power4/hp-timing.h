/* High precision, low overhead timing functions.  powerpc64 version.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1998.

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

/* We indeed have inlined functions.  */
#define HP_TIMING_INLINE	(1)

/* We use 64bit values for the times.  */
typedef unsigned long long int hp_timing_t;

/* That's quite simple.  Use the `mftb' instruction.  Note that the value
   might not be 100% accurate since there might be some more instructions
   running in this moment.  This could be changed by using a barrier like
   'lwsync' right before the `mftb' instruction.  But we are not interested
   in accurate clock cycles here so we don't do this.  */

#define HP_TIMING_NOW(Var)						\
  do {									\
    unsigned int hi, lo, tmp;						\
    __asm__ __volatile__ ("1:	mfspr	%0,269;"			\
			  "	mfspr	%1,268;"			\
			  "	mfspr	%2,269;"			\
			  "	cmpw	%0,%2;"				\
			  "	bne	1b;"				\
			  : "=&r" (hi), "=&r" (lo), "=&r" (tmp)		\
			  : : "cr0");					\
    Var = ((hp_timing_t) hi << 32) | lo;				\
  } while (0)

#include <hp-timing-common.h>

#endif	/* hp-timing.h */
