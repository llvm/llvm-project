/* High precision, low overhead timing functions.  IA-64 version.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 2001.

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
typedef unsigned long int hp_timing_t;

/* The Itanium/Merced has a bug where the ar.itc register value read
   is not correct in some situations.  The solution is to read again.
   For now we always do this until we know how to recognize a fixed
   processor implementation.  */
#define REPEAT_READ(val) __builtin_expect ((long int) val == -1, 0)

/* That's quite simple.  Use the `ar.itc' instruction.  */
#define HP_TIMING_NOW(Var) \
  ({ unsigned long int __itc;						      \
     do									      \
       asm volatile ("mov %0=ar.itc" : "=r" (__itc) : : "memory");	      \
     while (REPEAT_READ (__itc));					      \
     Var = __itc; })

#include <hp-timing-common.h>

#endif	/* hp-timing.h */
