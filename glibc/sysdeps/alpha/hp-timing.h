/* High precision, low overhead timing functions.  Alpha version.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Richard Henderson <rth@redhat.com>, 2001.

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

#ifndef _HP_TIMING_ALPHA_H
#define _HP_TIMING_ALPHA_H	1

#if IS_IN(rtld)
/* We always have the timestamp register, but it's got only a 4 second
   range.  Use it for ld.so profiling only.  */
# define HP_TIMING_INLINE	(1)

/* We use 32 bit values for the times.  */
typedef unsigned int hp_timing_t;

/* The "rpcc" instruction returns a 32-bit counting half and a 32-bit
   "virtual cycle counter displacement".  Subtracting the two gives us
   a virtual cycle count.  */
# define HP_TIMING_NOW(VAR) \
  do {									      \
    unsigned long int x_;						      \
    asm volatile ("rpcc %0" : "=r"(x_));				      \
    (VAR) = (int) (x_) - (int) (x_ >> 32);				      \
  } while (0)
# include <hp-timing-common.h>

#else
# include <sysdeps/generic/hp-timing.h>
#endif /* IS_IN(rtld)  */

#endif	/* hp-timing.h */
