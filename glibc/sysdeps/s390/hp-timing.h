/* High precision, low overhead timing functions.  s390 version.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#ifndef _HP_TIMING_S390_H
#define _HP_TIMING_S390_H	1

/* The stckf instruction is available starting with z9-109 zarch CPUs.
   As there is no extra configure check for z9-109, the z10 one is used.  */
#ifdef HAVE_S390_MIN_Z10_ZARCH_ASM_SUPPORT
# include <hp-timing-common.h>

/* We use 64 bit values for the times.
   Note: Bit 51 is incremented every 0.000 001s = 1us.  */
typedef unsigned long long int hp_timing_t;

# define HP_TIMING_INLINE	(1)

# define HP_TIMING_NOW(VAR)						\
  do {									\
    __asm__ __volatile__ ("stckf %0" : "=Q" (VAR) : : "cc");		\
  } while (0)

#else
# include_next <hp-timing.h>
#endif

#endif	/* hp-timing.h */
