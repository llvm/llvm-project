/* High precision, low overhead timing functions.  Generic version.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
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

#include <time.h>
#include <stdint.h>
#include <hp-timing-common.h>

/* It should not be used for ld.so.  */
#define HP_TIMING_INLINE	(0)

typedef uint64_t hp_timing_t;

/* The clock_gettime (CLOCK_MONOTONIC) has unspecified starting time,
   nano-second accuracy, and for some architectues is implemented as
   vDSO symbol.  */
#ifdef _ISOMAC
# define HP_TIMING_NOW(var) \
({								\
  struct timespec tv;						\
  clock_gettime (CLOCK_MONOTONIC, &tv);				\
  (var) = (tv.tv_nsec + UINT64_C(1000000000) * tv.tv_sec);	\
})
#else
# define HP_TIMING_NOW(var) \
({								\
  struct __timespec64 tv;						\
  __clock_gettime64 (CLOCK_MONOTONIC, &tv);			\
  (var) = (tv.tv_nsec + UINT64_C(1000000000) * tv.tv_sec);	\
})
#endif

#endif	/* hp-timing.h */
