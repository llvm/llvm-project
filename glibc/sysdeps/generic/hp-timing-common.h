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

/* In case a platform supports timers in the hardware the following macros
   and types must be defined:

   - HP_TIMING_INLINE: this macro is non-zero if the functionality is not
     implemented using function calls but instead uses some inlined code
     which might simply consist of a few assembler instructions.  We have to
     know this since we might want to use the macros here in places where we
     cannot make function calls.

   - hp_timing_t: This is the type for variables used to store the time
     values.  This type must be integral.

   - HP_TIMING_NOW: place timestamp for current time in variable given as
     parameter.
*/

/* The target supports hp-timing.  Share the common infrastructure.  */

#include <string.h>
#include <sys/param.h>
#include <_itoa.h>

/* Compute the difference between START and END, storing into DIFF.  */
#define HP_TIMING_DIFF(Diff, Start, End)	((Diff) = (End) - (Start))

/* Accumulate ADD into SUM.  No attempt is made to be thread-safe.  */
#define HP_TIMING_ACCUM_NT(Sum, Diff)		((Sum) += (Diff))

#define HP_TIMING_PRINT_SIZE (3 * sizeof (hp_timing_t) + 1)

/* Write a decimal representation of the timing value into the given string.  */
#define HP_TIMING_PRINT(Dest, Len, Val) 				\
  do {									\
    char __buf[HP_TIMING_PRINT_SIZE];					\
    char *__dest = (Dest);						\
    size_t __len = (Len);						\
    char *__cp = _itoa ((Val), __buf + sizeof (__buf), 10, 0);		\
    size_t __cp_len = MIN (__buf + sizeof (__buf) - __cp, __len);	\
    memcpy (__dest, __cp, __cp_len);					\
    __dest[__cp_len - 1] = '\0';					\
  } while (0)
