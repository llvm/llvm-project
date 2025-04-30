/* Set current rounding direction.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Richard Henderson <rth@tamu.edu>, 1997

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

#include <fenv_libc.h>

int
__fesetround (int round)
{
  unsigned long fpcr;

  if (round & ~3)
    return 1;

  /* Get the current state.  */
  __asm__ __volatile__("excb; mf_fpcr %0" : "=f"(fpcr));

  /* Set the relevant bits.  */
  fpcr = ((fpcr & ~FPCR_ROUND_MASK)
	  | ((unsigned long)round << FPCR_ROUND_SHIFT));

  /* Put the new state in effect.  */
  __asm__ __volatile__("mt_fpcr %0; excb" : : "f"(fpcr));

  return 0;
}
libm_hidden_def (__fesetround)
weak_alias (__fesetround, fesetround)
libm_hidden_weak (fesetround)
