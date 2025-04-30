/* Store current representation for exceptions.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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

#include <fenv.h>
#include <unistd.h>
#include <ldsodefs.h>
#include <dl-procinfo.h>


int
__fegetexceptflag (fexcept_t *flagp, int excepts)
{
  fexcept_t temp;

  /* Get the current exceptions.  */
  __asm__ ("fnstsw %0" : "=m" (*&temp));

  *flagp = temp & excepts & FE_ALL_EXCEPT;

  /* If the CPU supports SSE, we clear the MXCSR as well.  */
  if (CPU_FEATURE_USABLE (SSE))
    {
      unsigned int sse_exc;

      /* Get the current MXCSR.  */
      __asm__ ("stmxcsr %0" : "=m" (*&sse_exc));

      *flagp |= sse_exc & excepts & FE_ALL_EXCEPT;
    }

  /* Success.  */
  return 0;
}

#include <shlib-compat.h>
#if SHLIB_COMPAT (libm, GLIBC_2_1, GLIBC_2_2)
strong_alias (__fegetexceptflag, __old_fegetexceptflag)
compat_symbol (libm, __old_fegetexceptflag, fegetexceptflag, GLIBC_2_1);
#endif

versioned_symbol (libm, __fegetexceptflag, fegetexceptflag, GLIBC_2_2);
