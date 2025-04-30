/* Store current representation for exceptions.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Christian Boissat <Christian.Boissat@cern.ch>, 1999.

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

int
fegetexceptflag (fexcept_t *flagp, int excepts)
{
  fenv_t fpsr;

  /* Get the current exceptions.  */
  __asm__ __volatile__ ("mov.m %0=ar.fpsr" : "=r" (fpsr));

  *flagp = (fexcept_t) ((fpsr >> 13) & excepts & FE_ALL_EXCEPT);

  /* Success.  */
  return 0;
}
