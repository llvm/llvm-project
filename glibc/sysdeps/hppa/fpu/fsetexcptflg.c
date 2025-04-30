/* Set floating-point environment exception handling.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David Huggins-Daines <dhd@debian.org>, 2000

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

#include <fenv.h>
#include <fpu_control.h>

int
fesetexceptflag (const fexcept_t *flagp, int excepts)
{
  fpu_control_t fpsr;
  fpu_control_t fpsr_new;

  /* Get the current status word. */
  _FPU_GETCW (fpsr);
  excepts &= FE_ALL_EXCEPT;

  /* Install new raised flags.  */
  fpsr_new = fpsr & ~(excepts << _FPU_HPPA_SHIFT_FLAGS);
  fpsr_new |= (*flagp & excepts) << _FPU_HPPA_SHIFT_FLAGS;

  /* Store the new status word.  */
  if (fpsr != fpsr_new)
    _FPU_SETCW (fpsr_new);

  /* Success.  */
  return 0;
}
