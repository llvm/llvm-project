/* Store current floating-point environment and clear exceptions.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <fenv.h>
#include <fpu_control.h>

int
__feholdexcept (fenv_t *envp)
{
  fpu_control_t temp;

  /* Store the environment.  */
  _FPU_GETCW (temp);
  envp->__fpscr = temp;

  /* Clear the status flags.  */
  temp &= ~FE_ALL_EXCEPT;

  /* Now set all exceptions to non-stop.  */
  temp &= ~(FE_ALL_EXCEPT << 5);

  _FPU_SETCW (temp);

  /* Success.  */
  return 0;
}
libm_hidden_def (__feholdexcept)
weak_alias (__feholdexcept, feholdexcept)
libm_hidden_weak (feholdexcept)
