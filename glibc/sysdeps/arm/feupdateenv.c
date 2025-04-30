/* Install given floating-point environment and raise exceptions.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <fenv.h>
#include <fpu_control.h>
#include <arm-features.h>


int
__feupdateenv (const fenv_t *envp)
{
  fpu_control_t fpscr, new_fpscr, updated_fpscr;
  int excepts;

  /* Fail if a VFP unit isn't present.  */
  if (!ARM_HAVE_VFP)
    return 1;

  _FPU_GETCW (fpscr);
  excepts = fpscr & FE_ALL_EXCEPT;

  if ((envp != FE_DFL_ENV) && (envp != FE_NOMASK_ENV))
    {
      /* Merge current exception flags with the saved fenv.  */
      new_fpscr = envp->__cw | excepts;

      /* Write new FPSCR if different (ignoring NZCV flags).  */
      if (((fpscr ^ new_fpscr) & ~_FPU_MASK_NZCV) != 0)
	_FPU_SETCW (new_fpscr);

      /* Raise the exceptions if enabled in the new FP state.  */
      if (excepts & (new_fpscr >> FE_EXCEPT_SHIFT))
	return __feraiseexcept (excepts);

      return 0;
    }

  /* Preserve the reserved FPSCR flags.  */
  new_fpscr = fpscr & (_FPU_RESERVED | FE_ALL_EXCEPT);
  new_fpscr |= (envp == FE_DFL_ENV) ? _FPU_DEFAULT : _FPU_IEEE;

  if (((new_fpscr ^ fpscr) & ~_FPU_MASK_NZCV) != 0)
    {
      _FPU_SETCW (new_fpscr);

      /* Not all VFP architectures support trapping exceptions, so
	 test whether the relevant bits were set and fail if not.  */
      _FPU_GETCW (updated_fpscr);

      if (new_fpscr & ~updated_fpscr)
	return 1;
    }

  /* Raise the exceptions if enabled in the new FP state.  */
  if (excepts & (new_fpscr >> FE_EXCEPT_SHIFT))
    return __feraiseexcept (excepts);

  return 0;
}
libm_hidden_def (__feupdateenv)
weak_alias (__feupdateenv, feupdateenv)
libm_hidden_weak (feupdateenv)
