/* Install given floating-point environment.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Jaeger <aj@suse.de>, 1998.

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
__fesetenv (const fenv_t *envp)
{
  fpu_control_t cw;

  /* Read first current state to flush fpu pipeline.  */
  _FPU_GETCW (cw);

  if (envp == FE_DFL_ENV)
    _FPU_SETCW (_FPU_DEFAULT);
  else if (envp == FE_NOMASK_ENV)
    _FPU_SETCW (_FPU_IEEE);
  else
    _FPU_SETCW (envp->__fp_control_register);

  /* Success.  */
  return 0;
}

libm_hidden_def (__fesetenv)
weak_alias (__fesetenv, fesetenv)
libm_hidden_weak (fesetenv)
