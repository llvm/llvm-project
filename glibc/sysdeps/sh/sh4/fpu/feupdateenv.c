/* Install given floating-point environment and raise exceptions.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Nobuhiro Iwamatsu <iwamatsu@nigauri.org>, 2012.

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
__feupdateenv (const fenv_t *envp)
{
  fpu_control_t temp;

  _FPU_GETCW (temp);
  temp = (temp & FE_ALL_EXCEPT);

  /* Raise the saved exception. Incidently for us the implementation
    defined format of the values in objects of type fexcept_t is the
    same as the ones specified using the FE_* constants. */
  __fesetenv (envp);
  __feraiseexcept ((int) temp);

  return 0;
}
libm_hidden_def (__feupdateenv)
weak_alias (__feupdateenv, feupdateenv)
libm_hidden_weak (feupdateenv)
