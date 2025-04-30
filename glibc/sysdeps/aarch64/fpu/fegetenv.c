/* Copyright (C) 1997-2021 Free Software Foundation, Inc.

   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

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
__fegetenv (fenv_t *envp)
{
  fpu_control_t fpcr;
  fpu_fpsr_t fpsr;
  _FPU_GETCW (fpcr);
  _FPU_GETFPSR (fpsr);
  envp->__fpcr = fpcr;
  envp->__fpsr = fpsr;
  return 0;
}
libm_hidden_def (__fegetenv)
weak_alias (__fegetenv, fegetenv)
libm_hidden_weak (fegetenv)
