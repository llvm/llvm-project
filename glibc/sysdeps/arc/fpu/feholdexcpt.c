/* Store current floating-point environment and clear exceptions.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <fenv.h>
#include <fenv_private.h>

int
__feholdexcept (fenv_t *envp)
{
  unsigned int fpcr;
  unsigned int fpsr;

  _FPU_GETCW (fpcr);
  _FPU_GETS (fpsr);

  envp->__fpcr = fpcr;
  envp->__fpsr = fpsr;

  fpsr &= ~FE_ALL_EXCEPT;

  _FPU_SETCW (fpcr);
  _FPU_SETS (fpsr);

  return 0;
}
libm_hidden_def (__feholdexcept)
weak_alias (__feholdexcept, feholdexcept)
libm_hidden_weak (feholdexcept)
