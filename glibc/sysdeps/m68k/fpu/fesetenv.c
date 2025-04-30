/* Install given floating-point environment.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Schwab <schwab@issan.informatik.uni-dortmund.de>

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

int
__fesetenv (const fenv_t *envp)
{
  fenv_t temp;

  /* Install the environment specified by ENVP.  But there are a few
     values which we do not want to come from the saved environment.
     Therefore, we get the current environment and replace the values
     we want to use from the environment specified by the parameter.  */
#ifdef __mcoldfire__
  __asm__ ("fmove%.l %/fpcr,%0" : "=dm" (temp.__control_register));
  __asm__ ("fmove%.l %/fpsr,%0" : "=dm" (temp.__status_register));
  __asm__ ("fmove%.l %/fpiar,%0" : "=dm" (temp.__instruction_address));
#else
  __asm__ ("fmovem%.l %/fpcr/%/fpsr/%/fpiar,%0" : "=m" (*&temp));
#endif

  temp.__status_register &= ~FE_ALL_EXCEPT;
  temp.__control_register &= ~((FE_ALL_EXCEPT << 6) | FE_UPWARD);
  if (envp == FE_DFL_ENV)
    ;
  else if (envp == FE_NOMASK_ENV)
    temp.__control_register |= FE_ALL_EXCEPT << 6;
  else
    {
      temp.__control_register |= (envp->__control_register
				  & ((FE_ALL_EXCEPT << 6) | FE_UPWARD));
      temp.__status_register |= envp->__status_register & FE_ALL_EXCEPT;
    }

#ifdef __mcoldfire__
  __asm__ __volatile__ ("fmove%.l %0,%/fpiar"
			:: "dm" (temp.__instruction_address));
  __asm__ __volatile__ ("fmove%.l %0,%/fpcr"
			:: "dm" (temp.__control_register));
  __asm__ __volatile__ ("fmove%.l %0,%/fpsr"
			:: "dm" (temp.__status_register));
#else
  __asm__ __volatile__ ("fmovem%.l %0,%/fpcr/%/fpsr/%/fpiar" : : "m" (*&temp));
#endif

  /* Success.  */
  return 0;
}

#include <shlib-compat.h>
#if SHLIB_COMPAT (libm, GLIBC_2_1, GLIBC_2_2)
strong_alias (__fesetenv, __old_fesetenv)
compat_symbol (libm, __old_fesetenv, fesetenv, GLIBC_2_1);
#endif

libm_hidden_def (__fesetenv)
libm_hidden_ver (__fesetenv, fesetenv)
versioned_symbol (libm, __fesetenv, fesetenv, GLIBC_2_2);
