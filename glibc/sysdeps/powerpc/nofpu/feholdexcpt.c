/* Store current floating-point environment and clear exceptions
   (soft-float edition).
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
   Contributed by Aldy Hernandez <aldyh@redhat.com>, 2002.
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

#include "soft-fp.h"
#include "soft-supp.h"

int
__feholdexcept (fenv_t *envp)
{
  fenv_union_t u;

  /* Get the current state.  */
  __fegetenv (envp);

  u.fenv = *envp;
  /* Clear everything except the rounding mode.  */
  u.l[0] &= 0x3;
  /* Disable exceptions */
  u.l[1] = FE_ALL_EXCEPT;

  /* Put the new state in effect.  */
  __fesetenv (&u.fenv);

  return 0;
}
libm_hidden_def (__feholdexcept)
weak_alias (__feholdexcept, feholdexcept)
libm_hidden_weak (feholdexcept)
