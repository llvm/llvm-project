/* Store current floating-point environment (soft-float edition).
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
   Contributed by Aldy Hernandez <aldyh@redhat.com>, 2002, 2010.
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
__fegetenv (fenv_t *envp)
{
  fenv_union_t u;

  u.l[0] = __sim_exceptions_thread;
  u.l[0] |= __sim_round_mode_thread;
  u.l[1] = __sim_disabled_exceptions_thread;

  *envp = u.fenv;

  return 0;
}

#include <shlib-compat.h>
#if SHLIB_COMPAT (libm, GLIBC_2_1, GLIBC_2_2)
strong_alias (__fegetenv, __old_fegetenv)
compat_symbol (libm, __old_fegetenv, fegetenv, GLIBC_2_1);
#endif

libm_hidden_def (__fegetenv)
libm_hidden_ver (__fegetenv, fegetenv)
versioned_symbol (libm, __fegetenv, fegetenv, GLIBC_2_2);
