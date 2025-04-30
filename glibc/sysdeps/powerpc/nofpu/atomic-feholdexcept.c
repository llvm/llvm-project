/* Store current floating-point environment and clear exceptions for
   atomic compound assignment.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

void
__atomic_feholdexcept (fenv_t *envp)
{
  fenv_union_t u;

  u.l[0] = __sim_exceptions_thread;
  /* The rounding mode is not changed by arithmetic, so no need to
     save it.  */
  u.l[1] = __sim_disabled_exceptions_thread;
  *envp = u.fenv;

  /* This function postdates the global variables being turned into
     compat symbols, so no need to set them.  */
  __sim_exceptions_thread = 0;
  __sim_disabled_exceptions_thread = FE_ALL_EXCEPT;
}
