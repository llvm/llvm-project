/* Disable exceptions (soft-float edition).
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
#include <fenv.h>

int
fedisableexcept (int x)
{
  int old_exceptions = ~__sim_disabled_exceptions_thread & FE_ALL_EXCEPT;

  __sim_disabled_exceptions_thread |= x;
  SIM_SET_GLOBAL (__sim_disabled_exceptions_global,
		  __sim_disabled_exceptions_thread);

  return old_exceptions;
}
