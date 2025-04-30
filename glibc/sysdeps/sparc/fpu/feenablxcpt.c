/* Enable floating-point exceptions.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2000.

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
#include <fenv_private.h>

int
feenableexcept (int excepts)
{
  fenv_t new_exc, old_exc;

  __fenv_stfsr (new_exc);

  old_exc = (new_exc >> 18) & FE_ALL_EXCEPT;
  new_exc |= (((fenv_t)excepts & FE_ALL_EXCEPT) << 18);

  __fenv_ldfsr (new_exc);

  return old_exc;
}
