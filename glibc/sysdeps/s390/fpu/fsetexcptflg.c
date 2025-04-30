/* Set floating-point environment exception handling.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Denis Joseph Barrow (djbarrow@de.ibm.com).

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

#include <fenv_libc.h>
#include <math.h>
#include <fpu_control.h>

int
fesetexceptflag (const fexcept_t *flagp, int excepts)
{
  fexcept_t fpc, fpc_new;

  /* Get the current environment.  We have to do this since we cannot
     separately set the status word.  */
  _FPU_GETCW (fpc);

  /* Clear the current exception bits.  */
  fpc_new = fpc & ~((excepts & FE_ALL_EXCEPT) << FPC_FLAGS_SHIFT);
  if ((fpc & FPC_NOT_FPU_EXCEPTION) == 0)
    /* Bits 6, 7 of dxc-byte are zero,
       thus bits 0-5 of dxc-byte correspond to the flag-bits.
       Clear given exceptions in dxc-field.  */
    fpc_new &= ~((excepts & FE_ALL_EXCEPT) << FPC_DXC_SHIFT);

  /* Set exceptions from flagp in flags-field.  */
  fpc_new |= (*flagp & excepts & FE_ALL_EXCEPT) << FPC_FLAGS_SHIFT;

  /* Store the new status word (along with the rest of the environment.
     Possibly new exceptions are set but they won't get executed.  */
  _FPU_SETCW (fpc_new);

  /* Success.  */
  return 0;
}
