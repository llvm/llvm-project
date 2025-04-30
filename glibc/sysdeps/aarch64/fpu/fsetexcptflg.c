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
fesetexceptflag (const fexcept_t *flagp, int excepts)
{
  fpu_fpsr_t fpsr;
  fpu_fpsr_t fpsr_new;

  /* Get the current environment.  */
  _FPU_GETFPSR (fpsr);
  excepts &= FE_ALL_EXCEPT;

  /* Set the desired exception mask.  */
  fpsr_new = fpsr & ~excepts;
  fpsr_new |= *flagp & excepts;

  /* Save state back to the FPU.  */
  if (fpsr != fpsr_new)
    _FPU_SETFPSR (fpsr_new);

  return 0;
}
