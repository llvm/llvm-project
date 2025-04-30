/* Copyright (C) 2001-2021 Free Software Foundation, Inc.

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
feenableexcept (int excepts)
{
  fpu_control_t fpcr;
  fpu_control_t fpcr_new;
  fpu_control_t updated_fpcr;

  _FPU_GETCW (fpcr);
  excepts &= FE_ALL_EXCEPT;
  fpcr_new = fpcr | (excepts << FE_EXCEPT_SHIFT);

  if (fpcr != fpcr_new)
    {
      _FPU_SETCW (fpcr_new);

      /* Trapping exceptions are optional in AArch64; the relevant enable
	 bits in FPCR are RES0 hence the absence of support can be detected
	 by reading back the FPCR and comparing with the required value.  */
      _FPU_GETCW (updated_fpcr);

      if (fpcr_new & ~updated_fpcr)
	return -1;
    }

  return (fpcr >> FE_EXCEPT_SHIFT) & FE_ALL_EXCEPT;
}
