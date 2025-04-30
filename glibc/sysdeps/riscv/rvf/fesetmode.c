/* Install given floating-point control modes.  RISC-V version.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <fenv.h>
#include <fpu_control.h>

int
fesetmode (const femode_t *modep)
{
  asm volatile ("csrc fcsr, %0" : : "r" (~FE_ALL_EXCEPT));

  if (modep != FE_DFL_MODE)
    asm volatile ("csrs fcsr, %0" : : "r" (*modep & ~FE_ALL_EXCEPT));

  return 0;
}
