/* Determine floating-point rounding mode within libc.  RISC-V version.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#ifndef _RISCV_GET_ROUNDING_MODE_H
#define _RISCV_GET_ROUNDING_MODE_H

/* Return the floating-point rounding mode.  */

static inline int
get_rounding_mode (void)
{
  int rm;
  asm volatile ("frrm %0" : "=r" (rm));
  return rm;
}

#endif /* get-rounding-mode.h */
