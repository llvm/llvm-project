/* Control when floating-point expressions are evaluated.  Alpha version.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#ifndef ALPHA_MATH_BARRIERS_H
#define ALPHA_MATH_BARRIERS_H 1

/* Generic code forces values to memory; we don't need to do that.  */
#define math_opt_barrier(x) \
  ({ __typeof (x) __x = (x); __asm ("" : "+frm" (__x)); __x; })
#define math_force_eval(x) \
  ({ __typeof (x) __x = (x); __asm __volatile__ ("" : : "frm" (__x)); })

#endif
