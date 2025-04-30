/* Determine floating-point rounding mode within libc.  ARC version.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#ifndef _ARC_GET_ROUNDING_MODE_H
#define _ARC_GET_ROUNDING_MODE_H	1

#include <fenv.h>
#include <fpu_control.h>

static inline int
get_rounding_mode (void)
{
#if defined(__ARC_FPU_SP__) ||  defined(__ARC_FPU_DP__)
  unsigned int fpcr;
  _FPU_GETCW (fpcr);

  return (fpcr >> __FPU_RND_SHIFT) & __FPU_RND_MASK;
#else
  return FE_TONEAREST;
#endif
}

#endif /* get-rounding-mode.h */
