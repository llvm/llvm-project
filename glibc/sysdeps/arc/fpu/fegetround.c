/* Return current rounding direction.
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

#include <fenv.h>
#include <fenv_private.h>

int
__fegetround (void)
{
  unsigned int fpcr;
  _FPU_GETCW (fpcr);

  return (fpcr >> __FPU_RND_SHIFT) & __FPU_RND_MASK;
}
libm_hidden_def (__fegetround)
weak_alias (__fegetround, fegetround)
libm_hidden_weak (fegetround)
