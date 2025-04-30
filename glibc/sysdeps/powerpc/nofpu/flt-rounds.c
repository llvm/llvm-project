/* Return current rounding mode as correct value for FLT_ROUNDS.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#include <stdlib.h>

#include "soft-fp.h"
#include "soft-supp.h"

int
__flt_rounds (void)
{
  switch (__sim_round_mode_thread)
    {
    case FP_RND_ZERO:
      return 0;
    case FP_RND_NEAREST:
      return 1;
    case FP_RND_PINF:
      return 2;
    case FP_RND_MINF:
      return 3;
    default:
      abort ();
    }
}
