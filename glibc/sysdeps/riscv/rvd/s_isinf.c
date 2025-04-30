/* isinf().  RISC-V version.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <math.h>
#include <fenv_private.h>

int
__isinf (double x)
{
  int cls = _FCLASS (x);
  return -((cls & _FCLASS_MINF) ? 1 : 0) | ((cls & _FCLASS_PINF) ? 1 : 0);
}
hidden_def (__isinf)
weak_alias (__isinf, isinf)
