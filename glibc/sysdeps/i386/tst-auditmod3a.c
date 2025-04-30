/* Test case for i386 preserved registers in dynamic linker.
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

#include <stdlib.h>
#include "tst-audit3.h"

long long
__attribute__ ((regparm(3)))
audit1_test (int i, int j, int k)
{
  if (i != 1 || j != 2 || k != 3)
    abort ();
  return 30;
}

float
__attribute__ ((regparm(3)))
audit2_test (int i, int j, int k)
{
  if (i != 1 || j != 2 || k != 3)
    abort ();
  return 30;
}
