/* Benchmark utility functions.
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


static volatile unsigned int dontoptimize = 0;

void
bench_start (void)
{
  /* This loop should cause CPU to switch to maximal freqency.
     This makes subsequent measurement more accurate.  We need a side effect
     to prevent the loop being deleted by compiler.
     This should be enough to cause CPU to speed up and it is simpler than
     running loop for constant time.  This is used when user does not have root
     access to set a constant freqency.  */

  for (int k = 0; k < START_ITER; k++)
    dontoptimize += 23 * dontoptimize + 2;
}
