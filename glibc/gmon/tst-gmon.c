/* Test program for profiling information collection (_mcount/gprof).
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

/* This program does not use the test harness because we want tight
   control over the call graph.  */

__attribute__ ((noinline, noclone, weak)) void
f1 (void)
{
}

__attribute__ ((noinline, noclone, weak)) void
f2 (void)
{
  f1 ();
  /* Prevent tail call.  */
  asm volatile ("");
}

__attribute__ ((noinline, noclone, weak)) void
f3 (int count)
{
  for (int i = 0; i < count; ++i)
    {
      f1 ();
      f2 ();
    }
}

int
main (void)
{
  f3 (1000);
  return 0;
}
