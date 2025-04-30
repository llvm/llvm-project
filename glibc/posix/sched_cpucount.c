/* Copyright (C) 2007-2021 Free Software Foundation, Inc.
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

#include <sched.h>

/* Counting bits set, Brian Kernighan's way.
   Using a open-coded routine is slight better for architectures that
   do not have a popcount instruction (compiler might emit a library
   call).  */
static inline int
countbits (__cpu_mask v)
{
  int s = 0;
  for (; v != 0; s++)
    v &= v - 1;
  return s;
}

int
__sched_cpucount (size_t setsize, const cpu_set_t *setp)
{
  int s = 0;
  for (int i = 0; i < setsize / sizeof (__cpu_mask); i++)
    s += countbits (setp->__bits[i]);
  return s;
}
libc_hidden_def (__sched_cpucount)
