/* utimes -- change file timestamps.  Linux/Alpha/tv32 version.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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
   <http://www.gnu.org/licenses/>.  */

#include <shlib-compat.h>

#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_1)

#include <sys/time.h>
#include <sys/resource.h>
#include <tv32-compat.h>

int
__getrusage_tv32 (int who, struct __rusage32 *usage32)
{
  struct rusage usage;
  if (__getrusage (who, &usage) == -1)
    return -1;

  rusage64_to_rusage32 (&usage, usage32);
  return 0;
}

compat_symbol (libc, __getrusage_tv32, getrusage, GLIBC_2_0);
#endif
