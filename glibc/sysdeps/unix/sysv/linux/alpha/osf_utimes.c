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

#include <time.h>
#include <sys/time.h>

int
attribute_compat_text_section
__utimes_tv32 (const char *filename, const struct __timeval32 times32[2])
{
  struct timeval times[2];
  times[0] = valid_timeval32_to_timeval (times32[0]);
  times[1] = valid_timeval32_to_timeval (times32[1]);
  return __utimes (filename, times);
}

compat_symbol (libc, __utimes_tv32, utimes, GLIBC_2_0);
#endif
