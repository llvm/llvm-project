/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.ai.mit.edu>, August 1995.

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

#include <sys/ipc.h>
#include <sys/stat.h>

key_t
ftok (const char *pathname, int proj_id)
{
  struct __stat64_t64 st;
  key_t key;

  if (__stat64_time64 (pathname, &st) < 0)
    return (key_t) -1;

  key = ((st.st_ino & 0xffff) | ((st.st_dev & 0xff) << 16)
	 | ((proj_id & 0xff) << 24));

  return key;
}
