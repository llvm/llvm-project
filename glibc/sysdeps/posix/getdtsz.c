/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <limits.h>
#include <unistd.h>
#include <sys/resource.h>

/* Return the maximum number of file descriptors
   the current process could possibly have.  */
int
__getdtablesize (void)
{
  struct rlimit ru;

  /* This should even work if `getrlimit' is not implemented.  POSIX.1
     does not define this function but we will generate a stub which
     returns -1.  */
  return __getrlimit (RLIMIT_NOFILE, &ru) < 0 ? OPEN_MAX : ru.rlim_cur;
}

weak_alias (__getdtablesize, getdtablesize)
