/* waitpid with error checking.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <support/xunistd.h>

#include <stdio.h>
#include <stdlib.h>
#include <support/check.h>
#include <sys/wait.h>

int
xwaitpid (int pid, int *status, int flags)
{
  pid_t result = waitpid (pid, status, flags);
  if (result < 0)
    FAIL_EXIT1 ("waitpid: %m\n");
  return result;
}
