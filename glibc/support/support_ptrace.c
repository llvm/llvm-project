/* Support functions handling ptrace_scope.
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
   <https://www.gnu.org/licenses/>.  */

#include <support/check.h>
#include <support/xstdio.h>
#include <support/xptrace.h>

int
support_ptrace_scope (void)
{
  int ptrace_scope = -1;

#ifdef __linux__
  /* YAMA may be not enabled.  Otherwise it contains a value from 0 to 3:
     - 0 classic ptrace permissions
     - 1 restricted ptrace
     - 2 admin-only attach
     - 3 no attach  */
  FILE *f = fopen ("/proc/sys/kernel/yama/ptrace_scope", "r");
  if (f != NULL)
    {
      TEST_COMPARE (fscanf (f, "%d", &ptrace_scope), 1);
      xfclose (f);
    }
#endif

  return ptrace_scope;
}
