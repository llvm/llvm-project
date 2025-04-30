/* Define and initialize `__progname' et. al.
   Copyright (C) 1994-2021 Free Software Foundation, Inc.
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

#include <string.h>
#include <libc-internal.h>

char *__progname_full = (char *) "";
char *__progname = (char *) "";
weak_alias (__progname_full, program_invocation_name)
weak_alias (__progname, program_invocation_short_name)


void
__init_misc (int argc, char **argv, char **envp)
{
  if (argv && argv[0])
    {
      char *p = strrchr (argv[0], '/');
      if (p == NULL)
	__progname = argv[0];
      else
	__progname = p + 1;
      __progname_full = argv[0];
    }
}
