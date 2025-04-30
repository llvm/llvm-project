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

#include <unistd.h>
#include <stdarg.h>
#include <errno.h>
#include <sys/param.h>
#include <stddef.h>

/* Execute PATH with all arguments after PATH until a NULL pointer,
   and the argument after that for environment.  */
int
execle (const char *path, const char *arg, ...)
{
  ptrdiff_t argc;
  va_list ap;
  va_start (ap, arg);
  for (argc = 1; va_arg (ap, const char *); argc++)
    {
      if (argc == INT_MAX)
	{
	  va_end (ap);
	  errno = E2BIG;
	  return -1;
	}
    }
  va_end (ap);

  /* Avoid dynamic memory allocation due two main issues:
     1. The function should be async-signal-safe and a running on a signal
        handler with a fail outcome might lead to malloc bad state.
     2. It might be used in a vfork/clone(VFORK) scenario where using
        malloc also might lead to internal bad state.  */
  ptrdiff_t i;
  char *argv[argc + 1];
  char **envp;
  va_start (ap, arg);
  argv[0] = (char *) arg;
  for (i = 1; i <= argc; i++)
    argv[i] = va_arg (ap, char *);
  envp = va_arg (ap, char **);
  va_end (ap);

  return __execve (path, argv, envp);
}
libc_hidden_def (execle)
