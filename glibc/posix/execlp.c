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

/* Execute FILE, searching in the `PATH' environment variable if
   it contains no slashes, with all arguments after FILE until a
   NULL pointer and environment from `environ'.  */
int
execlp (const char *file, const char *arg, ...)
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

  /* Although posix does not state execlp as an async-safe function
     it can not use malloc to allocate the arguments since it might
     be used in a vfork scenario and it may lead to malloc internal
     bad state.  */
  ptrdiff_t i;
  char *argv[argc + 1];
  va_start (ap, arg);
  argv[0] = (char *) arg;
  for (i = 1; i <= argc; i++)
    argv[i] = va_arg (ap, char *);
  va_end (ap);

  return __execvpe (file, argv, __environ);
}
libc_hidden_def (execlp)
