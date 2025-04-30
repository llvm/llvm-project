/* 4.4BSD utility functions for error messages.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#include <stdarg.h>
#include <err.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>

#include <wchar.h>
#define flockfile(s) _IO_flockfile (s)
#define funlockfile(s) _IO_funlockfile (s)

extern char *__progname;

#define VA(call)							      \
{									      \
  va_list ap;								      \
  va_start (ap, format);						      \
  call;									      \
  va_end (ap);								      \
}

void
__vwarnx_internal (const char *format, __gnuc_va_list ap,
		   unsigned int mode_flags)
{
  flockfile (stderr);
  __fxprintf (stderr, "%s: ", __progname);
  if (format != NULL)
    __vfxprintf (stderr, format, ap, mode_flags);
  __fxprintf (stderr, "\n");
  funlockfile (stderr);
}

void
__vwarn_internal (const char *format, __gnuc_va_list ap,
		   unsigned int mode_flags)
{
  int error = errno;

  flockfile (stderr);
  if (format != NULL)
    {
      __fxprintf (stderr, "%s: ", __progname);
      __vfxprintf (stderr, format, ap, mode_flags);
      __set_errno (error);
      __fxprintf (stderr, ": %m\n");
    }
  else
    {
      __set_errno (error);
      __fxprintf (stderr, "%s: %m\n", __progname);
    }
  funlockfile (stderr);
}

void
vwarn (const char *format, __gnuc_va_list ap)
{
  __vwarn_internal (format, ap, 0);
}
libc_hidden_def (vwarn)

void
vwarnx (const char *format, __gnuc_va_list ap)
{
  __vwarnx_internal (format, ap, 0);
}
libc_hidden_def (vwarnx)

void
warn (const char *format, ...)
{
  VA (vwarn (format, ap))
}
libc_hidden_def (warn)

void
warnx (const char *format, ...)
{
  VA (vwarnx (format, ap))
}
libc_hidden_def (warnx)

void
verr (int status, const char *format, __gnuc_va_list ap)
{
  vwarn (format, ap);
  exit (status);
}
libc_hidden_def (verr)

void
verrx (int status, const char *format, __gnuc_va_list ap)
{
  vwarnx (format, ap);
  exit (status);
}
libc_hidden_def (verrx)

void
err (int status, const char *format, ...)
{
  VA (verr (status, format, ap))
}

void
errx (int status, const char *format, ...)
{
  VA (verrx (status, format, ap))
}
