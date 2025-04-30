/* Copyright (C) 1993-2021 Free Software Foundation, Inc.
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
#include <libintl.h>
#include <stdio.h>
#include <string.h>
#include <mach/error.h>
#include <errorlib.h>

/* Return a string describing the errno code in ERRNUM.  */
char *
__strerror_r (int errnum, char *buf, size_t buflen)
{
  int system;
  int sub;
  int code;
  const struct error_system *es;
  extern void __mach_error_map_compat (int *);

  __mach_error_map_compat (&errnum);

  system = err_get_system (errnum);
  sub = err_get_sub (errnum);
  code = err_get_code (errnum);

  if (system > err_max_system || ! __mach_error_systems[system].bad_sub)
    {
      __snprintf (buf, buflen, "%s: %d", _("Error in unknown error system: "),
		  errnum);
      return buf;
    }

  es = &__mach_error_systems[system];

  if (sub >= es->max_sub)
    return (char *) es->bad_sub;

  if (code >= es->subsystem[sub].max_code)
    {
      __snprintf (buf, buflen, "%s%s %d", _("Unknown error "),
		  es->subsystem[sub].subsys_name, errnum);
      return buf;
    }

  return (char *) _(es->subsystem[sub].codes[code]);
}
libc_hidden_def (__strerror_r)
weak_alias (__strerror_r, strerror_r)
