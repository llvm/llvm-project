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

#include <errno.h>
#include <libintl.h>
#include <stdio.h>
#include <string.h>
#include <mach/error.h>
#include <errorlib.h>
#include <sys/param.h>
#include <_itoa.h>

/* It is critical here that we always use the `dcgettext' function for
   the message translation.  Since <libintl.h> only defines the macro
   `dgettext' to use `dcgettext' for optimizing programs this is not
   always guaranteed.  */
#ifndef dgettext
# include <locale.h>		/* We need LC_MESSAGES.  */
# define dgettext(domainname, msgid) dcgettext (domainname, msgid, LC_MESSAGES)
#endif

/* Fill buf with a string describing the errno code in ERRNUM.  */
int
__xpg_strerror_r (int errnum, char *buf, size_t buflen)
{
  int system;
  int sub;
  int code;
  const struct error_system *es;
  extern void __mach_error_map_compat (int *);
  const char *estr;

  __mach_error_map_compat (&errnum);

  system = err_get_system (errnum);
  sub = err_get_sub (errnum);
  code = err_get_code (errnum);

  if (system > err_max_system || ! __mach_error_systems[system].bad_sub)
    return EINVAL;

  es = &__mach_error_systems[system];

  if (sub >= es->max_sub)
    estr = (const char *) es->bad_sub;
  else if (code >= es->subsystem[sub].max_code)
    return EINVAL;
  else
    estr = (const char *) _(es->subsystem[sub].codes[code]);

  size_t estrlen = strlen (estr) + 1;

  if (buflen < estrlen)
    return ERANGE;

  memcpy (buf, estr, estrlen);
  return 0;
}
