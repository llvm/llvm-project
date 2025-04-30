/* strerror_l - Get errno description string in given locale.  Mach version.
   Copyright (C) 2007-2021 Free Software Foundation, Inc.
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

#include <libintl.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <mach/error.h>
#include <errorlib.h>
#include <tls-internal.h>


static const char *
translate (const char *str, locale_t loc)
{
  locale_t oldloc = __uselocale (loc);
  const char *res = _(str);
  __uselocale (oldloc);
  return res;
}


/* Return a string describing the errno code in ERRNUM.  */
char *
__strerror_l (int errnum, locale_t loc)
{
  int saved_errno = errno;
  char *err;
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
      struct tls_internal_t *tls_internal = __glibc_tls_internal ();
      free (tls_internal->strerror_l_buf);
      if (__asprintf (&tls_internal->strerror_l_buf, "%s%X",
		      translate ("Error in unknown error system: ", loc),
		      errnum) == -1)
	tls_internal->strerror_l_buf = NULL;

      __set_errno (saved_errno);
      return tls_internal->strerror_l_buf;
    }

  es = &__mach_error_systems[system];

  if (sub >= es->max_sub)
    err = (char *) translate (es->bad_sub, loc);
  else if (code >= es->subsystem[sub].max_code)
    {
      struct tls_internal_t *tls_internal = __glibc_tls_internal ();
      free (tls_internal->strerror_l_buf);
      if (__asprintf (&tls_internal->strerror_l_buf, "%s%s %d",
		      translate ("Unknown error ", loc),
		      translate (es->subsystem[sub].subsys_name, loc),
		      errnum) == -1)
	tls_internal->strerror_l_buf = NULL;

      err = tls_internal->strerror_l_buf;
    }
  else
    err = (char *) translate (es->subsystem[sub].codes[code], loc);

  __set_errno (saved_errno);
  return err;
}
weak_alias (__strerror_l, strerror_l)
libc_hidden_def (__strerror_l)
