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

#include <errno.h>
#include <nss.h>
#include <stdio.h>
#include <shadow.h>

#define flockfile(s) _IO_flockfile (s)
#define funlockfile(s) _IO_funlockfile (s)

#define _S(x)	x ? x : ""


/* Write an entry to the given stream.
   This must know the format of the password file.  */
int
putspent (const struct spwd *p, FILE *stream)
{
  int errors = 0;

  if (p->sp_namp == NULL || !__nss_valid_field (p->sp_namp)
      || !__nss_valid_field (p->sp_pwdp))
    {
      __set_errno (EINVAL);
      return -1;
    }

  flockfile (stream);

  if (fprintf (stream, "%s:%s:", p->sp_namp, _S (p->sp_pwdp)) < 0)
    ++errors;

  if ((p->sp_lstchg != (long int) -1
       && fprintf (stream, "%ld:", p->sp_lstchg) < 0)
      || (p->sp_lstchg == (long int) -1
	  && putc_unlocked (':', stream) == EOF))
    ++errors;

  if ((p->sp_min != (long int) -1
       && fprintf (stream, "%ld:", p->sp_min) < 0)
      || (p->sp_min == (long int) -1
	  && putc_unlocked (':', stream) == EOF))
    ++errors;

  if ((p->sp_max != (long int) -1
       && fprintf (stream, "%ld:", p->sp_max) < 0)
      || (p->sp_max == (long int) -1
	  && putc_unlocked (':', stream) == EOF))
    ++errors;

  if ((p->sp_warn != (long int) -1
       && fprintf (stream, "%ld:", p->sp_warn) < 0)
      || (p->sp_warn == (long int) -1
	  && putc_unlocked (':', stream) == EOF))
    ++errors;

  if ((p->sp_inact != (long int) -1
       && fprintf (stream, "%ld:", p->sp_inact) < 0)
      || (p->sp_inact == (long int) -1
	  && putc_unlocked (':', stream) == EOF))
    ++errors;

  if ((p->sp_expire != (long int) -1
       && fprintf (stream, "%ld:", p->sp_expire) < 0)
      || (p->sp_expire == (long int) -1
	  && putc_unlocked (':', stream) == EOF))
    ++errors;

  if (p->sp_flag != ~0ul
      && fprintf (stream, "%ld", p->sp_flag) < 0)
    ++errors;

  if (putc_unlocked ('\n', stream) == EOF)
    ++errors;

  funlockfile (stream);

  return errors ? -1 : 0;
}
