/* Copyright (c) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Thorsten Kukuk <kukuk@vt.uni-paderborn.de>, 1997.

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
#include <syslog.h>
#include <string.h>
#include <libintl.h>
#include <rpcsvc/nis.h>
#include <shlib-compat.h>


#define MF(line) MF1 (line)
#define MF1(line) str##line
static const union msgstr_t
{
  struct
  {
#define S(s) char MF(__LINE__)[sizeof (s)];
#include "nis_error.h"
#undef S
  };
  char str[0];
} msgstr =
  {
    {
#define S(s) s,
#include "nis_error.h"
#undef S
    }
  };

static const unsigned short int msgidx[] =
  {
#define S(s) offsetof (union msgstr_t, MF (__LINE__)),
#include "nis_error.h"
#undef S
  };


const char *
nis_sperrno (const nis_error status)
{
  if (status >= sizeof (msgidx) / sizeof (msgidx[0]))
    return "???";
  else
    return gettext (msgstr.str + msgidx[status]);
}
libnsl_hidden_nolink_def (nis_sperrno, GLIBC_2_1)

void
nis_perror (const nis_error status, const char *label)
{
  fprintf (stderr, "%s: %s\n", label, nis_sperrno (status));
}
libnsl_hidden_nolink_def (nis_perror, GLIBC_2_1)

void
nis_lerror (const nis_error status, const char *label)
{
  syslog (LOG_ERR, "%s: %s", label, nis_sperrno (status));
}
libnsl_hidden_nolink_def (nis_lerror, GLIBC_2_1)

char *
nis_sperror_r (const nis_error status, const char *label,
	       char *buffer, size_t buflen)
{
  if (snprintf (buffer, buflen, "%s: %s", label, nis_sperrno (status))
      >= buflen)
    {
      __set_errno (ERANGE);
      return NULL;
    }

  return buffer;
}
libnsl_hidden_nolink_def (nis_sperror_r, GLIBC_2_1)

char *
nis_sperror (const nis_error status, const char *label)
{
  static char buffer[NIS_MAXNAMELEN + 1];

  return nis_sperror_r (status, label, buffer, sizeof (buffer));
}
libnsl_hidden_nolink_def (nis_sperror, GLIBC_2_1)
