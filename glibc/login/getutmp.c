/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
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
#include <utmp.h>
#include <utmpx.h>

/* Copy the information in UTMPX to UTMP. */
void
getutmp (const struct utmpx *utmpx, struct utmp *utmp)
{
  utmp->ut_type = utmpx->ut_type;
  utmp->ut_pid = utmpx->ut_pid;
  memcpy (utmp->ut_line, utmpx->ut_line, sizeof (utmp->ut_line));
  memcpy (utmp->ut_user, utmpx->ut_user, sizeof (utmp->ut_user));
  memcpy (utmp->ut_id, utmpx->ut_id, sizeof (utmp->ut_id));
  memcpy (utmp->ut_host, utmpx->ut_host, sizeof (utmp->ut_host));
  utmp->ut_tv = utmpx->ut_tv;
}
