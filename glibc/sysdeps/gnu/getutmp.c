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

#include <assert.h>
#include <string.h>
#include <utmp.h>
#ifndef _UTMPX_H
/* This is an ugly hack but we must not see the getutmpx declaration.  */
# define getutmpx XXXgetutmpx
# include <utmpx.h>
# undef getutmpx
#endif

void
getutmp (const struct utmpx *utmpx, struct utmp *utmp)
{
  assert (sizeof (struct utmp) == sizeof (struct utmpx));
  memcpy (utmp, utmpx, sizeof (struct utmp));
}
strong_alias (getutmp, getutmpx)
