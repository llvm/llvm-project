/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
   Contributed by David Mosberger-Tang <davidm@hpl.hp.com>.
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

#include <sys/profil.h>
#include <sys/time.h>

int
__sprofil (struct prof *profp, int profcnt, struct timeval *tvp,
	   unsigned int flags)
{
  if (profcnt == 0)
    return 0;

  __set_errno (ENOSYS);
  return -1;
}

weak_alias (__sprofil, sprofil)
