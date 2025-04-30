/* Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <libc-lock.h>
#include "nisdomain.h"

#define MAXDOMAINNAMELEN 1024

static char domainname[MAXDOMAINNAMELEN];

__libc_lock_define_initialized (static, domainname_lock)

int
__nss_get_default_domain (char **outdomain)
{
  int result = 0;
  *outdomain = NULL;

  __libc_lock_lock (domainname_lock);

  if (domainname[0] != '\0')
    {
      if (getdomainname (domainname, MAXDOMAINNAMELEN) < 0)
	result = errno;
      else if (strcmp (domainname, "(none)") == 0)
	{
	  /* If domainname is not set, some systems will return "(none)" */
	  domainname[0] = '\0';
	  result = ENOENT;
	}
      else
	*outdomain = domainname;
    }
  else
    *outdomain = domainname;

  __libc_lock_unlock (domainname_lock);

  return result;
}
