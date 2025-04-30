/* uname -- Report basic information about the system.  Generic version.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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
#include <string.h>
#include <sys/utsname.h>
#include <unistd.h>

/* This file defines UNAME_* to string constants.  */
#include <uname-values.h>

/* Put information about the system in NAME.  */
int
__uname (struct utsname *name)
{
  int save;

  if (name == NULL)
    {
      __set_errno (EINVAL);
      return -1;
    }

  save = errno;
  if (__gethostname (name->nodename, sizeof (name->nodename)) < 0)
    {
      if (errno == ENOSYS)
	{
	  /* Hostname is meaningless for this machine.  */
	  name->nodename[0] = '\0';
	  __set_errno (save);
	}
#ifdef	ENAMETOOLONG
      else if (errno == ENAMETOOLONG)
	/* The name was truncated.  */
	__set_errno (save);
#endif
      else
	return -1;
    }
  strncpy (name->sysname, UNAME_SYSNAME, sizeof (name->sysname));
  strncpy (name->release, UNAME_RELEASE, sizeof (name->release));
  strncpy (name->version, UNAME_VERSION, sizeof (name->version));
  strncpy (name->machine, UNAME_MACHINE, sizeof (name->machine));

  return 0;
}
weak_alias (__uname, uname)
libc_hidden_def (__uname)
libc_hidden_def (uname)
