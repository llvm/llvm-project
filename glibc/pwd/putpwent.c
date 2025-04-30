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
#include <stdio.h>
#include <pwd.h>
#include <stdlib.h>
#include <nss.h>

#define _S(x)	x ?: ""

/* Write an entry to the given stream.  This must know the format of
   the password file.  If the input contains invalid characters,
   return EINVAL, or replace them with spaces (if they are contained
   in the GECOS field).  */
int
putpwent (const struct passwd *p, FILE *stream)
{
  if (p == NULL || stream == NULL
      || p->pw_name == NULL || !__nss_valid_field (p->pw_name)
      || !__nss_valid_field (p->pw_passwd)
      || !__nss_valid_field (p->pw_dir)
      || !__nss_valid_field (p->pw_shell))
    {
      __set_errno (EINVAL);
      return -1;
    }

  int ret;
  char *gecos_alloc;
  const char *gecos = __nss_rewrite_field (p->pw_gecos, &gecos_alloc);

  if (gecos == NULL)
    return -1;

  if (p->pw_name[0] == '+' || p->pw_name[0] == '-')
      ret = fprintf (stream, "%s:%s:::%s:%s:%s\n",
		     p->pw_name, _S (p->pw_passwd),
		     gecos, _S (p->pw_dir), _S (p->pw_shell));
  else
      ret = fprintf (stream, "%s:%s:%lu:%lu:%s:%s:%s\n",
		     p->pw_name, _S (p->pw_passwd),
		     (unsigned long int) p->pw_uid,
		     (unsigned long int) p->pw_gid,
		     gecos, _S (p->pw_dir), _S (p->pw_shell));

  free (gecos_alloc);
  if (ret >= 0)
    ret = 0;
  return ret;
}
