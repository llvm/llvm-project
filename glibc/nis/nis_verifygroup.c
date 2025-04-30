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

#include <string.h>
#include <rpcsvc/nis.h>
#include <shlib-compat.h>

nis_error
nis_verifygroup (const_nis_name group)
{
  if (group != NULL && group[0] != '\0')
    {
      size_t grouplen = strlen (group);
      char buf[grouplen + 50];
      char leafbuf[grouplen + 2];
      char domainbuf[grouplen + 2];
      nis_result *res;
      nis_error status;
      char *cp, *cp2;

      cp = stpcpy (buf, nis_leaf_of_r (group, leafbuf, sizeof (leafbuf) - 1));
      cp = stpcpy (cp, ".groups_dir");
      cp2 = nis_domain_of_r (group, domainbuf, sizeof (domainbuf) - 1);
      if (cp2 != NULL && cp2[0] != '\0')
	{
	  *cp++ = '.';
	  stpcpy (cp, cp2);
	}
      res = nis_lookup (buf, 0);
      status = NIS_RES_STATUS (res);
      nis_freeresult (res);
      return status;
    }
  else
    return NIS_FAIL;
}
libnsl_hidden_nolink_def (nis_verifygroup, GLIBC_2_1)
