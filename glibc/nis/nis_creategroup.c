/* Copyright (c) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Thorsten Kukuk <kukuk@suse.de>, 1997.

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

#include <time.h>
#include <string.h>
#include <rpcsvc/nis.h>
#include <shlib-compat.h>

nis_error
nis_creategroup (const_nis_name group, unsigned int flags)
{
  if (group != NULL && group[0] != '\0')
    {
      size_t grouplen = strlen (group);
      char buf[grouplen + 50];
      char leafbuf[grouplen + 2];
      char domainbuf[grouplen + 2];
      nis_error status;
      nis_result *res;
      char *cp, *cp2;
      nis_object *obj;

      cp = stpcpy (buf, nis_leaf_of_r (group, leafbuf, sizeof (leafbuf) - 1));
      cp = stpcpy (cp, ".groups_dir");
      cp2 = nis_domain_of_r (group, domainbuf, sizeof (domainbuf) - 1);
      if (cp2 != NULL && cp2[0] != '\0')
        {
	  *cp++ = '.';
          stpcpy (cp, cp2);
        }
      else
	return NIS_BADNAME;

      obj = calloc (1, sizeof (nis_object));
      if (__glibc_unlikely (obj == NULL))
	return NIS_NOMEMORY;

      obj->zo_oid.ctime = obj->zo_oid.mtime = time (NULL);
      obj->zo_name = strdup (leafbuf);
      obj->zo_owner = __nis_default_owner (NULL);
      obj->zo_group = __nis_default_group (NULL);
      obj->zo_domain = strdup (domainbuf);
      if (obj->zo_name == NULL || obj->zo_owner == NULL
	  || obj->zo_group == NULL || obj->zo_domain == NULL)
	{
	  free (obj->zo_group);
	  free (obj->zo_owner);
	  free (obj->zo_name);
	  free (obj);
	  return NIS_NOMEMORY;
	}
      obj->zo_access = __nis_default_access (NULL, 0);
      obj->zo_ttl = 60 * 60;
      obj->zo_data.zo_type = NIS_GROUP_OBJ;
      obj->zo_data.objdata_u.gr_data.gr_flags = flags;
      obj->zo_data.objdata_u.gr_data.gr_members.gr_members_len = 0;
      obj->zo_data.objdata_u.gr_data.gr_members.gr_members_val = NULL;

      res = nis_add (buf, obj);
      nis_free_object (obj);
      if (res == NULL)
	return NIS_NOMEMORY;
      status = NIS_RES_STATUS (res);
      nis_freeresult (res);

      return status;
    }
  return NIS_FAIL;
}
libnsl_hidden_nolink_def (nis_creategroup, GLIBC_2_1)
