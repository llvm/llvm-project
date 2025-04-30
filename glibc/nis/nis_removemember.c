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

#include <assert.h>
#include <string.h>
#include <rpcsvc/nis.h>
#include <shlib-compat.h>

nis_error
nis_removemember (const_nis_name member, const_nis_name group)
{
  if (group != NULL && group[0] != '\0')
    {
      size_t grouplen = strlen (group);
      char buf[grouplen + 14 + NIS_MAXNAMELEN];
      char domainbuf[grouplen + 2];
      nis_result *res, *res2;
      nis_error status;
      char *cp, *cp2;

      cp = rawmemchr (nis_leaf_of_r (group, buf, sizeof (buf) - 1), '\0');
      cp = stpcpy (cp, ".groups_dir");
      cp2 = nis_domain_of_r (group, domainbuf, sizeof (domainbuf) - 1);
      if (cp2 != NULL && cp2[0] != '\0')
        {
          cp = stpcpy (cp, ".");
          stpcpy (cp, cp2);
        }
      res = nis_lookup (buf, FOLLOW_LINKS | EXPAND_NAME);
      if (res == NULL)
	return NIS_NOMEMORY;
      if (NIS_RES_STATUS (res) != NIS_SUCCESS)
        {
	  status = NIS_RES_STATUS (res);
	  nis_freeresult (res);
          return status;
        }

      if (NIS_RES_NUMOBJ (res) != 1
	  || __type_of (NIS_RES_OBJECT (res)) != NIS_GROUP_OBJ)
	{
	  nis_freeresult (res);
	  return NIS_INVALIDOBJ;
	}

      nis_name *gr_members_val
	= NIS_RES_OBJECT(res)->GR_data.gr_members.gr_members_val;
      u_int gr_members_len
	= NIS_RES_OBJECT(res)->GR_data.gr_members.gr_members_len;

      u_int j = 0;
      for (u_int i = 0; i < gr_members_len; ++i)
	if (strcmp (gr_members_val[i], member) != 0)
	  gr_members_val[j++] = gr_members_val[i];
	else
	  free (gr_members_val[i]);

      /* There is no need to reallocate the gr_members_val array.  We
	 just adjust the size to match the number of strings still in
	 it.  Yes, xdr_array will use mem_free with a size parameter
	 but this is mapped to a simple free call which determines the
	 size of the block by itself.  */
      NIS_RES_OBJECT (res)->GR_data.gr_members.gr_members_len = j;

      cp = stpcpy (buf, NIS_RES_OBJECT (res)->zo_name);
      *cp++ = '.';
      strncpy (cp, NIS_RES_OBJECT (res)->zo_domain, NIS_MAXNAMELEN);
      res2 = nis_modify (buf, NIS_RES_OBJECT (res));
      status = NIS_RES_STATUS (res2);
      nis_freeresult (res);
      nis_freeresult (res2);

      return status;
    }
  else
    return NIS_FAIL;
}
libnsl_hidden_nolink_def (nis_removemember, GLIBC_2_1)
