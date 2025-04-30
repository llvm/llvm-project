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

#include <string.h>
#include <rpcsvc/nis.h>
#include <shlib-compat.h>

/* internal_nis_ismember ()
   return codes: -1 principal is in -group
                  0 principal isn't in any group
		  1 pirncipal is in group */
static int
internal_ismember (const_nis_name principal, const_nis_name group)
{
  size_t grouplen = strlen (group);
  char buf[grouplen + 50];
  char leafbuf[grouplen + 2];
  char domainbuf[grouplen + 2];
  nis_result *res;
  char *cp, *cp2;
  u_int i;

  cp = stpcpy (buf, nis_leaf_of_r (group, leafbuf, sizeof (leafbuf) - 1));
  cp = stpcpy (cp, ".groups_dir");
  cp2 = nis_domain_of_r (group, domainbuf, sizeof (domainbuf) - 1);
  if (cp2 != NULL && cp2[0] != '\0')
    {
      *cp++ = '.';
      strcpy (cp, cp2);
    }

  res = nis_lookup (buf, EXPAND_NAME|FOLLOW_LINKS);
  if (res == NULL || NIS_RES_STATUS (res) != NIS_SUCCESS)
    {
      nis_freeresult (res);
      return 0;
    }

  if ((NIS_RES_NUMOBJ (res) != 1)
      || (__type_of (NIS_RES_OBJECT (res)) != NIS_GROUP_OBJ))
    {
      nis_freeresult (res);
      return 0;
    }

  /* We search twice in the list, at first, if we have the name
     with a "-", then if without. "-member" has priority */
  for (i = 0; i < NIS_RES_OBJECT(res)->GR_data.gr_members.gr_members_len; ++i)
    {
      cp = NIS_RES_OBJECT (res)->GR_data.gr_members.gr_members_val[i];
      if (cp[0] == '-')
	{
	  if (strcmp (&cp[1], principal) == 0)
	    {
	      nis_freeresult (res);
	      return -1;
	    }
	  if (cp[1] == '@')
	    switch (internal_ismember (principal, &cp[2]))
	      {
	      case -1:
		nis_freeresult (res);
		return -1;
	      case 1:
		nis_freeresult (res);
		return 1;
	      default:
		break;
	      }
	  else
	    if (cp[1] == '*')
	      {
		char buf1[strlen (principal) + 2];
		char buf2[strlen (cp) + 2];

		if (strcmp (nis_domain_of_r (principal, buf1, sizeof buf1),
			    nis_domain_of_r (cp, buf2, sizeof buf2)) == 0)
		  {
		    nis_freeresult (res);
		    return -1;
		  }
	      }
	}
    }

  for (i = 0; i < NIS_RES_OBJECT (res)->GR_data.gr_members.gr_members_len; ++i)
    {
      cp = NIS_RES_OBJECT (res)->GR_data.gr_members.gr_members_val[i];
      if (cp[0] != '-')
	{
	  if (strcmp (cp, principal) == 0)
	    {
	      nis_freeresult (res);
	      return 1;
	    }
	  if (cp[0] == '@')
	    switch (internal_ismember (principal, &cp[1]))
	      {
	      case -1:
		nis_freeresult (res);
		return -1;
	      case 1:
		nis_freeresult (res);
		return 1;
	      default:
		break;
	      }
	  else
	    if (cp[0] == '*')
	      {
		char buf1[strlen (principal) + 2];
		char buf2[strlen (cp) + 2];

		if (strcmp (nis_domain_of_r (principal, buf1, sizeof buf1),
			    nis_domain_of_r (cp, buf2, sizeof buf2)) == 0)
		  {
		    nis_freeresult (res);
		    return 1;
		  }
	      }
	}
    }
  nis_freeresult (res);
  return 0;
}

bool_t
nis_ismember (const_nis_name principal, const_nis_name group)
{
  if (group != NULL && group[0] != '\0' && principal != NULL)
    return internal_ismember (principal, group) == 1 ? TRUE : FALSE;
  else
    return FALSE;
}
libnsl_hidden_nolink_def (nis_ismember, GLIBC_2_1)
