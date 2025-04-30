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

#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <libintl.h>
#include <rpcsvc/nis.h>
#include <shlib-compat.h>

nis_name
nis_local_group (void)
{
  static char __nisgroup[NIS_MAXNAMELEN + 1];

  char *cptr;
  if (__nisgroup[0] == '\0'
      && (cptr = getenv ("NIS_GROUP")) != NULL
      && cptr[0] != '\0'
      && strlen (cptr) < NIS_MAXNAMELEN)
    {
      char *cp = stpcpy (__nisgroup, cptr);

      if (cp[-1] != '.')
	{
	  cptr = nis_local_directory ();
	  if ((cp - __nisgroup) + strlen (cptr) + 1 < NIS_MAXNAMELEN)
	    {
	      *cp++ = '.';
	      strcpy (cp, cptr);
	    }
	  else
	    __nisgroup[0] = '\0';
	}
    }

  return __nisgroup;
}
libnsl_hidden_nolink_def (nis_local_group, GLIBC_2_1)

nis_name
nis_local_directory (void)
{
  static char __nisdomainname[NIS_MAXNAMELEN + 1];

  if (__nisdomainname[0] == '\0')
    {
      if (getdomainname (__nisdomainname, NIS_MAXNAMELEN) < 0)
	__nisdomainname[0] = '\0';
      else
	{
	  char *cp = rawmemchr (__nisdomainname, '\0');

	  /* Missing trailing dot? */
	  if (cp[-1] != '.')
	    {
	      *cp++ = '.';
	      *cp = '\0';
	    }
	}
    }

  return __nisdomainname;
}
libnsl_hidden_nolink_def (nis_local_directory, GLIBC_2_1)

nis_name
nis_local_principal (void)
{
  static char __principal[NIS_MAXNAMELEN + 1];

  if (__principal[0] == '\0')
    {
      char buf[NIS_MAXNAMELEN + 1];
      nis_result *res;
      uid_t uid = geteuid ();

      if (uid != 0)
	{
	  int len = snprintf (buf, NIS_MAXNAMELEN - 1,
			      "[auth_name=%d,auth_type=LOCAL],cred.org_dir.%s",
			      uid, nis_local_directory ());

	  if (len >= NIS_MAXNAMELEN - 1)
	    nobody:
	    /* XXX The buffer is too small.  Can this happen???  */
	    return strcpy (__principal, "nobody");

	  if (buf[len - 1] != '.')
	    {
	      buf[len++] = '.';
	      buf[len] = '\0';
	    }

	  res = nis_list (buf, USE_DGRAM + NO_AUTHINFO + FOLLOW_LINKS
			  + FOLLOW_PATH, NULL, NULL);

	  if (res == NULL)
	    goto nobody;

	  if (NIS_RES_STATUS (res) == NIS_SUCCESS)
	    {
	      if (res->objects.objects_len > 1)
		{
		  /* More than one principal with same uid?  something
		     wrong with cred table.  Should be unique.  Warn user
		     and continue.  */
		  printf (_("\
LOCAL entry for UID %d in directory %s not unique\n"),
			  uid, nis_local_directory ());
		}
	      strcpy (__principal, ENTRY_VAL (res->objects.objects_val, 0));
	      nis_freeresult (res);
	      return __principal;
	    }
	  else
	    {
	      nis_freeresult (res);
	      goto nobody;
	    }
	}
      else
	return strcpy (__principal, nis_local_host ());

      /* Should be never reached */
      goto nobody;
    }
  return __principal;
}
libnsl_hidden_nolink_def (nis_local_principal, GLIBC_2_1)

nis_name
nis_local_host (void)
{
  static char __nishostname[NIS_MAXNAMELEN + 1];

  if (__nishostname[0] == '\0')
    {
      if (gethostname (__nishostname, NIS_MAXNAMELEN) < 0)
	__nishostname[0] = '\0';
      else
	{
	  char *cp = rawmemchr (__nishostname, '\0');
	  int len = cp - __nishostname;

	  /* Hostname already fully qualified? */
	  if (cp[-1] == '.')
	    return __nishostname;

	  if (len + strlen (nis_local_directory ()) + 1 > NIS_MAXNAMELEN)
	    {
	      __nishostname[0] = '\0';
	      return __nishostname;
	    }

	  *cp++ = '.';
	  strncpy (cp, nis_local_directory (), NIS_MAXNAMELEN - len -1);
	  __nishostname[NIS_MAXNAMELEN] = '\0';
	}
    }

  return __nishostname;
}
libnsl_hidden_nolink_def (nis_local_host, GLIBC_2_1)
