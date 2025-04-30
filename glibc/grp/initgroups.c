/* Copyright (C) 1989, 1991-2021 Free Software Foundation, Inc.
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
#include <errno.h>
#include <grp.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/param.h>
#include <sys/types.h>
#include <nsswitch.h>
#include <scratch_buffer.h>
#include <config.h>

#include "../nscd/nscd-client.h"
#include "../nscd/nscd_proto.h"

/* Type of the lookup function.  */
typedef enum nss_status (*initgroups_dyn_function) (const char *, gid_t,
						    long int *, long int *,
						    gid_t **, long int, int *);

static bool use_initgroups_entry;


#include "compat-initgroups.c"


static int
internal_getgrouplist (const char *user, gid_t group, long int *size,
		       gid_t **groupsp, long int limit)
{
#ifdef USE_NSCD
  if (__nss_not_use_nscd_group > 0
      && ++__nss_not_use_nscd_group > NSS_NSCD_RETRY)
    __nss_not_use_nscd_group = 0;
  if (!__nss_not_use_nscd_group
      && !__nss_database_custom[NSS_DBSIDX_group])
    {
      int n = __nscd_getgrouplist (user, group, size, groupsp, limit);
      if (n >= 0)
	return n;

      /* nscd is not usable.  */
      __nss_not_use_nscd_group = 1;
    }
#endif

  enum nss_status status = NSS_STATUS_UNAVAIL;

  /* Never store more than the starting *SIZE number of elements.  */
  assert (*size > 0);
  (*groupsp)[0] = group;
  /* Start is one, because we have the first group as parameter.  */
  long int start = 1;

  nss_action_list nip;

  if (__nss_database_get (nss_database_initgroups, &nip)
      && nip != NULL)
    {
      use_initgroups_entry = true;
    }
  else if (__nss_database_get (nss_database_group, &nip)
	   && nip != NULL)
    {
      use_initgroups_entry = false;
    }
  else
    {
      nip = __nss_action_parse ("files");
      use_initgroups_entry = false;
    }

  while (nip && nip->module)
    {
      long int prev_start = start;

      initgroups_dyn_function fct = __nss_lookup_function (nip,
							   "initgroups_dyn");
      if (fct == NULL)
	status = compat_call (nip, user, group, &start, size, groupsp,
			      limit, &errno);
      else
	status = DL_CALL_FCT (fct, (user, group, &start, size, groupsp,
				    limit, &errno));

      /* Remove duplicates.  */
      long int cnt = prev_start;
      while (cnt < start)
	{
	  long int inner;
	  for (inner = 0; inner < prev_start; ++inner)
	    if ((*groupsp)[inner] == (*groupsp)[cnt])
	      break;

	  if (inner < prev_start)
	    (*groupsp)[cnt] = (*groupsp)[--start];
	  else
	    ++cnt;
	}

      /* This is really only for debugging.  */
      if (NSS_STATUS_TRYAGAIN > status || status > NSS_STATUS_RETURN)
	__libc_fatal ("Illegal status in internal_getgrouplist.\n");

      /* For compatibility reason we will continue to look for more
	 entries using the next service even though data has already
	 been found if the nsswitch.conf file contained only a 'groups'
	 line and no 'initgroups' line.  If the latter is available
	 we always respect the status.  This means that the default
	 for successful lookups is to return.  */
      if ((use_initgroups_entry || status != NSS_STATUS_SUCCESS)
	  && nss_next_action (nip, status) == NSS_ACTION_RETURN)
	 break;

      nip++;
    }

  return start;
}

/* Store at most *NGROUPS members of the group set for USER into
   *GROUPS.  Also include GROUP.  The actual number of groups found is
   returned in *NGROUPS.  Return -1 if the if *NGROUPS is too small.  */
int
getgrouplist (const char *user, gid_t group, gid_t *groups, int *ngroups)
{
  long int size = MAX (1, *ngroups);

  gid_t *newgroups = (gid_t *) malloc (size * sizeof (gid_t));
  if (__glibc_unlikely (newgroups == NULL))
    /* No more memory.  */
    // XXX This is wrong.  The user provided memory, we have to use
    // XXX it.  The internal functions must be called with the user
    // XXX provided buffer and not try to increase the size if it is
    // XXX too small.  For initgroups a flag could say: increase size.
    return -1;

  int total = internal_getgrouplist (user, group, &size, &newgroups, -1);

  memcpy (groups, newgroups, MIN (*ngroups, total) * sizeof (gid_t));

  free (newgroups);

  int retval = total > *ngroups ? -1 : total;
  *ngroups = total;

  return retval;
}

nss_interface_function (getgrouplist)

/* Initialize the group set for the current user
   by reading the group database and using all groups
   of which USER is a member.  Also include GROUP.  */
int
initgroups (const char *user, gid_t group)
{
#if defined NGROUPS_MAX && NGROUPS_MAX == 0

  /* No extra groups allowed.  */
  return 0;

#else

  long int size;
  gid_t *groups;
  int ngroups;
  int result;

 /* We always use sysconf even if NGROUPS_MAX is defined.  That way, the
     limit can be raised in the kernel configuration without having to
     recompile libc.  */
  long int limit = __sysconf (_SC_NGROUPS_MAX);

  if (limit > 0)
    /* We limit the size of the intially allocated array.  */
    size = MIN (limit, 64);
  else
    /* No fixed limit on groups.  Pick a starting buffer size.  */
    size = 16;

  groups = (gid_t *) malloc (size * sizeof (gid_t));
  if (__glibc_unlikely (groups == NULL))
    /* No more memory.  */
    return -1;

  ngroups = internal_getgrouplist (user, group, &size, &groups, limit);

  /* Try to set the maximum number of groups the kernel can handle.  */
  do
    result = setgroups (ngroups, groups);
  while (result == -1 && errno == EINVAL && --ngroups > 0);

  free (groups);

  return result;
#endif
}

nss_interface_function (initgroups)
