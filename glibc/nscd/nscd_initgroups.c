/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2004.

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
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <not-cancel.h>

#include "nscd-client.h"
#include "nscd_proto.h"


/* We use the same mapping as in nscd_getgr.   */
libc_locked_map_ptr (extern, __gr_map_handle) attribute_hidden;


int
__nscd_getgrouplist (const char *user, gid_t group, long int *size,
		     gid_t **groupsp, long int limit)
{
  size_t userlen = strlen (user) + 1;
  int gc_cycle;
  int nretries = 0;

  /* If the mapping is available, try to search there instead of
     communicating with the nscd.  */
  struct mapped_database *mapped;
  mapped = __nscd_get_map_ref (GETFDGR, "group", &__gr_map_handle, &gc_cycle);

 retry:;
  char *respdata = NULL;
  int retval = -1;
  int sock = -1;
  initgr_response_header initgr_resp;

  if (mapped != NO_MAPPING)
    {
      struct datahead *found = __nscd_cache_search (INITGROUPS, user,
						    userlen, mapped,
						    sizeof initgr_resp);
      if (found != NULL)
	{
	  respdata = (char *) (&found->data[0].initgrdata + 1);
	  initgr_resp = found->data[0].initgrdata;
	  char *recend = (char *) found->data + found->recsize;

	  /* Now check if we can trust initgr_resp fields.  If GC is
	     in progress, it can contain anything.  */
	  if (mapped->head->gc_cycle != gc_cycle)
	    {
	      retval = -2;
	      goto out;
	    }

	  if (respdata + initgr_resp.ngrps * sizeof (int32_t) > recend)
	    goto out;
	}
    }

  /* If we do not have the cache mapped, try to get the data over the
     socket.  */
  if (respdata == NULL)
    {
      sock = __nscd_open_socket (user, userlen, INITGROUPS, &initgr_resp,
				 sizeof (initgr_resp));
      if (sock == -1)
	{
	  /* nscd not running or wrong version.  */
	  __nss_not_use_nscd_group = 1;
	  goto out;
	}
    }

  if (initgr_resp.found == 1)
    {
      /* The following code assumes that gid_t and int32_t are the
	 same size.  This is the case for al existing implementation.
	 If this should change some code needs to be added which
	 doesn't use memcpy but instead copies each array element one
	 by one.  */
      assert (sizeof (int32_t) == sizeof (gid_t));
      assert (initgr_resp.ngrps >= 0);

      /* Make sure we have enough room.  We always count GROUP in even
	 though we might not end up adding it.  */
      if (*size < initgr_resp.ngrps + 1)
	{
	  gid_t *newp = realloc (*groupsp,
				 (initgr_resp.ngrps + 1) * sizeof (gid_t));
	  if (newp == NULL)
	    /* We cannot increase the buffer size.  */
	    goto out_close;

	  *groupsp = newp;
	  *size = initgr_resp.ngrps + 1;
	}

      if (respdata == NULL)
	{
	  /* Read the data from the socket.  */
	  if ((size_t) __readall (sock, *groupsp, initgr_resp.ngrps
						  * sizeof (gid_t))
	      == initgr_resp.ngrps * sizeof (gid_t))
	    retval = initgr_resp.ngrps;
	}
      else
	{
	  /* Just copy the data.  */
	  retval = initgr_resp.ngrps;
	  memcpy (*groupsp, respdata, retval * sizeof (gid_t));
	}
    }
  else
    {
      if (__glibc_unlikely (initgr_resp.found == -1))
	{
	  /* The daemon does not cache this database.  */
	  __nss_not_use_nscd_group = 1;
	  goto out_close;
	}

      /* No group found yet.   */
      retval = 0;

      assert (*size >= 1);
    }

  /* Check whether GROUP is part of the mix.  If not, add it.  */
  if (retval >= 0)
    {
      int cnt;
      for (cnt = 0; cnt < retval; ++cnt)
	if ((*groupsp)[cnt] == group)
	  break;

      if (cnt == retval)
	(*groupsp)[retval++] = group;
    }

 out_close:
  if (sock != -1)
    __close_nocancel_nostatus (sock);
 out:
  if (__nscd_drop_map_ref (mapped, &gc_cycle) != 0)
    {
      /* When we come here this means there has been a GC cycle while we
	 were looking for the data.  This means the data might have been
	 inconsistent.  Retry if possible.  */
      if ((gc_cycle & 1) != 0 || ++nretries == 5 || retval == -1)
	{
	  /* nscd is just running gc now.  Disable using the mapping.  */
	  if (atomic_decrement_val (&mapped->counter) == 0)
	    __nscd_unmap (mapped);
	  mapped = NO_MAPPING;
	}

      if (retval != -1)
	goto retry;
    }

  return retval;
}
