/* Copyright (C) 2011-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gmail.com>, 2011.

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

#include <alloca.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <not-cancel.h>

#include "nscd-client.h"
#include "nscd_proto.h"

int __nss_not_use_nscd_netgroup;


libc_locked_map_ptr (static, map_handle);
/* Note that we only free the structure if necessary.  The memory
   mapping is not removed since it is not visible to the malloc
   handling.  */
libc_freeres_fn (pw_map_free)
{
  if (map_handle.mapped != NO_MAPPING)
    {
      void *p = map_handle.mapped;
      map_handle.mapped = NO_MAPPING;
      free (p);
    }
}


int
__nscd_setnetgrent (const char *group, struct __netgrent *datap)
{
  int gc_cycle;
  int nretries = 0;
  size_t group_len = strlen (group) + 1;

  /* If the mapping is available, try to search there instead of
     communicating with the nscd.  */
  struct mapped_database *mapped;
  mapped = __nscd_get_map_ref (GETFDNETGR, "netgroup", &map_handle, &gc_cycle);

 retry:;
  char *respdata = NULL;
  int retval = -1;
  netgroup_response_header netgroup_resp;

  if (mapped != NO_MAPPING)
    {
      struct datahead *found = __nscd_cache_search (GETNETGRENT, group,
						    group_len, mapped,
						    sizeof netgroup_resp);
      if (found != NULL)
	{
	  respdata = (char *) (&found->data[0].netgroupdata + 1);
	  netgroup_resp = found->data[0].netgroupdata;
	  /* Now check if we can trust pw_resp fields.  If GC is
	     in progress, it can contain anything.  */
	  if (mapped->head->gc_cycle != gc_cycle)
	    {
	      retval = -2;
	      goto out;
	    }
	}
    }

  int sock = -1;
  if (respdata == NULL)
    {
      sock = __nscd_open_socket (group, group_len, GETNETGRENT,
				 &netgroup_resp, sizeof (netgroup_resp));
      if (sock == -1)
	{
	  /* nscd not running or wrong version.  */
	  __nss_not_use_nscd_netgroup = 1;
	  goto out;
	}
    }

  if (netgroup_resp.found == 1)
    {
      size_t datalen = netgroup_resp.result_len;

      /* If we do not have to read the data here it comes from the
	 mapped data and does not have to be freed.  */
      if (respdata == NULL)
	{
	  /* The data will come via the socket.  */
	  respdata = malloc (datalen);
	  if (respdata == NULL)
	    goto out_close;

	  if ((size_t) __readall (sock, respdata, datalen) != datalen)
	    {
	      free (respdata);
	      goto out_close;
	    }
	}

      datap->data = respdata;
      datap->data_size = datalen;
      datap->cursor = respdata;
      datap->first = 1;
      datap->nip = (nss_action_list) -1l;
      datap->known_groups = NULL;
      datap->needed_groups = NULL;

      retval = 1;
    }
  else
    {
      if (__glibc_unlikely (netgroup_resp.found == -1))
	{
	  /* The daemon does not cache this database.  */
	  __nss_not_use_nscd_netgroup = 1;
	  goto out_close;
	}

      /* Set errno to 0 to indicate no error, just no found record.  */
      __set_errno (0);
      /* Even though we have not found anything, the result is zero.  */
      retval = 0;
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


int
__nscd_innetgr (const char *netgroup, const char *host, const char *user,
		const char *domain)
{
  size_t key_len = (strlen (netgroup) + strlen (host ?: "")
		    + strlen (user ?: "") + strlen (domain ?: "") + 7);
  char *key;
  bool use_alloca = __libc_use_alloca (key_len);
  if (use_alloca)
    key = alloca (key_len);
  else
    {
      key = malloc (key_len);
      if (key == NULL)
	return -1;
    }
  char *wp = stpcpy (key, netgroup) + 1;
  if (host != NULL)
    {
      *wp++ = '\1';
      wp = stpcpy (wp, host) + 1;
    }
  else
    *wp++ = '\0';
  if (user != NULL)
    {
      *wp++ = '\1';
      wp = stpcpy (wp, user) + 1;
    }
  else
    *wp++ = '\0';
  if (domain != NULL)
    {
      *wp++ = '\1';
      wp = stpcpy (wp, domain) + 1;
    }
  else
    *wp++ = '\0';
  key_len = wp - key;

  /* If the mapping is available, try to search there instead of
     communicating with the nscd.  */
  int gc_cycle;
  int nretries = 0;
  struct mapped_database *mapped;
  mapped = __nscd_get_map_ref (GETFDNETGR, "netgroup", &map_handle, &gc_cycle);

 retry:;
  int retval = -1;
  innetgroup_response_header innetgroup_resp;
  int sock = -1;

  if (mapped != NO_MAPPING)
    {
      struct datahead *found = __nscd_cache_search (INNETGR, key,
						    key_len, mapped,
						    sizeof innetgroup_resp);
      if (found != NULL)
	{
	  innetgroup_resp = found->data[0].innetgroupdata;
	  /* Now check if we can trust pw_resp fields.  If GC is
	     in progress, it can contain anything.  */
	  if (mapped->head->gc_cycle != gc_cycle)
	    {
	      retval = -2;
	      goto out;
	    }

	  goto found_entry;
	}
    }

  sock = __nscd_open_socket (key, key_len, INNETGR,
			     &innetgroup_resp, sizeof (innetgroup_resp));
  if (sock == -1)
    {
      /* nscd not running or wrong version.  */
      __nss_not_use_nscd_netgroup = 1;
      goto out;
    }

 found_entry:
  if (innetgroup_resp.found == 1)
    retval = innetgroup_resp.result;
  else
    {
      if (__glibc_unlikely (innetgroup_resp.found == -1))
	{
	  /* The daemon does not cache this database.  */
	  __nss_not_use_nscd_netgroup = 1;
	  goto out_close;
	}

      /* Set errno to 0 to indicate no error, just no found record.  */
      __set_errno (0);
      /* Even though we have not found anything, the result is zero.  */
      retval = 0;
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

  if (! use_alloca)
    free (key);

  return retval;
}
