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
#include <netdb.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <not-cancel.h>

#include "nscd-client.h"
#include "nscd_proto.h"


/* Define in nscd_gethst_r.c.  */
extern int __nss_not_use_nscd_hosts;


/* We use the mapping from nscd_gethst.  */
libc_locked_map_ptr (extern, __hst_map_handle) attribute_hidden;

/* Defined in nscd_gethst_r.c.  */
extern int __nss_have_localdomain attribute_hidden;


int
__nscd_getai (const char *key, struct nscd_ai_result **result, int *h_errnop)
{
  if (__glibc_unlikely (__nss_have_localdomain >= 0))
    {
      if (__nss_have_localdomain == 0)
	__nss_have_localdomain = getenv ("LOCALDOMAIN") != NULL ? 1 : -1;
      if (__nss_have_localdomain > 0)
	{
	  __nss_not_use_nscd_hosts = 1;
	  return -1;
	}
    }

  size_t keylen = strlen (key) + 1;
  int gc_cycle;
  int nretries = 0;

  /* If the mapping is available, try to search there instead of
     communicating with the nscd.  */
  struct mapped_database *mapped;
  mapped = __nscd_get_map_ref (GETFDHST, "hosts", &__hst_map_handle,
			       &gc_cycle);

 retry:;
  struct nscd_ai_result *resultbuf = NULL;
  const char *recend = (const char *) ~UINTMAX_C (0);
  char *respdata = NULL;
  int retval = -1;
  int sock = -1;
  ai_response_header ai_resp;

  if (mapped != NO_MAPPING)
    {
      struct datahead *found = __nscd_cache_search (GETAI, key, keylen,
						    mapped, sizeof ai_resp);
      if (found != NULL)
	{
	  respdata = (char *) (&found->data[0].aidata + 1);
	  ai_resp = found->data[0].aidata;
	  recend = (const char *) found->data + found->recsize;
	  /* Now check if we can trust ai_resp fields.  If GC is
	     in progress, it can contain anything.  */
	  if (mapped->head->gc_cycle != gc_cycle)
	    {
	      retval = -2;
	      goto out;
	    }
	}
    }

  /* If we do not have the cache mapped, try to get the data over the
     socket.  */
  if (respdata == NULL)
    {
      sock = __nscd_open_socket (key, keylen, GETAI, &ai_resp,
				 sizeof (ai_resp));
      if (sock == -1)
	{
	  /* nscd not running or wrong version.  */
	  __nss_not_use_nscd_hosts = 1;
	  goto out;
	}
    }

  if (ai_resp.found == 1)
    {
      size_t datalen = ai_resp.naddrs + ai_resp.addrslen + ai_resp.canonlen;

      /* This check really only affects the case where the data
	 comes from the mapped cache.  */
      if (respdata + datalen > recend)
	{
	  assert (sock == -1);
	  goto out;
	}

      /* Create result.  */
      resultbuf = (struct nscd_ai_result *) malloc (sizeof (*resultbuf)
						    + datalen);
      if (resultbuf == NULL)
	{
	  *h_errnop = NETDB_INTERNAL;
	  goto out_close;
	}

      /* Set up the data structure, including pointers.  */
      resultbuf->naddrs = ai_resp.naddrs;
      resultbuf->addrs = (char *) (resultbuf + 1);
      resultbuf->family = (uint8_t *) (resultbuf->addrs + ai_resp.addrslen);
      if (ai_resp.canonlen != 0)
	resultbuf->canon = (char *) (resultbuf->family + resultbuf->naddrs);
      else
	resultbuf->canon = NULL;

      if (respdata == NULL)
	{
	  /* Read the data from the socket.  */
	  if ((size_t) __readall (sock, resultbuf + 1, datalen) == datalen)
	    {
	      retval = 0;
	      *result = resultbuf;
	    }
	  else
	    {
	      free (resultbuf);
	      *h_errnop = NETDB_INTERNAL;
	    }
	}
      else
	{
	  /* Copy the data in the block.  */
	  memcpy (resultbuf + 1, respdata, datalen);

	  /* Try to detect corrupt databases.  */
	  if (resultbuf->canon != NULL
	      && resultbuf->canon[ai_resp.canonlen - 1] != '\0')
	    /* We cannot use the database.  */
	    {
	      if (mapped->head->gc_cycle != gc_cycle)
		retval = -2;
	      else
		free (resultbuf);
	      goto out_close;
	    }

	  retval = 0;
	  *result = resultbuf;
	}
    }
  else
    {
      if (__glibc_unlikely (ai_resp.found == -1))
	{
	  /* The daemon does not cache this database.  */
	  __nss_not_use_nscd_hosts = 1;
	  goto out_close;
	}

      /* Store the error number.  */
      *h_errnop = ai_resp.error;

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
	{
	  *result = NULL;
	  free (resultbuf);
	  goto retry;
	}
    }

  return retval;
}
