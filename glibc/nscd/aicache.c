/* Cache handling for host lookup.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2004.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published
   by the Free Software Foundation; version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.  */

#include <assert.h>
#include <errno.h>
#include <libintl.h>
#include <netdb.h>
#include <nss.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <resolv/resolv-internal.h>
#include <resolv/resolv_context.h>
#include <scratch_buffer.h>

#include "dbg_log.h"
#include "nscd.h"


static const ai_response_header notfound =
{
  .version = NSCD_VERSION,
  .found = 0,
  .naddrs = 0,
  .addrslen = 0,
  .canonlen = 0,
  .error = 0
};


static time_t
addhstaiX (struct database_dyn *db, int fd, request_header *req,
	   void *key, uid_t uid, struct hashentry *const he,
	   struct datahead *dh)
{
  /* Search for the entry matching the key.  Please note that we don't
     look again in the table whether the dataset is now available.  We
     simply insert it.  It does not matter if it is in there twice.  The
     pruning function only will look at the timestamp.  */

  /* We allocate all data in one memory block: the iov vector,
     the response header and the dataset itself.  */
  struct dataset
  {
    struct datahead head;
    ai_response_header resp;
    char strdata[0];
  } *dataset = NULL;

  if (__glibc_unlikely (debug_level > 0))
    {
      if (he == NULL)
	dbg_log (_("Haven't found \"%s\" in hosts cache!"), (char *) key);
      else
	dbg_log (_("Reloading \"%s\" in hosts cache!"), (char *) key);
    }

  nss_action_list nip;
  int no_more;
  int rc6 = 0;
  int rc4 = 0;
  int herrno = 0;

  no_more = !__nss_database_get (nss_database_hosts, &nip);

  /* Initialize configurations.  */
  struct resolv_context *ctx = __resolv_context_get ();
  if (ctx == NULL)
    no_more = 1;

  struct scratch_buffer tmpbuf6;
  scratch_buffer_init (&tmpbuf6);
  struct scratch_buffer tmpbuf4;
  scratch_buffer_init (&tmpbuf4);
  struct scratch_buffer canonbuf;
  scratch_buffer_init (&canonbuf);

  int32_t ttl = INT32_MAX;
  ssize_t total = 0;
  char *key_copy = NULL;
  bool alloca_used = false;
  time_t timeout = MAX_TIMEOUT_VALUE;

  while (!no_more)
    {
      void *cp;
      int status[2] = { NSS_STATUS_UNAVAIL, NSS_STATUS_UNAVAIL };
      int naddrs = 0;
      size_t addrslen = 0;

      char *canon = NULL;
      size_t canonlen;

      nss_gethostbyname4_r *fct4 = __nss_lookup_function (nip,
							  "gethostbyname4_r");
      if (fct4 != NULL)
	{
	  struct gaih_addrtuple atmem;
	  struct gaih_addrtuple *at;
	  while (1)
	    {
	      at = &atmem;
	      rc6 = 0;
	      herrno = 0;
	      status[1] = DL_CALL_FCT (fct4, (key, &at,
					      tmpbuf6.data, tmpbuf6.length,
					      &rc6, &herrno, &ttl));
	      if (rc6 != ERANGE || (herrno != NETDB_INTERNAL
				    && herrno != TRY_AGAIN))
		break;
	      if (!scratch_buffer_grow (&tmpbuf6))
		{
		  rc6 = ENOMEM;
		  break;
		}
	    }

	  if (rc6 != 0 && herrno == NETDB_INTERNAL)
	    goto out;

	  if (status[1] != NSS_STATUS_SUCCESS)
	    goto next_nip;

	  /* We found the data.  Count the addresses and the size.  */
	  for (const struct gaih_addrtuple *at2 = at = &atmem; at2 != NULL;
	       at2 = at2->next)
	    {
	      ++naddrs;
	      /* We do not handle anything other than IPv4 and IPv6
		 addresses.  The getaddrinfo implementation does not
		 either so it is not worth trying to do more.  */
	      if (at2->family == AF_INET)
		addrslen += INADDRSZ;
	      else if (at2->family == AF_INET6)
		addrslen += IN6ADDRSZ;
	    }
	  canon = at->name;
	  canonlen = strlen (canon) + 1;

	  total = sizeof (*dataset) + naddrs + addrslen + canonlen;

	  /* Now we can allocate the data structure.  If the TTL of the
	     entry is reported as zero do not cache the entry at all.  */
	  if (ttl != 0 && he == NULL)
	    dataset = (struct dataset *) mempool_alloc (db, total
							+ req->key_len, 1);

	  if (dataset == NULL)
	    {
	      /* We cannot permanently add the result in the moment.  But
		 we can provide the result as is.  Store the data in some
		 temporary memory.  */
	      dataset = (struct dataset *) alloca (total + req->key_len);

	      /* We cannot add this record to the permanent database.  */
	      alloca_used = true;
	    }

	  /* Fill in the address and address families.  */
	  char *addrs = dataset->strdata;
	  uint8_t *family = (uint8_t *) (addrs + addrslen);

	  for (const struct gaih_addrtuple *at2 = at; at2 != NULL;
	       at2 = at2->next)
	    {
	      *family++ = at2->family;
	      if (at2->family == AF_INET)
		addrs = mempcpy (addrs, at2->addr, INADDRSZ);
	      else if (at2->family == AF_INET6)
		addrs = mempcpy (addrs, at2->addr, IN6ADDRSZ);
	    }

	  cp = family;
	}
      else
	{
	  /* Prefer the function which also returns the TTL and
	     canonical name.  */
	  nss_gethostbyname3_r *fct
	    = __nss_lookup_function (nip, "gethostbyname3_r");
	  if (fct == NULL)
	    fct = __nss_lookup_function (nip, "gethostbyname2_r");

	  if (fct == NULL)
	    goto next_nip;

	  struct hostent th[2];

	  /* Collect IPv6 information first.  */
	  while (1)
	    {
	      rc6 = 0;
	      status[0] = DL_CALL_FCT (fct, (key, AF_INET6, &th[0],
					     tmpbuf6.data, tmpbuf6.length,
					     &rc6, &herrno, &ttl,
					     &canon));
	      if (rc6 != ERANGE || herrno != NETDB_INTERNAL)
		break;
	      if (!scratch_buffer_grow (&tmpbuf6))
		{
		  rc6 = ENOMEM;
		  break;
		}
	    }

	  if (rc6 != 0 && herrno == NETDB_INTERNAL)
	    goto out;

	  /* Next collect IPv4 information.  */
	  while (1)
	    {
	      rc4 = 0;
	      status[1] = DL_CALL_FCT (fct, (key, AF_INET, &th[1],
					     tmpbuf4.data, tmpbuf4.length,
					     &rc4, &herrno,
					     ttl == INT32_MAX ? &ttl : NULL,
					     canon == NULL ? &canon : NULL));
	      if (rc4 != ERANGE || herrno != NETDB_INTERNAL)
		break;
	      if (!scratch_buffer_grow (&tmpbuf4))
		{
		  rc4 = ENOMEM;
		  break;
		}
	    }

	  if (rc4 != 0 && herrno == NETDB_INTERNAL)
	    goto out;

	  if (status[0] != NSS_STATUS_SUCCESS
	      && status[1] != NSS_STATUS_SUCCESS)
	    goto next_nip;

	  /* We found the data.  Count the addresses and the size.  */
	  for (int j = 0; j < 2; ++j)
	    if (status[j] == NSS_STATUS_SUCCESS)
	      for (int i = 0; th[j].h_addr_list[i] != NULL; ++i)
		{
		  ++naddrs;
		  addrslen += th[j].h_length;
		}

	  if (canon == NULL)
	    {
	      /* Determine the canonical name.  */
	      nss_getcanonname_r *cfct;
	      cfct = __nss_lookup_function (nip, "getcanonname_r");
	      if (cfct != NULL)
		{
		  char *s;
		  int rc;

		  if (DL_CALL_FCT (cfct, (key, canonbuf.data, canonbuf.length,
					  &s, &rc, &herrno))
		      == NSS_STATUS_SUCCESS)
		    canon = s;
		  else
		    /* Set to name now to avoid using gethostbyaddr.  */
		    canon = key;
		}
	      else
		{
		  struct hostent *hstent = NULL;
		  int herrno;
		  struct hostent hstent_mem;
		  void *addr;
		  size_t addrlen;
		  int addrfamily;

		  if (status[1] == NSS_STATUS_SUCCESS)
		    {
		      addr = th[1].h_addr_list[0];
		      addrlen = sizeof (struct in_addr);
		      addrfamily = AF_INET;
		    }
		  else
		    {
		      addr = th[0].h_addr_list[0];
		      addrlen = sizeof (struct in6_addr);
		      addrfamily = AF_INET6;
		    }

		  int rc;
		  while (1)
		    {
		      rc = __gethostbyaddr2_r (addr, addrlen, addrfamily,
					       &hstent_mem,
					       canonbuf.data, canonbuf.length,
					       &hstent, &herrno, NULL);
		      if (rc != ERANGE || herrno != NETDB_INTERNAL)
			break;
		      if (!scratch_buffer_grow (&canonbuf))
			{
			  rc = ENOMEM;
			  break;
			}
		    }

		  if (rc == 0)
		    {
		      if (hstent != NULL)
			canon = hstent->h_name;
		      else
			canon = key;
		    }
		}
	    }

	  canonlen = canon == NULL ? 0 : (strlen (canon) + 1);

	  total = sizeof (*dataset) + naddrs + addrslen + canonlen;


	  /* Now we can allocate the data structure.  If the TTL of the
	     entry is reported as zero do not cache the entry at all.  */
	  if (ttl != 0 && he == NULL)
	    dataset = (struct dataset *) mempool_alloc (db, total
							+ req->key_len, 1);

	  if (dataset == NULL)
	    {
	      /* We cannot permanently add the result in the moment.  But
		 we can provide the result as is.  Store the data in some
		 temporary memory.  */
	      dataset = (struct dataset *) alloca (total + req->key_len);

	      /* We cannot add this record to the permanent database.  */
	      alloca_used = true;
	    }

	  /* Fill in the address and address families.  */
	  char *addrs = dataset->strdata;
	  uint8_t *family = (uint8_t *) (addrs + addrslen);

	  for (int j = 0; j < 2; ++j)
	    if (status[j] == NSS_STATUS_SUCCESS)
	      for (int i = 0; th[j].h_addr_list[i] != NULL; ++i)
		{
		  addrs = mempcpy (addrs, th[j].h_addr_list[i],
				   th[j].h_length);
		  *family++ = th[j].h_addrtype;
		}

	  cp = family;
	}

      timeout = datahead_init_pos (&dataset->head, total + req->key_len,
				   total - offsetof (struct dataset, resp),
				   he == NULL ? 0 : dh->nreloads + 1,
				   ttl == INT32_MAX ? db->postimeout : ttl);

      /* Fill in the rest of the dataset.  */
      dataset->resp.version = NSCD_VERSION;
      dataset->resp.found = 1;
      dataset->resp.naddrs = naddrs;
      dataset->resp.addrslen = addrslen;
      dataset->resp.canonlen = canonlen;
      dataset->resp.error = NETDB_SUCCESS;

      if (canon != NULL)
	cp = mempcpy (cp, canon, canonlen);

      key_copy = memcpy (cp, key, req->key_len);

      assert (cp == (char *) dataset + total);

      /* Now we can determine whether on refill we have to create a
	 new record or not.  */
      if (he != NULL)
	{
	  assert (fd == -1);

	  if (total + req->key_len == dh->allocsize
	      && total - offsetof (struct dataset, resp) == dh->recsize
	      && memcmp (&dataset->resp, dh->data,
			 dh->allocsize - offsetof (struct dataset,
						   resp)) == 0)
	    {
	      /* The data has not changed.  We will just bump the
		 timeout value.  Note that the new record has been
		 allocated on the stack and need not be freed.  */
	      dh->timeout = dataset->head.timeout;
	      dh->ttl = dataset->head.ttl;
	      ++dh->nreloads;
	    }
	  else
	    {
	      /* We have to create a new record.  Just allocate
		 appropriate memory and copy it.  */
	      struct dataset *newp
		= (struct dataset *) mempool_alloc (db, total + req->key_len,
						    1);
	      if (__glibc_likely (newp != NULL))
		{
		  /* Adjust pointer into the memory block.  */
		  key_copy = (char *) newp + (key_copy - (char *) dataset);

		  dataset = memcpy (newp, dataset, total + req->key_len);
		  alloca_used = false;
		}

	      /* Mark the old record as obsolete.  */
	      dh->usable = false;
	    }
	}
      else
	{
	  /* We write the dataset before inserting it to the database
	     since while inserting this thread might block and so
	     would unnecessarily let the receiver wait.  */
	  assert (fd != -1);

	  writeall (fd, &dataset->resp, dataset->head.recsize);
	}

      goto out;

next_nip:
      if (nss_next_action (nip, status[1]) == NSS_ACTION_RETURN)
	break;

      if (nip[1].module == NULL)
	no_more = -1;
      else
	++nip;
    }

  /* No result found.  Create a negative result record.  */
  if (he != NULL && rc4 == EAGAIN)
    {
      /* If we have an old record available but cannot find one now
	 because the service is not available we keep the old record
	 and make sure it does not get removed.  */
      if (reload_count != UINT_MAX && dh->nreloads == reload_count)
	/* Do not reset the value if we never not reload the record.  */
	dh->nreloads = reload_count - 1;

      /* Reload with the same time-to-live value.  */
      timeout = dh->timeout = time (NULL) + dh->ttl;
    }
  else
    {
      /* We have no data.  This means we send the standard reply for
	 this case.  */
      total = sizeof (notfound);

      if (fd != -1)
	TEMP_FAILURE_RETRY (send (fd, &notfound, total, MSG_NOSIGNAL));

      /* If we have a transient error or cannot permanently store the
	 result, so be it.  */
      if (rc4 == EAGAIN || __builtin_expect (db->negtimeout == 0, 0))
	{
	  /* Mark the old entry as obsolete.  */
	  if (dh != NULL)
	    dh->usable = false;
	  dataset = NULL;
	}
      else if ((dataset = mempool_alloc (db, (sizeof (struct dataset)
					      + req->key_len), 1)) != NULL)
	{
	  timeout = datahead_init_neg (&dataset->head,
				       sizeof (struct dataset) + req->key_len,
				       total, db->negtimeout);

	  /* This is the reply.  */
	  memcpy (&dataset->resp, &notfound, total);

	  /* Copy the key data.  */
	  key_copy = memcpy (dataset->strdata, key, req->key_len);
	}
   }

 out:
  __resolv_context_put (ctx);

  if (dataset != NULL && !alloca_used)
    {
      /* If necessary, we also propagate the data to disk.  */
      if (db->persistent)
	{
	  // XXX async OK?
	  uintptr_t pval = (uintptr_t) dataset & ~pagesize_m1;
	  msync ((void *) pval,
		 ((uintptr_t) dataset & pagesize_m1) + total + req->key_len,
		 MS_ASYNC);
	}

      (void) cache_add (req->type, key_copy, req->key_len, &dataset->head,
			true, db, uid, he == NULL);

      pthread_rwlock_unlock (&db->lock);

      /* Mark the old entry as obsolete.  */
      if (dh != NULL)
	dh->usable = false;
    }

  scratch_buffer_free (&tmpbuf6);
  scratch_buffer_free (&tmpbuf4);
  scratch_buffer_free (&canonbuf);

  return timeout;
}


void
addhstai (struct database_dyn *db, int fd, request_header *req, void *key,
	  uid_t uid)
{
  addhstaiX (db, fd, req, key, uid, NULL, NULL);
}


time_t
readdhstai (struct database_dyn *db, struct hashentry *he, struct datahead *dh)
{
  request_header req =
    {
      .type = GETAI,
      .key_len = he->len
    };

  return addhstaiX (db, -1, &req, db->data + he->key, he->owner, he, dh);
}
