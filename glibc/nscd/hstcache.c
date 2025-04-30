/* Cache handling for host lookup.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1998.

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

#include <alloca.h>
#include <assert.h>
#include <errno.h>
#include <error.h>
#include <libintl.h>
#include <netdb.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <arpa/nameser.h>
#include <sys/mman.h>
#include <stackinfo.h>
#include <scratch_buffer.h>

#include "nscd.h"
#include "dbg_log.h"


/* This is the standard reply in case the service is disabled.  */
static const hst_response_header disabled =
{
  .version = NSCD_VERSION,
  .found = -1,
  .h_name_len = 0,
  .h_aliases_cnt = 0,
  .h_addrtype = -1,
  .h_length = -1,
  .h_addr_list_cnt = 0,
  .error = NETDB_INTERNAL
};

/* This is the struct describing how to write this record.  */
const struct iovec hst_iov_disabled =
{
  .iov_base = (void *) &disabled,
  .iov_len = sizeof (disabled)
};


/* This is the standard reply in case we haven't found the dataset.  */
static const hst_response_header notfound =
{
  .version = NSCD_VERSION,
  .found = 0,
  .h_name_len = 0,
  .h_aliases_cnt = 0,
  .h_addrtype = -1,
  .h_length = -1,
  .h_addr_list_cnt = 0,
  .error = HOST_NOT_FOUND
};


/* This is the standard reply in case there are temporary problems.  */
static const hst_response_header tryagain =
{
  .version = NSCD_VERSION,
  .found = 0,
  .h_name_len = 0,
  .h_aliases_cnt = 0,
  .h_addrtype = -1,
  .h_length = -1,
  .h_addr_list_cnt = 0,
  .error = TRY_AGAIN
};


static time_t
cache_addhst (struct database_dyn *db, int fd, request_header *req,
	      const void *key, struct hostent *hst, uid_t owner,
	      struct hashentry *const he, struct datahead *dh, int errval,
	      int32_t ttl)
{
  bool all_written = true;
  time_t t = time (NULL);

  /* We allocate all data in one memory block: the iov vector,
     the response header and the dataset itself.  */
  struct dataset
  {
    struct datahead head;
    hst_response_header resp;
    char strdata[0];
  } *dataset;

  assert (offsetof (struct dataset, resp) == offsetof (struct datahead, data));

  time_t timeout = MAX_TIMEOUT_VALUE;
  if (hst == NULL)
    {
      if (he != NULL && errval == EAGAIN)
	{
	  /* If we have an old record available but cannot find one
	     now because the service is not available we keep the old
	     record and make sure it does not get removed.  */
	  if (reload_count != UINT_MAX)
	    /* Do not reset the value if we never not reload the record.  */
	    dh->nreloads = reload_count - 1;

	  /* Reload with the same time-to-live value.  */
	  timeout = dh->timeout = t + dh->ttl;
	}
      else
	{
	  /* We have no data.  This means we send the standard reply for this
	     case.  Possibly this is only temporary.  */
	  ssize_t total = sizeof (notfound);
	  assert (sizeof (notfound) == sizeof (tryagain));

	  const hst_response_header *resp = (errval == EAGAIN
					     ? &tryagain : &notfound);

	  if (fd != -1
	      && TEMP_FAILURE_RETRY (send (fd, resp, total,
					   MSG_NOSIGNAL)) != total)
	    all_written = false;

	  /* If we have a transient error or cannot permanently store
	     the result, so be it.  */
	  if (errval == EAGAIN || __builtin_expect (db->negtimeout == 0, 0))
	    {
	      /* Mark the old entry as obsolete.  */
	      if (dh != NULL)
		dh->usable = false;
	    }
	  else if ((dataset = mempool_alloc (db, (sizeof (struct dataset)
						  + req->key_len), 1)) != NULL)
	    {
	      timeout = datahead_init_neg (&dataset->head,
					   (sizeof (struct dataset)
					    + req->key_len), total,
					   (ttl == INT32_MAX
					    ? db->negtimeout : ttl));

	      /* This is the reply.  */
	      memcpy (&dataset->resp, resp, total);

	      /* Copy the key data.  */
	      memcpy (dataset->strdata, key, req->key_len);

	      /* If necessary, we also propagate the data to disk.  */
	      if (db->persistent)
		{
		  // XXX async OK?
		  uintptr_t pval = (uintptr_t) dataset & ~pagesize_m1;
		  msync ((void *) pval,
			 ((uintptr_t) dataset & pagesize_m1)
			 + sizeof (struct dataset) + req->key_len, MS_ASYNC);
		}

	      (void) cache_add (req->type, &dataset->strdata, req->key_len,
				&dataset->head, true, db, owner, he == NULL);

	      pthread_rwlock_unlock (&db->lock);

	      /* Mark the old entry as obsolete.  */
	      if (dh != NULL)
		dh->usable = false;
	    }
	}
    }
  else
    {
      /* Determine the I/O structure.  */
      size_t h_name_len = strlen (hst->h_name) + 1;
      size_t h_aliases_cnt;
      uint32_t *h_aliases_len;
      size_t h_addr_list_cnt;
      char *addresses;
      char *aliases;
      char *key_copy = NULL;
      char *cp;
      size_t cnt;
      ssize_t total;

      /* Determine the number of aliases.  */
      h_aliases_cnt = 0;
      for (cnt = 0; hst->h_aliases[cnt] != NULL; ++cnt)
	++h_aliases_cnt;
      /* Determine the length of all aliases.  */
      h_aliases_len = (uint32_t *) alloca (h_aliases_cnt * sizeof (uint32_t));
      total = 0;
      for (cnt = 0; cnt < h_aliases_cnt; ++cnt)
	{
	  h_aliases_len[cnt] = strlen (hst->h_aliases[cnt]) + 1;
	  total += h_aliases_len[cnt];
	}

      /* Determine the number of addresses.  */
      h_addr_list_cnt = 0;
      while (hst->h_addr_list[h_addr_list_cnt] != NULL)
	++h_addr_list_cnt;

      if (h_addr_list_cnt == 0)
	/* Invalid entry.  */
	return MAX_TIMEOUT_VALUE;

      total += (sizeof (struct dataset)
		+ h_name_len
		+ h_aliases_cnt * sizeof (uint32_t)
		+ h_addr_list_cnt * hst->h_length);

      /* If we refill the cache, first assume the reconrd did not
	 change.  Allocate memory on the cache since it is likely
	 discarded anyway.  If it turns out to be necessary to have a
	 new record we can still allocate real memory.  */
      bool alloca_used = false;
      dataset = NULL;

      /* If the record contains more than one IP address (used for
	 load balancing etc) don't cache the entry.  This is something
	 the current cache handling cannot handle and it is more than
	 questionable whether it is worthwhile complicating the cache
	 handling just for handling such a special case. */
      if (he == NULL && h_addr_list_cnt == 1)
	dataset = (struct dataset *) mempool_alloc (db, total + req->key_len,
						    1);

      if (dataset == NULL)
	{
	  /* We cannot permanently add the result in the moment.  But
	     we can provide the result as is.  Store the data in some
	     temporary memory.  */
	  dataset = (struct dataset *) alloca (total + req->key_len);

	  /* We cannot add this record to the permanent database.  */
	  alloca_used = true;
	}

      timeout = datahead_init_pos (&dataset->head, total + req->key_len,
				   total - offsetof (struct dataset, resp),
				   he == NULL ? 0 : dh->nreloads + 1,
				   ttl == INT32_MAX ? db->postimeout : ttl);

      dataset->resp.version = NSCD_VERSION;
      dataset->resp.found = 1;
      dataset->resp.h_name_len = h_name_len;
      dataset->resp.h_aliases_cnt = h_aliases_cnt;
      dataset->resp.h_addrtype = hst->h_addrtype;
      dataset->resp.h_length = hst->h_length;
      dataset->resp.h_addr_list_cnt = h_addr_list_cnt;
      dataset->resp.error = NETDB_SUCCESS;

      /* Make sure there is no gap.  */
      assert ((char *) (&dataset->resp.error + 1) == dataset->strdata);

      cp = dataset->strdata;

      cp = mempcpy (cp, hst->h_name, h_name_len);
      cp = mempcpy (cp, h_aliases_len, h_aliases_cnt * sizeof (uint32_t));

      /* The normal addresses first.  */
      addresses = cp;
      for (cnt = 0; cnt < h_addr_list_cnt; ++cnt)
	cp = mempcpy (cp, hst->h_addr_list[cnt], hst->h_length);

      /* Then the aliases.  */
      aliases = cp;
      for (cnt = 0; cnt < h_aliases_cnt; ++cnt)
	cp = mempcpy (cp, hst->h_aliases[cnt], h_aliases_len[cnt]);

      assert (cp
	      == dataset->strdata + total - offsetof (struct dataset,
						      strdata));

      /* If we are adding a GETHOSTBYNAME{,v6} entry we must be prepared
	 that the answer we get from the NSS does not contain the key
	 itself.  This is the case if the resolver is used and the name
	 is extended by the domainnames from /etc/resolv.conf.  Therefore
	 we explicitly add the name here.  */
      key_copy = memcpy (cp, key, req->key_len);

      assert ((char *) &dataset->resp + dataset->head.recsize == cp);

      /* Now we can determine whether on refill we have to create a new
	 record or not.  */
      if (he != NULL)
	{
	  assert (fd == -1);

	  if (total + req->key_len == dh->allocsize
	      && total - offsetof (struct dataset, resp) == dh->recsize
	      && memcmp (&dataset->resp, dh->data,
			 dh->allocsize - offsetof (struct dataset, resp)) == 0)
	    {
	      /* The data has not changed.  We will just bump the
		 timeout value.  Note that the new record has been
		 allocated on the stack and need not be freed.  */
	      assert (h_addr_list_cnt == 1);
	      dh->ttl = dataset->head.ttl;
	      dh->timeout = dataset->head.timeout;
	      ++dh->nreloads;
	    }
	  else
	    {
	      if (h_addr_list_cnt == 1)
		{
		  /* We have to create a new record.  Just allocate
		     appropriate memory and copy it.  */
		  struct dataset *newp
		    = (struct dataset *) mempool_alloc (db,
							total + req->key_len,
							1);
		  if (newp != NULL)
		    {
		      /* Adjust pointers into the memory block.  */
		      addresses = (char *) newp + (addresses
						   - (char *) dataset);
		      aliases = (char *) newp + (aliases - (char *) dataset);
		      assert (key_copy != NULL);
		      key_copy = (char *) newp + (key_copy - (char *) dataset);

		      dataset = memcpy (newp, dataset, total + req->key_len);
		      alloca_used = false;
		    }
		}

	      /* Mark the old record as obsolete.  */
	      dh->usable = false;
	    }
	}
      else
	{
	  /* We write the dataset before inserting it to the database
	     since while inserting this thread might block and so would
	     unnecessarily keep the receiver waiting.  */
	  assert (fd != -1);

	  if (writeall (fd, &dataset->resp, dataset->head.recsize)
	      != dataset->head.recsize)
	    all_written = false;
	}

      /* Add the record to the database.  But only if it has not been
	 stored on the stack.

	 If the record contains more than one IP address (used for
	 load balancing etc) don't cache the entry.  This is something
	 the current cache handling cannot handle and it is more than
	 questionable whether it is worthwhile complicating the cache
	 handling just for handling such a special case. */
      if (! alloca_used)
	{
	  /* If necessary, we also propagate the data to disk.  */
	  if (db->persistent)
	    {
	      // XXX async OK?
	      uintptr_t pval = (uintptr_t) dataset & ~pagesize_m1;
	      msync ((void *) pval,
		     ((uintptr_t) dataset & pagesize_m1)
		     + total + req->key_len, MS_ASYNC);
	    }

	  /* NB: the following code is really complicated.  It has
	     seemlingly duplicated code paths which do the same.  The
	     problem is that we always must add the hash table entry
	     with the FIRST flag set first.  Otherwise we get dangling
	     pointers in case memory allocation fails.  */
	  assert (hst->h_addr_list[1] == NULL);

	  /* Avoid adding names if more than one address is available.  See
	     above for more info.  */
	  assert (req->type == GETHOSTBYNAME
		  || req->type == GETHOSTBYNAMEv6
		  || req->type == GETHOSTBYADDR
		  || req->type == GETHOSTBYADDRv6);

	  (void) cache_add (req->type, key_copy, req->key_len,
			    &dataset->head, true, db, owner, he == NULL);

	  pthread_rwlock_unlock (&db->lock);
	}
    }

  if (__builtin_expect (!all_written, 0) && debug_level > 0)
    {
      char buf[256];
      dbg_log (_("short write in %s: %s"),  __FUNCTION__,
	       strerror_r (errno, buf, sizeof (buf)));
    }

  return timeout;
}


static int
lookup (int type, void *key, struct hostent *resultbufp, char *buffer,
	size_t buflen, struct hostent **hst, int32_t *ttlp)
{
  if (type == GETHOSTBYNAME)
    return __gethostbyname3_r (key, AF_INET, resultbufp, buffer, buflen, hst,
			       &h_errno, ttlp, NULL);
  if (type == GETHOSTBYNAMEv6)
    return __gethostbyname3_r (key, AF_INET6, resultbufp, buffer, buflen, hst,
			       &h_errno, ttlp, NULL);
  if (type == GETHOSTBYADDR)
    return __gethostbyaddr2_r (key, NS_INADDRSZ, AF_INET, resultbufp, buffer,
			       buflen, hst, &h_errno, ttlp);
  return __gethostbyaddr2_r (key, NS_IN6ADDRSZ, AF_INET6, resultbufp, buffer,
			     buflen, hst, &h_errno, ttlp);
}


static time_t
addhstbyX (struct database_dyn *db, int fd, request_header *req,
	   void *key, uid_t uid, struct hashentry *he, struct datahead *dh)
{
  /* Search for the entry matching the key.  Please note that we don't
     look again in the table whether the dataset is now available.  We
     simply insert it.  It does not matter if it is in there twice.  The
     pruning function only will look at the timestamp.  */
  struct hostent resultbuf;
  struct hostent *hst;
  int errval = 0;
  int32_t ttl = INT32_MAX;

  if (__glibc_unlikely (debug_level > 0))
    {
      const char *str;
      char buf[INET6_ADDRSTRLEN + 1];
      if (req->type == GETHOSTBYNAME || req->type == GETHOSTBYNAMEv6)
	str = key;
      else
	str = inet_ntop (req->type == GETHOSTBYADDR ? AF_INET : AF_INET6,
			 key, buf, sizeof (buf));

      if (he == NULL)
	dbg_log (_("Haven't found \"%s\" in hosts cache!"), (char *) str);
      else
	dbg_log (_("Reloading \"%s\" in hosts cache!"), (char *) str);
    }

  struct scratch_buffer tmpbuf;
  scratch_buffer_init (&tmpbuf);

  while (lookup (req->type, key, &resultbuf,
		 tmpbuf.data, tmpbuf.length, &hst, &ttl) != 0
	 && h_errno == NETDB_INTERNAL
	 && (errval = errno) == ERANGE)
    if (!scratch_buffer_grow (&tmpbuf))
      {
	/* We ran out of memory.  We cannot do anything but sending a
	   negative response.  In reality this should never
	   happen.  */
	hst = NULL;
	/* We set the error to indicate this is (possibly) a temporary
	   error and that it does not mean the entry is not
	   available at all.  */
	h_errno = TRY_AGAIN;
	errval = EAGAIN;
	break;
      }

  time_t timeout = cache_addhst (db, fd, req, key, hst, uid, he, dh,
				 h_errno == TRY_AGAIN ? errval : 0, ttl);
  scratch_buffer_free (&tmpbuf);
  return timeout;
}


void
addhstbyname (struct database_dyn *db, int fd, request_header *req,
	      void *key, uid_t uid)
{
  addhstbyX (db, fd, req, key, uid, NULL, NULL);
}


time_t
readdhstbyname (struct database_dyn *db, struct hashentry *he,
		struct datahead *dh)
{
  request_header req =
    {
      .type = GETHOSTBYNAME,
      .key_len = he->len
    };

  return addhstbyX (db, -1, &req, db->data + he->key, he->owner, he, dh);
}


void
addhstbyaddr (struct database_dyn *db, int fd, request_header *req,
	      void *key, uid_t uid)
{
  addhstbyX (db, fd, req, key, uid, NULL, NULL);
}


time_t
readdhstbyaddr (struct database_dyn *db, struct hashentry *he,
		struct datahead *dh)
{
  request_header req =
    {
      .type = GETHOSTBYADDR,
      .key_len = he->len
    };

  return addhstbyX (db, -1, &req, db->data + he->key, he->owner, he, dh);
}


void
addhstbynamev6 (struct database_dyn *db, int fd, request_header *req,
		void *key, uid_t uid)
{
  addhstbyX (db, fd, req, key, uid, NULL, NULL);
}


time_t
readdhstbynamev6 (struct database_dyn *db, struct hashentry *he,
		  struct datahead *dh)
{
  request_header req =
    {
      .type = GETHOSTBYNAMEv6,
      .key_len = he->len
    };

  return addhstbyX (db, -1, &req, db->data + he->key, he->owner, he, dh);
}


void
addhstbyaddrv6 (struct database_dyn *db, int fd, request_header *req,
		void *key, uid_t uid)
{
  addhstbyX (db, fd, req, key, uid, NULL, NULL);
}


time_t
readdhstbyaddrv6 (struct database_dyn *db, struct hashentry *he,
		  struct datahead *dh)
{
  request_header req =
    {
      .type = GETHOSTBYADDRv6,
      .key_len = he->len
    };

  return addhstbyX (db, -1, &req, db->data + he->key, he->owner, he, dh);
}
