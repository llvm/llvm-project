/* Cache handling for netgroup lookup.
   Copyright (C) 2011-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gmail.com>, 2011.

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
#include <libintl.h>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>

#include "../inet/netgroup.h"
#include "nscd.h"
#include "dbg_log.h"

#include <kernel-features.h>


/* This is the standard reply in case the service is disabled.  */
static const netgroup_response_header disabled =
{
  .version = NSCD_VERSION,
  .found = -1,
  .nresults = 0,
  .result_len = 0
};

/* This is the struct describing how to write this record.  */
const struct iovec netgroup_iov_disabled =
{
  .iov_base = (void *) &disabled,
  .iov_len = sizeof (disabled)
};


/* This is the standard reply in case we haven't found the dataset.  */
static const netgroup_response_header notfound =
{
  .version = NSCD_VERSION,
  .found = 0,
  .nresults = 0,
  .result_len = 0
};


struct dataset
{
  struct datahead head;
  netgroup_response_header resp;
  char strdata[0];
};

/* Sends a notfound message and prepares a notfound dataset to write to the
   cache.  Returns true if there was enough memory to allocate the dataset and
   returns the dataset in DATASETP, total bytes to write in TOTALP and the
   timeout in TIMEOUTP.  KEY_COPY is set to point to the copy of the key in the
   dataset. */
static bool
do_notfound (struct database_dyn *db, int fd, request_header *req,
	       const char *key, struct dataset **datasetp, ssize_t *totalp,
	       time_t *timeoutp, char **key_copy)
{
  struct dataset *dataset;
  ssize_t total;
  time_t timeout;
  bool cacheable = false;

  total = sizeof (notfound);
  timeout = time (NULL) + db->negtimeout;

  if (fd != -1)
    TEMP_FAILURE_RETRY (send (fd, &notfound, total, MSG_NOSIGNAL));

  dataset = mempool_alloc (db, sizeof (struct dataset) + req->key_len, 1);
  /* If we cannot permanently store the result, so be it.  */
  if (dataset != NULL)
    {
      timeout = datahead_init_neg (&dataset->head,
				   sizeof (struct dataset) + req->key_len,
				   total, db->negtimeout);

      /* This is the reply.  */
      memcpy (&dataset->resp, &notfound, total);

      /* Copy the key data.  */
      memcpy (dataset->strdata, key, req->key_len);
      *key_copy = dataset->strdata;

      cacheable = true;
    }
  *timeoutp = timeout;
  *totalp = total;
  *datasetp = dataset;
  return cacheable;
}

static time_t
addgetnetgrentX (struct database_dyn *db, int fd, request_header *req,
		 const char *key, uid_t uid, struct hashentry *he,
		 struct datahead *dh, struct dataset **resultp,
		 void **tofreep)
{
  if (__glibc_unlikely (debug_level > 0))
    {
      if (he == NULL)
	dbg_log (_("Haven't found \"%s\" in netgroup cache!"), key);
      else
	dbg_log (_("Reloading \"%s\" in netgroup cache!"), key);
    }

  static nss_action_list netgroup_database;
  time_t timeout;
  struct dataset *dataset;
  bool cacheable = false;
  ssize_t total;
  bool found = false;

  char *key_copy = NULL;
  struct __netgrent data;
  size_t buflen = MAX (1024, sizeof (*dataset) + req->key_len);
  size_t buffilled = sizeof (*dataset);
  char *buffer = NULL;
  size_t nentries = 0;
  size_t group_len = strlen (key) + 1;
  struct name_list *first_needed
    = alloca (sizeof (struct name_list) + group_len);
  *tofreep = NULL;

  if (netgroup_database == NULL
      && !__nss_database_get (nss_database_netgroup, &netgroup_database))
    {
      /* No such service.  */
      cacheable = do_notfound (db, fd, req, key, &dataset, &total, &timeout,
			       &key_copy);
      goto writeout;
    }

  memset (&data, '\0', sizeof (data));
  buffer = xmalloc (buflen);
  *tofreep = buffer;
  first_needed->next = first_needed;
  memcpy (first_needed->name, key, group_len);
  data.needed_groups = first_needed;

  while (data.needed_groups != NULL)
    {
      /* Add the next group to the list of those which are known.  */
      struct name_list *this_group = data.needed_groups->next;
      if (this_group == data.needed_groups)
	data.needed_groups = NULL;
      else
	data.needed_groups->next = this_group->next;
      this_group->next = data.known_groups;
      data.known_groups = this_group;

      union
      {
	enum nss_status (*f) (const char *, struct __netgrent *);
	void *ptr;
      } setfct;

      nss_action_list nip = netgroup_database;
      int no_more = __nss_lookup (&nip, "setnetgrent", NULL, &setfct.ptr);
      while (!no_more)
	{
	  enum nss_status status
	    = DL_CALL_FCT (*setfct.f, (data.known_groups->name, &data));

	  if (status == NSS_STATUS_SUCCESS)
	    {
	      found = true;
	      union
	      {
		enum nss_status (*f) (struct __netgrent *, char *, size_t,
				      int *);
		void *ptr;
	      } getfct;
	      getfct.ptr = __nss_lookup_function (nip, "getnetgrent_r");
	      if (getfct.f != NULL)
		while (1)
		  {
		    int e;
		    status = getfct.f (&data, buffer + buffilled,
				       buflen - buffilled - req->key_len, &e);
		    if (status == NSS_STATUS_SUCCESS)
		      {
			if (data.type == triple_val)
			  {
			    const char *nhost = data.val.triple.host;
			    const char *nuser = data.val.triple.user;
			    const char *ndomain = data.val.triple.domain;

			    size_t hostlen = strlen (nhost ?: "") + 1;
			    size_t userlen = strlen (nuser ?: "") + 1;
			    size_t domainlen = strlen (ndomain ?: "") + 1;

			    if (nhost == NULL || nuser == NULL || ndomain == NULL
				|| nhost > nuser || nuser > ndomain)
			      {
				const char *last = nhost;
				if (last == NULL
				    || (nuser != NULL && nuser > last))
				  last = nuser;
				if (last == NULL
				    || (ndomain != NULL && ndomain > last))
				  last = ndomain;

				size_t bufused
				  = (last == NULL
				     ? buffilled
				     : last + strlen (last) + 1 - buffer);

				/* We have to make temporary copies.  */
				size_t needed = hostlen + userlen + domainlen;

				if (buflen - req->key_len - bufused < needed)
				  {
				    buflen += MAX (buflen, 2 * needed);
				    /* Save offset in the old buffer.  We don't
				       bother with the NULL check here since
				       we'll do that later anyway.  */
				    size_t nhostdiff = nhost - buffer;
				    size_t nuserdiff = nuser - buffer;
				    size_t ndomaindiff = ndomain - buffer;

				    char *newbuf = xrealloc (buffer, buflen);
				    /* Fix up the triplet pointers into the new
				       buffer.  */
				    nhost = (nhost ? newbuf + nhostdiff
					     : NULL);
				    nuser = (nuser ? newbuf + nuserdiff
					     : NULL);
				    ndomain = (ndomain ? newbuf + ndomaindiff
					       : NULL);
				    *tofreep = buffer = newbuf;
				  }

				nhost = memcpy (buffer + bufused,
						nhost ?: "", hostlen);
				nuser = memcpy ((char *) nhost + hostlen,
						nuser ?: "", userlen);
				ndomain = memcpy ((char *) nuser + userlen,
						  ndomain ?: "", domainlen);
			      }

			    char *wp = buffer + buffilled;
			    wp = memmove (wp, nhost ?: "", hostlen);
			    wp += hostlen;
			    wp = memmove (wp, nuser ?: "", userlen);
			    wp += userlen;
			    wp = memmove (wp, ndomain ?: "", domainlen);
			    wp += domainlen;
			    buffilled = wp - buffer;
			    ++nentries;
			  }
			else
			  {
			    /* Check that the group has not been
			       requested before.  */
			    struct name_list *runp = data.needed_groups;
			    if (runp != NULL)
			      while (1)
				{
				  if (strcmp (runp->name, data.val.group) == 0)
				    break;

				  runp = runp->next;
				  if (runp == data.needed_groups)
				    {
				      runp = NULL;
				      break;
				    }
				}

			    if (runp == NULL)
			      {
				runp = data.known_groups;
				while (runp != NULL)
				  if (strcmp (runp->name, data.val.group) == 0)
				    break;
				  else
				    runp = runp->next;
				}

			    if (runp == NULL)
			      {
				/* A new group is requested.  */
				size_t namelen = strlen (data.val.group) + 1;
				struct name_list *newg = alloca (sizeof (*newg)
								 + namelen);
				memcpy (newg->name, data.val.group, namelen);
				if (data.needed_groups == NULL)
				  data.needed_groups = newg->next = newg;
				else
				  {
				    newg->next = data.needed_groups->next;
				    data.needed_groups->next = newg;
				    data.needed_groups = newg;
				  }
			      }
			  }
		      }
		    else if (status == NSS_STATUS_TRYAGAIN && e == ERANGE)
		      {
			buflen *= 2;
			*tofreep = buffer = xrealloc (buffer, buflen);
		      }
		    else if (status == NSS_STATUS_RETURN
			     || status == NSS_STATUS_NOTFOUND
			     || status == NSS_STATUS_UNAVAIL)
		      /* This was either the last one for this group or the
			 group was empty or the NSS module had an internal
			 failure.  Look at next group if available.  */
		      break;
		  }

	      enum nss_status (*endfct) (struct __netgrent *);
	      endfct = __nss_lookup_function (nip, "endnetgrent");
	      if (endfct != NULL)
		(void) DL_CALL_FCT (*endfct, (&data));

	      break;
	    }

	  no_more = __nss_next2 (&nip, "setnetgrent", NULL, &setfct.ptr,
				 status, 0);
	}
    }

  /* No results.  Return a failure and write out a notfound record in the
     cache.  */
  if (!found)
    {
      cacheable = do_notfound (db, fd, req, key, &dataset, &total, &timeout,
			       &key_copy);
      goto writeout;
    }

  total = buffilled;

  /* Fill in the dataset.  */
  dataset = (struct dataset *) buffer;
  timeout = datahead_init_pos (&dataset->head, total + req->key_len,
			       total - offsetof (struct dataset, resp),
			       he == NULL ? 0 : dh->nreloads + 1,
			       db->postimeout);

  dataset->resp.version = NSCD_VERSION;
  dataset->resp.found = 1;
  dataset->resp.nresults = nentries;
  dataset->resp.result_len = buffilled - sizeof (*dataset);

  assert (buflen - buffilled >= req->key_len);
  key_copy = memcpy (buffer + buffilled, key, req->key_len);
  buffilled += req->key_len;

  /* Now we can determine whether on refill we have to create a new
     record or not.  */
  if (he != NULL)
    {
      assert (fd == -1);

      if (dataset->head.allocsize == dh->allocsize
	  && dataset->head.recsize == dh->recsize
	  && memcmp (&dataset->resp, dh->data,
		     dh->allocsize - offsetof (struct dataset, resp)) == 0)
	{
	  /* The data has not changed.  We will just bump the timeout
	     value.  Note that the new record has been allocated on
	     the stack and need not be freed.  */
	  dh->timeout = dataset->head.timeout;
	  dh->ttl = dataset->head.ttl;
	  ++dh->nreloads;
	  dataset = (struct dataset *) dh;

	  goto out;
	}
    }

  {
    struct dataset *newp
      = (struct dataset *) mempool_alloc (db, total + req->key_len, 1);
    if (__glibc_likely (newp != NULL))
      {
	/* Adjust pointer into the memory block.  */
	key_copy = (char *) newp + (key_copy - buffer);

	dataset = memcpy (newp, dataset, total + req->key_len);
	cacheable = true;

	if (he != NULL)
	  /* Mark the old record as obsolete.  */
	  dh->usable = false;
      }
  }

  if (he == NULL && fd != -1)
    {
      /* We write the dataset before inserting it to the database
	 since while inserting this thread might block and so would
	 unnecessarily let the receiver wait.  */
    writeout:
      writeall (fd, &dataset->resp, dataset->head.recsize);
    }

  if (cacheable)
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

 out:
  *resultp = dataset;

  return timeout;
}


static time_t
addinnetgrX (struct database_dyn *db, int fd, request_header *req,
	     char *key, uid_t uid, struct hashentry *he,
	     struct datahead *dh)
{
  const char *group = key;
  key = (char *) rawmemchr (key, '\0') + 1;
  size_t group_len = key - group;
  const char *host = *key++ ? key : NULL;
  if (host != NULL)
    key = (char *) rawmemchr (key, '\0') + 1;
  const char *user = *key++ ? key : NULL;
  if (user != NULL)
    key = (char *) rawmemchr (key, '\0') + 1;
  const char *domain = *key++ ? key : NULL;

  if (__glibc_unlikely (debug_level > 0))
    {
      if (he == NULL)
	dbg_log (_("Haven't found \"%s (%s,%s,%s)\" in netgroup cache!"),
		 group, host ?: "", user ?: "", domain ?: "");
      else
	dbg_log (_("Reloading \"%s (%s,%s,%s)\" in netgroup cache!"),
		 group, host ?: "", user ?: "", domain ?: "");
    }

  struct dataset *result = (struct dataset *) cache_search (GETNETGRENT,
							    group, group_len,
							    db, uid);
  time_t timeout;
  void *tofree;
  if (result != NULL)
    {
      timeout = result->head.timeout;
      tofree = NULL;
    }
  else
    {
      request_header req_get =
	{
	  .type = GETNETGRENT,
	  .key_len = group_len
	};
      timeout = addgetnetgrentX (db, -1, &req_get, group, uid, NULL, NULL,
				 &result, &tofree);
    }

  struct indataset
  {
    struct datahead head;
    innetgroup_response_header resp;
  } *dataset
      = (struct indataset *) mempool_alloc (db,
					    sizeof (*dataset) + req->key_len,
					    1);
  struct indataset dataset_mem;
  bool cacheable = true;
  if (__glibc_unlikely (dataset == NULL))
    {
      cacheable = false;
      dataset = &dataset_mem;
    }

  datahead_init_pos (&dataset->head, sizeof (*dataset) + req->key_len,
		     sizeof (innetgroup_response_header),
		     he == NULL ? 0 : dh->nreloads + 1, result->head.ttl);
  /* Set the notfound status and timeout based on the result from
     getnetgrent.  */
  dataset->head.notfound = result->head.notfound;
  dataset->head.timeout = timeout;

  dataset->resp.version = NSCD_VERSION;
  dataset->resp.found = result->resp.found;
  /* Until we find a matching entry the result is 0.  */
  dataset->resp.result = 0;

  char *key_copy = memcpy ((char *) (dataset + 1), group, req->key_len);

  if (dataset->resp.found)
    {
      const char *triplets = (const char *) (&result->resp + 1);

      for (nscd_ssize_t i = result->resp.nresults; i > 0; --i)
	{
	  bool success = true;

	  /* For the host, user and domain in each triplet, we assume success
	     if the value is blank because that is how the wildcard entry to
	     match anything is stored in the netgroup cache.  */
	  if (host != NULL && *triplets != '\0')
	    success = strcmp (host, triplets) == 0;
	  triplets = (const char *) rawmemchr (triplets, '\0') + 1;

	  if (success && user != NULL && *triplets != '\0')
	    success = strcmp (user, triplets) == 0;
	  triplets = (const char *) rawmemchr (triplets, '\0') + 1;

	  if (success && (domain == NULL || *triplets == '\0'
			  || strcmp (domain, triplets) == 0))
	    {
	      dataset->resp.result = 1;
	      break;
	    }
	  triplets = (const char *) rawmemchr (triplets, '\0') + 1;
	}
    }

  if (he != NULL && dh->data[0].innetgroupdata.result == dataset->resp.result)
    {
      /* The data has not changed.  We will just bump the timeout
	 value.  Note that the new record has been allocated on
	 the stack and need not be freed.  */
      dh->timeout = timeout;
      dh->ttl = dataset->head.ttl;
      ++dh->nreloads;
      if (cacheable)
        pthread_rwlock_unlock (&db->lock);
      goto out;
    }

  if (he == NULL)
    {
      /* We write the dataset before inserting it to the database
	 since while inserting this thread might block and so would
	 unnecessarily let the receiver wait.  */
      assert (fd != -1);

      writeall (fd, &dataset->resp, sizeof (innetgroup_response_header));
    }

  if (cacheable)
    {
      /* If necessary, we also propagate the data to disk.  */
      if (db->persistent)
	{
	  // XXX async OK?
	  uintptr_t pval = (uintptr_t) dataset & ~pagesize_m1;
	  msync ((void *) pval,
		 ((uintptr_t) dataset & pagesize_m1) + sizeof (*dataset)
		 + req->key_len,
		 MS_ASYNC);
	}

      (void) cache_add (req->type, key_copy, req->key_len, &dataset->head,
			true, db, uid, he == NULL);

      pthread_rwlock_unlock (&db->lock);

      /* Mark the old entry as obsolete.  */
      if (dh != NULL)
	dh->usable = false;
    }

 out:
  free (tofree);
  return timeout;
}


static time_t
addgetnetgrentX_ignore (struct database_dyn *db, int fd, request_header *req,
			const char *key, uid_t uid, struct hashentry *he,
			struct datahead *dh)
{
  struct dataset *ignore;
  void *tofree;
  time_t timeout = addgetnetgrentX (db, fd, req, key, uid, he, dh,
				    &ignore, &tofree);
  free (tofree);
  return timeout;
}

void
addgetnetgrent (struct database_dyn *db, int fd, request_header *req,
		void *key, uid_t uid)
{
  addgetnetgrentX_ignore (db, fd, req, key, uid, NULL, NULL);
}


time_t
readdgetnetgrent (struct database_dyn *db, struct hashentry *he,
		  struct datahead *dh)
{
  request_header req =
    {
      .type = GETNETGRENT,
      .key_len = he->len
    };
  return addgetnetgrentX_ignore
    (db, -1, &req, db->data + he->key, he->owner, he, dh);
}


void
addinnetgr (struct database_dyn *db, int fd, request_header *req,
	    void *key, uid_t uid)
{
  addinnetgrX (db, fd, req, key, uid, NULL, NULL);
}


time_t
readdinnetgr (struct database_dyn *db, struct hashentry *he,
	      struct datahead *dh)
{
  request_header req =
    {
      .type = INNETGR,
      .key_len = he->len
    };

  return addinnetgrX (db, -1, &req, db->data + he->key, he->owner, he, dh);
}
