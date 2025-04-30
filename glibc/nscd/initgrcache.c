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
#include <grp.h>
#include <libintl.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <scratch_buffer.h>
#include <config.h>

#include "dbg_log.h"
#include "nscd.h"

#include "../nss/nsswitch.h"

/* Type of the lookup function.  */
typedef enum nss_status (*initgroups_dyn_function) (const char *, gid_t,
						    long int *, long int *,
						    gid_t **, long int, int *);


static const initgr_response_header notfound =
{
  .version = NSCD_VERSION,
  .found = 0,
  .ngrps = 0
};


#include "../grp/compat-initgroups.c"


static time_t
addinitgroupsX (struct database_dyn *db, int fd, request_header *req,
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
    initgr_response_header resp;
    char strdata[0];
  } *dataset = NULL;

  if (__glibc_unlikely (debug_level > 0))
    {
      if (he == NULL)
	dbg_log (_("Haven't found \"%s\" in group cache!"), (char *) key);
      else
	dbg_log (_("Reloading \"%s\" in group cache!"), (char *) key);
    }

  static nss_action_list group_database;
  nss_action_list nip;
  int no_more;

  if (group_database == NULL)
    no_more = !__nss_database_get (nss_database_group, &group_database);
  else
    no_more = 0;
  nip = group_database;

 /* We always use sysconf even if NGROUPS_MAX is defined.  That way, the
     limit can be raised in the kernel configuration without having to
     recompile libc.  */
  long int limit = __sysconf (_SC_NGROUPS_MAX);

  long int size;
  if (limit > 0)
    /* We limit the size of the intially allocated array.  */
    size = MIN (limit, 64);
  else
    /* No fixed limit on groups.  Pick a starting buffer size.  */
    size = 16;

  long int start = 0;
  bool all_tryagain = true;
  bool any_success = false;

  /* This is temporary memory, we need not (and must not) call
     mempool_alloc.  */
  // XXX This really should use alloca.  need to change the backends.
  gid_t *groups = (gid_t *) malloc (size * sizeof (gid_t));
  if (__glibc_unlikely (groups == NULL))
    /* No more memory.  */
    goto out;

  /* Nothing added yet.  */
  while (! no_more)
    {
      long int prev_start = start;
      enum nss_status status;
      initgroups_dyn_function fct;
      fct = __nss_lookup_function (nip, "initgroups_dyn");

      if (fct == NULL)
	{
	  status = compat_call (nip, key, -1, &start, &size, &groups,
				limit, &errno);

	  if (nss_next_action (nip, NSS_STATUS_UNAVAIL) != NSS_ACTION_CONTINUE)
	    break;
	}
      else
	status = DL_CALL_FCT (fct, (key, -1, &start, &size, &groups,
				    limit, &errno));

      /* Remove duplicates.  */
      long int cnt = prev_start;
      while (cnt < start)
	{
	  long int inner;
	  for (inner = 0; inner < prev_start; ++inner)
	    if (groups[inner] == groups[cnt])
	      break;

	  if (inner < prev_start)
	    groups[cnt] = groups[--start];
	  else
	    ++cnt;
	}

      if (status != NSS_STATUS_TRYAGAIN)
	all_tryagain = false;

      /* This is really only for debugging.  */
      if (NSS_STATUS_TRYAGAIN > status || status > NSS_STATUS_RETURN)
	__libc_fatal ("Illegal status in internal_getgrouplist.\n");

      any_success |= status == NSS_STATUS_SUCCESS;

      if (status != NSS_STATUS_SUCCESS
	  && nss_next_action (nip, status) == NSS_ACTION_RETURN)
	 break;

      if (nip[1].module == NULL)
	no_more = -1;
      else
	++nip;
    }

  bool all_written;
  ssize_t total;
  time_t timeout;
 out:
  all_written = true;
  timeout = MAX_TIMEOUT_VALUE;
  if (!any_success)
    {
      /* Nothing found.  Create a negative result record.  */
      total = sizeof (notfound);

      if (he != NULL && all_tryagain)
	{
	  /* If we have an old record available but cannot find one now
	     because the service is not available we keep the old record
	     and make sure it does not get removed.  */
	  if (reload_count != UINT_MAX && dh->nreloads == reload_count)
	    /* Do not reset the value if we never not reload the record.  */
	    dh->nreloads = reload_count - 1;

	  /* Reload with the same time-to-live value.  */
	  timeout = dh->timeout = time (NULL) + db->postimeout;
	}
      else
	{
	  /* We have no data.  This means we send the standard reply for this
	     case.  */
	  if (fd != -1
	      && TEMP_FAILURE_RETRY (send (fd, &notfound, total,
					   MSG_NOSIGNAL)) != total)
	    all_written = false;

	  /* If we have a transient error or cannot permanently store
	     the result, so be it.  */
	  if (all_tryagain || __builtin_expect (db->negtimeout == 0, 0))
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
					   db->negtimeout);

	      /* This is the reply.  */
	      memcpy (&dataset->resp, &notfound, total);

	      /* Copy the key data.  */
	      char *key_copy = memcpy (dataset->strdata, key, req->key_len);

	      /* If necessary, we also propagate the data to disk.  */
	      if (db->persistent)
		{
		  // XXX async OK?
		  uintptr_t pval = (uintptr_t) dataset & ~pagesize_m1;
		  msync ((void *) pval,
			 ((uintptr_t) dataset & pagesize_m1)
			 + sizeof (struct dataset) + req->key_len, MS_ASYNC);
		}

	      (void) cache_add (req->type, key_copy, req->key_len,
				&dataset->head, true, db, uid, he == NULL);

	      pthread_rwlock_unlock (&db->lock);

	      /* Mark the old entry as obsolete.  */
	      if (dh != NULL)
		dh->usable = false;
	    }
	}
    }
  else
    {

      total = offsetof (struct dataset, strdata) + start * sizeof (int32_t);

      /* If we refill the cache, first assume the reconrd did not
	 change.  Allocate memory on the cache since it is likely
	 discarded anyway.  If it turns out to be necessary to have a
	 new record we can still allocate real memory.  */
      bool alloca_used = false;
      dataset = NULL;

      if (he == NULL)
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
				   db->postimeout);

      dataset->resp.version = NSCD_VERSION;
      dataset->resp.found = 1;
      dataset->resp.ngrps = start;

      char *cp = dataset->strdata;

      /* Copy the GID values.  If the size of the types match this is
	 very simple.  */
      if (sizeof (gid_t) == sizeof (int32_t))
	cp = mempcpy (cp, groups, start * sizeof (gid_t));
      else
	{
	  gid_t *gcp = (gid_t *) cp;

	  for (int i = 0; i < start; ++i)
	    *gcp++ = groups[i];

	  cp = (char *) gcp;
	}

      /* Finally the user name.  */
      memcpy (cp, key, req->key_len);

      assert (cp == dataset->strdata + total - offsetof (struct dataset,
							 strdata));

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
	      dh->timeout = dataset->head.timeout;
	      ++dh->nreloads;
	    }
	  else
	    {
	      /* We have to create a new record.  Just allocate
		 appropriate memory and copy it.  */
	      struct dataset *newp
		= (struct dataset *) mempool_alloc (db, total + req->key_len,
						    1);
	      if (newp != NULL)
		{
		  /* Adjust pointer into the memory block.  */
		  cp = (char *) newp + (cp - (char *) dataset);

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
	     since while inserting this thread might block and so would
	     unnecessarily let the receiver wait.  */
	  assert (fd != -1);

	  if (writeall (fd, &dataset->resp, dataset->head.recsize)
	      != dataset->head.recsize)
	    all_written = false;
	}


      /* Add the record to the database.  But only if it has not been
	 stored on the stack.  */
      if (! alloca_used)
	{
	  /* If necessary, we also propagate the data to disk.  */
	  if (db->persistent)
	    {
	      // XXX async OK?
	      uintptr_t pval = (uintptr_t) dataset & ~pagesize_m1;
	      msync ((void *) pval,
		     ((uintptr_t) dataset & pagesize_m1) + total
		     + req->key_len, MS_ASYNC);
	    }

	  (void) cache_add (INITGROUPS, cp, req->key_len, &dataset->head, true,
			    db, uid, he == NULL);

	  pthread_rwlock_unlock (&db->lock);
	}
    }

  free (groups);

  if (__builtin_expect (!all_written, 0) && debug_level > 0)
    {
      char buf[256];
      dbg_log (_("short write in %s: %s"), __FUNCTION__,
	       strerror_r (errno, buf, sizeof (buf)));
    }

  return timeout;
}


void
addinitgroups (struct database_dyn *db, int fd, request_header *req, void *key,
	       uid_t uid)
{
  addinitgroupsX (db, fd, req, key, uid, NULL, NULL);
}


time_t
readdinitgroups (struct database_dyn *db, struct hashentry *he,
		 struct datahead *dh)
{
  request_header req =
    {
      .type = INITGROUPS,
      .key_len = he->len
    };

  return addinitgroupsX (db, -1, &req, db->data + he->key, he->owner, he, dh);
}
