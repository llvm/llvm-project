/* Cache handling for group lookup.
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

#include <assert.h>
#include <errno.h>
#include <error.h>
#include <grp.h>
#include <libintl.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <stackinfo.h>
#include <scratch_buffer.h>

#include "nscd.h"
#include "dbg_log.h"

/* This is the standard reply in case the service is disabled.  */
static const gr_response_header disabled =
{
  .version = NSCD_VERSION,
  .found = -1,
  .gr_name_len = 0,
  .gr_passwd_len = 0,
  .gr_gid = -1,
  .gr_mem_cnt = 0,
};

/* This is the struct describing how to write this record.  */
const struct iovec grp_iov_disabled =
{
  .iov_base = (void *) &disabled,
  .iov_len = sizeof (disabled)
};


/* This is the standard reply in case we haven't found the dataset.  */
static const gr_response_header notfound =
{
  .version = NSCD_VERSION,
  .found = 0,
  .gr_name_len = 0,
  .gr_passwd_len = 0,
  .gr_gid = -1,
  .gr_mem_cnt = 0,
};


static time_t
cache_addgr (struct database_dyn *db, int fd, request_header *req,
	     const void *key, struct group *grp, uid_t owner,
	     struct hashentry *const he, struct datahead *dh, int errval)
{
  bool all_written = true;
  ssize_t total;
  time_t t = time (NULL);

  /* We allocate all data in one memory block: the iov vector,
     the response header and the dataset itself.  */
  struct dataset
  {
    struct datahead head;
    gr_response_header resp;
    char strdata[0];
  } *dataset;

  assert (offsetof (struct dataset, resp) == offsetof (struct datahead, data));

  time_t timeout = MAX_TIMEOUT_VALUE;
  if (grp == NULL)
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
	  timeout = dh->timeout = t + db->postimeout;

	  total = 0;
	}
      else
	{
	  /* We have no data.  This means we send the standard reply for this
	     case.  */
	  total = sizeof (notfound);

	  if (fd != -1
	      && TEMP_FAILURE_RETRY (send (fd, &notfound, total,
					   MSG_NOSIGNAL)) != total)
	    all_written = false;

	  /* If we have a transient error or cannot permanently store
	     the result, so be it.  */
	  if (errno == EAGAIN || __builtin_expect (db->negtimeout == 0, 0))
	    {
	      /* Mark the old entry as obsolete.  */
	      if (dh != NULL)
		dh->usable = false;
	    }
	  else if ((dataset = mempool_alloc (db, sizeof (struct dataset) + req->key_len, 1)) != NULL)
	    {
	      timeout = datahead_init_neg (&dataset->head,
					   (sizeof (struct dataset)
					    + req->key_len), total,
					   db->negtimeout);

	      /* This is the reply.  */
	      memcpy (&dataset->resp, &notfound, total);

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
      size_t gr_name_len = strlen (grp->gr_name) + 1;
      size_t gr_passwd_len = strlen (grp->gr_passwd) + 1;
      size_t gr_mem_cnt = 0;
      uint32_t *gr_mem_len;
      size_t gr_mem_len_total = 0;
      char *gr_name;
      char *cp;
      const size_t key_len = strlen (key);
      const size_t buf_len = 3 * sizeof (grp->gr_gid) + key_len + 1;
      size_t alloca_used = 0;
      char *buf = alloca_account (buf_len, alloca_used);
      ssize_t n;
      size_t cnt;

      /* We need this to insert the `bygid' entry.  */
      int key_offset;
      n = snprintf (buf, buf_len, "%d%c%n%s", grp->gr_gid, '\0',
		    &key_offset, (char *) key) + 1;

      /* Determine the length of all members.  */
      while (grp->gr_mem[gr_mem_cnt])
	++gr_mem_cnt;
      gr_mem_len = alloca_account (gr_mem_cnt * sizeof (uint32_t), alloca_used);
      for (gr_mem_cnt = 0; grp->gr_mem[gr_mem_cnt]; ++gr_mem_cnt)
	{
	  gr_mem_len[gr_mem_cnt] = strlen (grp->gr_mem[gr_mem_cnt]) + 1;
	  gr_mem_len_total += gr_mem_len[gr_mem_cnt];
	}

      total = (offsetof (struct dataset, strdata)
	       + gr_mem_cnt * sizeof (uint32_t)
	       + gr_name_len + gr_passwd_len + gr_mem_len_total);

      /* If we refill the cache, first assume the reconrd did not
	 change.  Allocate memory on the cache since it is likely
	 discarded anyway.  If it turns out to be necessary to have a
	 new record we can still allocate real memory.  */
      bool dataset_temporary = false;
      bool dataset_malloced = false;
      dataset = NULL;

      if (he == NULL)
	{
	  /* Prevent an INVALIDATE request from pruning the data between
	     the two calls to cache_add.  */
	  if (db->propagate)
	    pthread_mutex_lock (&db->prune_run_lock);
	  dataset = (struct dataset *) mempool_alloc (db, total + n, 1);
	}

      if (dataset == NULL)
	{
	  if (he == NULL && db->propagate)
	    pthread_mutex_unlock (&db->prune_run_lock);

	  /* We cannot permanently add the result in the moment.  But
	     we can provide the result as is.  Store the data in some
	     temporary memory.  */
	  if (! __libc_use_alloca (alloca_used + total + n))
	    {
	      dataset = malloc (total + n);
	      /* Perhaps we should log a message that we were unable
		 to allocate memory for a large request.  */
	      if (dataset == NULL)
		goto out;
	      dataset_malloced = true;
	    }
	  else
	    dataset = alloca_account (total + n, alloca_used);

	  /* We cannot add this record to the permanent database.  */
	  dataset_temporary = true;
	}

      timeout = datahead_init_pos (&dataset->head, total + n,
				   total - offsetof (struct dataset, resp),
				   he == NULL ? 0 : dh->nreloads + 1,
				   db->postimeout);

      dataset->resp.version = NSCD_VERSION;
      dataset->resp.found = 1;
      dataset->resp.gr_name_len = gr_name_len;
      dataset->resp.gr_passwd_len = gr_passwd_len;
      dataset->resp.gr_gid = grp->gr_gid;
      dataset->resp.gr_mem_cnt = gr_mem_cnt;

      cp = dataset->strdata;

      /* This is the member string length array.  */
      cp = mempcpy (cp, gr_mem_len, gr_mem_cnt * sizeof (uint32_t));
      gr_name = cp;
      cp = mempcpy (cp, grp->gr_name, gr_name_len);
      cp = mempcpy (cp, grp->gr_passwd, gr_passwd_len);

      for (cnt = 0; cnt < gr_mem_cnt; ++cnt)
	cp = mempcpy (cp, grp->gr_mem[cnt], gr_mem_len[cnt]);

      /* Finally the stringified GID value.  */
      memcpy (cp, buf, n);
      char *key_copy = cp + key_offset;
      assert (key_copy == (char *) rawmemchr (cp, '\0') + 1);

      assert (cp == dataset->strdata + total - offsetof (struct dataset,
							 strdata));

      /* Now we can determine whether on refill we have to create a new
	 record or not.  */
      if (he != NULL)
	{
	  assert (fd == -1);

	  if (total + n == dh->allocsize
	      && total - offsetof (struct dataset, resp) == dh->recsize
	      && memcmp (&dataset->resp, dh->data,
			 dh->allocsize - offsetof (struct dataset, resp)) == 0)
	    {
	      /* The data has not changed.  We will just bump the
		 timeout value.  Note that the new record has been
		 allocated on the stack and need not be freed.  */
	      dh->timeout = dataset->head.timeout;
	      ++dh->nreloads;

	      /* If the new record was allocated via malloc, then we must free
		 it here.  */
	      if (dataset_malloced)
		free (dataset);
	    }
	  else
	    {
	      /* We have to create a new record.  Just allocate
		 appropriate memory and copy it.  */
	      struct dataset *newp
		= (struct dataset *) mempool_alloc (db, total + n, 1);
	      if (newp != NULL)
		{
		  /* Adjust pointers into the memory block.  */
		  gr_name = (char *) newp + (gr_name - (char *) dataset);
		  cp = (char *) newp + (cp - (char *) dataset);
		  key_copy = (char *) newp + (key_copy - (char *) dataset);

		  dataset = memcpy (newp, dataset, total + n);
		  dataset_temporary = false;
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
      if (! dataset_temporary)
	{
	  /* If necessary, we also propagate the data to disk.  */
	  if (db->persistent)
	    {
	      // XXX async OK?
	      uintptr_t pval = (uintptr_t) dataset & ~pagesize_m1;
	      msync ((void *) pval,
		     ((uintptr_t) dataset & pagesize_m1) + total + n,
		     MS_ASYNC);
	    }

	  /* NB: in the following code we always must add the entry
	     marked with FIRST first.  Otherwise we end up with
	     dangling "pointers" in case a latter hash entry cannot be
	     added.  */
	  bool first = true;

	  /* If the request was by GID, add that entry first.  */
	  if (req->type == GETGRBYGID)
	    {
	      if (cache_add (GETGRBYGID, cp, key_offset, &dataset->head, true,
			     db, owner, he == NULL) < 0)
		goto out;

	      first = false;
	    }
	  /* If the key is different from the name add a separate entry.  */
	  else if (strcmp (key_copy, gr_name) != 0)
	    {
	      if (cache_add (GETGRBYNAME, key_copy, key_len + 1,
			     &dataset->head, true, db, owner, he == NULL) < 0)
		goto out;

	      first = false;
	    }

	  /* We have to add the value for both, byname and byuid.  */
	  if ((req->type == GETGRBYNAME || db->propagate)
	      && __builtin_expect (cache_add (GETGRBYNAME, gr_name,
					      gr_name_len,
					      &dataset->head, first, db, owner,
					      he == NULL)
				   == 0, 1))
	    {
	      if (req->type == GETGRBYNAME && db->propagate)
		(void) cache_add (GETGRBYGID, cp, key_offset, &dataset->head,
				  false, db, owner, false);
	    }

	out:
	  pthread_rwlock_unlock (&db->lock);
	  if (he == NULL && db->propagate)
	    pthread_mutex_unlock (&db->prune_run_lock);
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


union keytype
{
  void *v;
  gid_t g;
};


static int
lookup (int type, union keytype key, struct group *resultbufp, char *buffer,
	size_t buflen, struct group **grp)
{
  if (type == GETGRBYNAME)
    return __getgrnam_r (key.v, resultbufp, buffer, buflen, grp);
  else
    return __getgrgid_r (key.g, resultbufp, buffer, buflen, grp);
}


static time_t
addgrbyX (struct database_dyn *db, int fd, request_header *req,
	  union keytype key, const char *keystr, uid_t uid,
	  struct hashentry *he, struct datahead *dh)
{
  /* Search for the entry matching the key.  Please note that we don't
     look again in the table whether the dataset is now available.  We
     simply insert it.  It does not matter if it is in there twice.  The
     pruning function only will look at the timestamp.  */

  struct group resultbuf;
  struct group *grp;
  int errval = 0;
  struct scratch_buffer tmpbuf;
  scratch_buffer_init (&tmpbuf);

  if (__glibc_unlikely (debug_level > 0))
    {
      if (he == NULL)
	dbg_log (_("Haven't found \"%s\" in group cache!"), keystr);
      else
	dbg_log (_("Reloading \"%s\" in group cache!"), keystr);
    }

  while (lookup (req->type, key, &resultbuf,
		 tmpbuf.data, tmpbuf.length, &grp) != 0
	 && (errval = errno) == ERANGE)
    if (!scratch_buffer_grow (&tmpbuf))
      {
	/* We ran out of memory.  We cannot do anything but sending a
	   negative response.  In reality this should never
	   happen.  */
	grp = NULL;
	/* We set the error to indicate this is (possibly) a temporary
	   error and that it does not mean the entry is not available
	   at all.  */
	errval = EAGAIN;
	break;
      }

  time_t timeout = cache_addgr (db, fd, req, keystr, grp, uid, he, dh, errval);
  scratch_buffer_free (&tmpbuf);
  return timeout;
}


void
addgrbyname (struct database_dyn *db, int fd, request_header *req,
	     void *key, uid_t uid)
{
  union keytype u = { .v = key };

  addgrbyX (db, fd, req, u, key, uid, NULL, NULL);
}


time_t
readdgrbyname (struct database_dyn *db, struct hashentry *he,
	       struct datahead *dh)
{
  request_header req =
    {
      .type = GETGRBYNAME,
      .key_len = he->len
    };
  union keytype u = { .v = db->data + he->key };

  return addgrbyX (db, -1, &req, u, db->data + he->key, he->owner, he, dh);
}


void
addgrbygid (struct database_dyn *db, int fd, request_header *req,
	    void *key, uid_t uid)
{
  char *ep;
  gid_t gid = strtoul ((char *) key, &ep, 10);

  if (*(char *) key == '\0' || *ep != '\0')  /* invalid numeric uid */
    {
      if (debug_level > 0)
	dbg_log (_("Invalid numeric gid \"%s\"!"), (char *) key);

      errno = EINVAL;
      return;
    }

  union keytype u = { .g = gid };

  addgrbyX (db, fd, req, u, key, uid, NULL, NULL);
}


time_t
readdgrbygid (struct database_dyn *db, struct hashentry *he,
	      struct datahead *dh)
{
  char *ep;
  gid_t gid = strtoul (db->data + he->key, &ep, 10);

  /* Since the key has been added before it must be OK.  */
  assert (*(db->data + he->key) != '\0' && *ep == '\0');

  request_header req =
    {
      .type = GETGRBYGID,
      .key_len = he->len
    };
  union keytype u = { .g = gid };

  return addgrbyX (db, -1, &req, u, db->data + he->key, he->owner, he, dh);
}
