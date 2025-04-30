/* Cache handling for services lookup.
   Copyright (C) 2007-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@drepper.com>, 2007.

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
#include <unistd.h>
#include <stdint.h>
#include <sys/mman.h>
#include <kernel-features.h>
#include <scratch_buffer.h>

#include "nscd.h"
#include "dbg_log.h"


/* This is the standard reply in case the service is disabled.  */
static const serv_response_header disabled =
{
  .version = NSCD_VERSION,
  .found = -1,
  .s_name_len = 0,
  .s_proto_len = 0,
  .s_aliases_cnt = 0,
  .s_port = -1
};

/* This is the struct describing how to write this record.  */
const struct iovec serv_iov_disabled =
{
  .iov_base = (void *) &disabled,
  .iov_len = sizeof (disabled)
};


/* This is the standard reply in case we haven't found the dataset.  */
static const serv_response_header notfound =
{
  .version = NSCD_VERSION,
  .found = 0,
  .s_name_len = 0,
  .s_proto_len = 0,
  .s_aliases_cnt = 0,
  .s_port = -1
};


static time_t
cache_addserv (struct database_dyn *db, int fd, request_header *req,
	       const void *key, struct servent *serv, uid_t owner,
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
    serv_response_header resp;
    char strdata[0];
  } *dataset;

  assert (offsetof (struct dataset, resp) == offsetof (struct datahead, data));

  time_t timeout = MAX_TIMEOUT_VALUE;
  if (serv == NULL)
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
      size_t s_name_len = strlen (serv->s_name) + 1;
      size_t s_proto_len = strlen (serv->s_proto) + 1;
      uint32_t *s_aliases_len;
      size_t s_aliases_cnt;
      char *aliases;
      char *cp;
      size_t cnt;

      /* Determine the number of aliases.  */
      s_aliases_cnt = 0;
      for (cnt = 0; serv->s_aliases[cnt] != NULL; ++cnt)
	++s_aliases_cnt;
      /* Determine the length of all aliases.  */
      s_aliases_len = (uint32_t *) alloca (s_aliases_cnt * sizeof (uint32_t));
      total = 0;
      for (cnt = 0; cnt < s_aliases_cnt; ++cnt)
	{
	  s_aliases_len[cnt] = strlen (serv->s_aliases[cnt]) + 1;
	  total += s_aliases_len[cnt];
	}

      total += (offsetof (struct dataset, strdata)
		+ s_name_len
		+ s_proto_len
		+ s_aliases_cnt * sizeof (uint32_t));

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
      dataset->resp.s_name_len = s_name_len;
      dataset->resp.s_proto_len = s_proto_len;
      dataset->resp.s_port = serv->s_port;
      dataset->resp.s_aliases_cnt = s_aliases_cnt;

      cp = dataset->strdata;

      cp = mempcpy (cp, serv->s_name, s_name_len);
      cp = mempcpy (cp, serv->s_proto, s_proto_len);
      cp = mempcpy (cp, s_aliases_len, s_aliases_cnt * sizeof (uint32_t));

      /* Then the aliases.  */
      aliases = cp;
      for (cnt = 0; cnt < s_aliases_cnt; ++cnt)
	cp = mempcpy (cp, serv->s_aliases[cnt], s_aliases_len[cnt]);

      assert (cp
	      == dataset->strdata + total - offsetof (struct dataset,
						      strdata));

      char *key_copy = memcpy (cp, key, req->key_len);

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
		  /* Adjust pointers into the memory block.  */
		  aliases = (char *) newp + (aliases - (char *) dataset);
		  assert (key_copy != NULL);
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
	     since while inserting this thread might block and so would
	     unnecessarily keep the receiver waiting.  */
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
		     ((uintptr_t) dataset & pagesize_m1)
		     + total + req->key_len, MS_ASYNC);
	    }

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
lookup (int type, char *key, struct servent *resultbufp, char *buffer,
	size_t buflen, struct servent **serv)
{
  char *proto = strrchr (key, '/');
  if (proto != NULL && proto != key)
    {
      key = strndupa (key, proto - key);
      if (proto[1] == '\0')
	proto = NULL;
      else
	++proto;
    }

  if (type == GETSERVBYNAME)
    return __getservbyname_r (key, proto, resultbufp, buffer, buflen, serv);

  assert (type == GETSERVBYPORT);
  return __getservbyport_r (atol (key), proto, resultbufp, buffer, buflen,
			    serv);
}


static time_t
addservbyX (struct database_dyn *db, int fd, request_header *req,
	    char *key, uid_t uid, struct hashentry *he, struct datahead *dh)
{
  /* Search for the entry matching the key.  Please note that we don't
     look again in the table whether the dataset is now available.  We
     simply insert it.  It does not matter if it is in there twice.  The
     pruning function only will look at the timestamp.  */
  struct servent resultbuf;
  struct servent *serv;
  int errval = 0;
  struct scratch_buffer tmpbuf;
  scratch_buffer_init (&tmpbuf);

  if (__glibc_unlikely (debug_level > 0))
    {
      if (he == NULL)
	dbg_log (_("Haven't found \"%s\" in services cache!"), key);
      else
	dbg_log (_("Reloading \"%s\" in services cache!"), key);
    }

  while (lookup (req->type, key, &resultbuf,
		 tmpbuf.data, tmpbuf.length, &serv) != 0
	 && (errval = errno) == ERANGE)
    if (!scratch_buffer_grow (&tmpbuf))
      {
	/* We ran out of memory.  We cannot do anything but sending a
	   negative response.  In reality this should never
	   happen.  */
	serv = NULL;
	/* We set the error to indicate this is (possibly) a temporary
	   error and that it does not mean the entry is not available
	   at all.  */
	errval = EAGAIN;
	break;
      }

  time_t timeout = cache_addserv (db, fd, req, key, serv, uid, he, dh, errval);
  scratch_buffer_free (&tmpbuf);
  return timeout;
}


void
addservbyname (struct database_dyn *db, int fd, request_header *req,
	       void *key, uid_t uid)
{
  addservbyX (db, fd, req, key, uid, NULL, NULL);
}


time_t
readdservbyname (struct database_dyn *db, struct hashentry *he,
		 struct datahead *dh)
{
  request_header req =
    {
      .type = GETSERVBYNAME,
      .key_len = he->len
    };

  return addservbyX (db, -1, &req, db->data + he->key, he->owner, he, dh);
}


void
addservbyport (struct database_dyn *db, int fd, request_header *req,
	       void *key, uid_t uid)
{
  addservbyX (db, fd, req, key, uid, NULL, NULL);
}


time_t
readdservbyport (struct database_dyn *db, struct hashentry *he,
		 struct datahead *dh)
{
  request_header req =
    {
      .type = GETSERVBYPORT,
      .key_len = he->len
    };

  return addservbyX (db, -1, &req, db->data + he->key, he->owner, he, dh);
}
