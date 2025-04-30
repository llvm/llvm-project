/* Copyright (c) 1998-2021 Free Software Foundation, Inc.
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
#include <atomic.h>
#include <errno.h>
#include <error.h>
#include <inttypes.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <libintl.h>
#include <arpa/inet.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <nss.h>

#include "nscd.h"
#include "dbg_log.h"


/* Wrapper functions with error checking for standard functions.  */
extern void *xcalloc (size_t n, size_t s);


/* Number of times a value is reloaded without being used.  UINT_MAX
   means unlimited.  */
unsigned int reload_count = DEFAULT_RELOAD_LIMIT;


static time_t (*const readdfcts[LASTREQ]) (struct database_dyn *,
					   struct hashentry *,
					   struct datahead *) =
{
  [GETPWBYNAME] = readdpwbyname,
  [GETPWBYUID] = readdpwbyuid,
  [GETGRBYNAME] = readdgrbyname,
  [GETGRBYGID] = readdgrbygid,
  [GETHOSTBYNAME] = readdhstbyname,
  [GETHOSTBYNAMEv6] = readdhstbynamev6,
  [GETHOSTBYADDR] = readdhstbyaddr,
  [GETHOSTBYADDRv6] = readdhstbyaddrv6,
  [GETAI] = readdhstai,
  [INITGROUPS] = readdinitgroups,
  [GETSERVBYNAME] = readdservbyname,
  [GETSERVBYPORT] = readdservbyport,
  [GETNETGRENT] = readdgetnetgrent,
  [INNETGR] = readdinnetgr
};


/* Search the cache for a matching entry and return it when found.  If
   this fails search the negative cache and return (void *) -1 if this
   search was successful.  Otherwise return NULL.

   This function must be called with the read-lock held.  */
struct datahead *
cache_search (request_type type, const void *key, size_t len,
	      struct database_dyn *table, uid_t owner)
{
  unsigned long int hash = __nss_hash (key, len) % table->head->module;

  unsigned long int nsearched = 0;
  struct datahead *result = NULL;

  ref_t work = table->head->array[hash];
  while (work != ENDREF)
    {
      ++nsearched;

      struct hashentry *here = (struct hashentry *) (table->data + work);

      if (type == here->type && len == here->len
	  && memcmp (key, table->data + here->key, len) == 0
	  && here->owner == owner)
	{
	  /* We found the entry.  Increment the appropriate counter.  */
	  struct datahead *dh
	    = (struct datahead *) (table->data + here->packet);

	  /* See whether we must ignore the entry.  */
	  if (dh->usable)
	    {
	      /* We do not synchronize the memory here.  The statistics
		 data is not crucial, we synchronize only once in a while
		 in the cleanup threads.  */
	      if (dh->notfound)
		++table->head->neghit;
	      else
		{
		  ++table->head->poshit;

		  if (dh->nreloads != 0)
		    dh->nreloads = 0;
		}

	      result = dh;
	      break;
	    }
	}

      work = here->next;
    }

  if (nsearched > table->head->maxnsearched)
    table->head->maxnsearched = nsearched;

  return result;
}

/* Add a new entry to the cache.  The return value is zero if the function
   call was successful.

   This function must be called with the read-lock held.

   We modify the table but we nevertheless only acquire a read-lock.
   This is ok since we use operations which would be safe even without
   locking, given that the `prune_cache' function never runs.  Using
   the readlock reduces the chance of conflicts.  */
int
cache_add (int type, const void *key, size_t len, struct datahead *packet,
	   bool first, struct database_dyn *table,
	   uid_t owner, bool prune_wakeup)
{
  if (__glibc_unlikely (debug_level >= 2))
    {
      const char *str;
      char buf[INET6_ADDRSTRLEN + 1];
      if (type == GETHOSTBYADDR || type == GETHOSTBYADDRv6)
	str = inet_ntop (type == GETHOSTBYADDR ? AF_INET : AF_INET6,
			 key, buf, sizeof (buf));
      else
	str = key;

      dbg_log (_("add new entry \"%s\" of type %s for %s to cache%s"),
	       str, serv2str[type], dbnames[table - dbs],
	       first ? _(" (first)") : "");
    }

  unsigned long int hash = __nss_hash (key, len) % table->head->module;
  struct hashentry *newp;

  newp = mempool_alloc (table, sizeof (struct hashentry), 0);
  /* If we cannot allocate memory, just do not do anything.  */
  if (newp == NULL)
    {
      /* If necessary mark the entry as unusable so that lookups will
	 not use it.  */
      if (first)
	packet->usable = false;

      return -1;
    }

  newp->type = type;
  newp->first = first;
  newp->len = len;
  newp->key = (char *) key - table->data;
  assert (newp->key + newp->len <= table->head->first_free);
  newp->owner = owner;
  newp->packet = (char *) packet - table->data;
  assert ((newp->packet & BLOCK_ALIGN_M1) == 0);

  /* Put the new entry in the first position.  */
  /* TODO Review concurrency.  Use atomic_exchange_release.  */
  newp->next = atomic_load_relaxed (&table->head->array[hash]);
  while (!atomic_compare_exchange_weak_release (&table->head->array[hash],
						(ref_t *) &newp->next,
						(ref_t) ((char *) newp
							 - table->data)));

  /* Update the statistics.  */
  if (packet->notfound)
    ++table->head->negmiss;
  else if (first)
    ++table->head->posmiss;

  /* We depend on this value being correct and at least as high as the
     real number of entries.  */
  atomic_increment (&table->head->nentries);

  /* It does not matter that we are not loading the just increment
     value, this is just for statistics.  */
  unsigned long int nentries = table->head->nentries;
  if (nentries > table->head->maxnentries)
    table->head->maxnentries = nentries;

  if (table->persistent)
    // XXX async OK?
    msync ((void *) table->head,
	   (char *) &table->head->array[hash] - (char *) table->head
	   + sizeof (ref_t), MS_ASYNC);

  /* We do not have to worry about the pruning thread if we are
     re-adding the data since this is done by the pruning thread.  We
     also do not have to do anything in case this is not the first
     time the data is entered since different data heads all have the
     same timeout.  */
  if (first && prune_wakeup)
    {
      /* Perhaps the prune thread for the table is not running in a long
	 time.  Wake it if necessary.  */
      pthread_mutex_lock (&table->prune_lock);
      time_t next_wakeup = table->wakeup_time;
      bool do_wakeup = false;
      if (next_wakeup > packet->timeout + CACHE_PRUNE_INTERVAL)
	{
	  table->wakeup_time = packet->timeout;
	  do_wakeup = true;
	}
      pthread_mutex_unlock (&table->prune_lock);
      if (do_wakeup)
	pthread_cond_signal (&table->prune_cond);
    }

  return 0;
}

/* Walk through the table and remove all entries which lifetime ended.

   We have a problem here.  To actually remove the entries we must get
   the write-lock.  But since we want to keep the time we have the
   lock as short as possible we cannot simply acquire the lock when we
   start looking for timedout entries.

   Therefore we do it in two stages: first we look for entries which
   must be invalidated and remember them.  Then we get the lock and
   actually remove them.  This is complicated by the way we have to
   free the data structures since some hash table entries share the same
   data.  */
time_t
prune_cache (struct database_dyn *table, time_t now, int fd)
{
  size_t cnt = table->head->module;

  /* If this table is not actually used don't do anything.  */
  if (cnt == 0)
    {
      if (fd != -1)
	{
	  /* Reply to the INVALIDATE initiator.  */
	  int32_t resp = 0;
	  writeall (fd, &resp, sizeof (resp));
	}

      /* No need to do this again anytime soon.  */
      return 24 * 60 * 60;
    }

  /* If we check for the modification of the underlying file we invalidate
     the entries also in this case.  */
  if (table->check_file && now != LONG_MAX)
    {
      struct traced_file *runp = table->traced_files;

      while (runp != NULL)
	{
#ifdef HAVE_INOTIFY
	  if (runp->inotify_descr[TRACED_FILE] == -1)
#endif
	    {
	      struct stat64 st;

	      if (stat64 (runp->fname, &st) < 0)
		{
		  /* Print a diagnostic that the traced file was missing.
		     We must not disable tracing since the file might return
		     shortly and we want to reload it at the next pruning.
		     Disabling tracing here would go against the configuration
		     as specified by the user via check-files.  */
		  char buf[128];
		  dbg_log (_("checking for monitored file `%s': %s"),
			   runp->fname, strerror_r (errno, buf, sizeof (buf)));
		}
	      else
		{
		  /* This must be `!=` to catch cases where users turn the
		     clocks back and we still want to detect any time difference
		     in mtime.  */
		  if (st.st_mtime != runp->mtime)
		    {
		      dbg_log (_("monitored file `%s` changed (mtime)"),
			       runp->fname);
		      /* The file changed. Invalidate all entries.  */
		      now = LONG_MAX;
		      runp->mtime = st.st_mtime;
#ifdef HAVE_INOTIFY
		      /* Attempt to install a watch on the file.  */
		      install_watches (runp);
#endif
		    }
		}
	    }

	  runp = runp->next;
	}
    }

  /* We run through the table and find values which are not valid anymore.

     Note that for the initial step, finding the entries to be removed,
     we don't need to get any lock.  It is at all timed assured that the
     linked lists are set up correctly and that no second thread prunes
     the cache.  */
  bool *mark;
  size_t memory_needed = cnt * sizeof (bool);
  bool mark_use_alloca;
  if (__glibc_likely (memory_needed <= MAX_STACK_USE))
    {
      mark = alloca (cnt * sizeof (bool));
      memset (mark, '\0', memory_needed);
      mark_use_alloca = true;
    }
  else
    {
      mark = xcalloc (1, memory_needed);
      mark_use_alloca = false;
    }
  size_t first = cnt + 1;
  size_t last = 0;
  char *const data = table->data;
  bool any = false;

  if (__glibc_unlikely (debug_level > 2))
    dbg_log (_("pruning %s cache; time %ld"),
	     dbnames[table - dbs], (long int) now);

#define NO_TIMEOUT LONG_MAX
  time_t next_timeout = NO_TIMEOUT;
  do
    {
      ref_t run = table->head->array[--cnt];

      while (run != ENDREF)
	{
	  struct hashentry *runp = (struct hashentry *) (data + run);
	  struct datahead *dh = (struct datahead *) (data + runp->packet);

	  /* Some debug support.  */
	  if (__glibc_unlikely (debug_level > 2))
	    {
	      char buf[INET6_ADDRSTRLEN];
	      const char *str;

	      if (runp->type == GETHOSTBYADDR || runp->type == GETHOSTBYADDRv6)
		{
		  inet_ntop (runp->type == GETHOSTBYADDR ? AF_INET : AF_INET6,
			     data + runp->key, buf, sizeof (buf));
		  str = buf;
		}
	      else
		str = data + runp->key;

	      dbg_log (_("considering %s entry \"%s\", timeout %" PRIu64),
		       serv2str[runp->type], str, dh->timeout);
	    }

	  /* Check whether the entry timed out.  */
	  if (dh->timeout < now)
	    {
	      /* This hash bucket could contain entries which need to
		 be looked at.  */
	      mark[cnt] = true;

	      first = MIN (first, cnt);
	      last = MAX (last, cnt);

	      /* We only have to look at the data of the first entries
		 since the count information is kept in the data part
		 which is shared.  */
	      if (runp->first)
		{

		  /* At this point there are two choices: we reload the
		     value or we discard it.  Do not change NRELOADS if
		     we never not reload the record.  */
		  if ((reload_count != UINT_MAX
		       && __builtin_expect (dh->nreloads >= reload_count, 0))
		      /* We always remove negative entries.  */
		      || dh->notfound
		      /* Discard everything if the user explicitly
			 requests it.  */
		      || now == LONG_MAX)
		    {
		      /* Remove the value.  */
		      dh->usable = false;

		      /* We definitely have some garbage entries now.  */
		      any = true;
		    }
		  else
		    {
		      /* Reload the value.  We do this only for the
			 initially used key, not the additionally
			 added derived value.  */
		      assert (runp->type < LASTREQ
			      && readdfcts[runp->type] != NULL);

		      time_t timeout = readdfcts[runp->type] (table, runp, dh);
		      next_timeout = MIN (next_timeout, timeout);

		      /* If the entry has been replaced, we might need
			 cleanup.  */
		      any |= !dh->usable;
		    }
		}
	    }
	  else
	    {
	      assert (dh->usable);
	      next_timeout = MIN (next_timeout, dh->timeout);
	    }

	  run = runp->next;
	}
    }
  while (cnt > 0);

  if (__glibc_unlikely (fd != -1))
    {
      /* Reply to the INVALIDATE initiator that the cache has been
	 invalidated.  */
      int32_t resp = 0;
      writeall (fd, &resp, sizeof (resp));
    }

  if (first <= last)
    {
      struct hashentry *head = NULL;

      /* Now we have to get the write lock since we are about to modify
	 the table.  */
      if (__glibc_unlikely (pthread_rwlock_trywrlock (&table->lock) != 0))
	{
	  ++table->head->wrlockdelayed;
	  pthread_rwlock_wrlock (&table->lock);
	}

      /* Now we start modifying the data.  Make sure all readers of the
	 data are aware of this and temporarily don't use the data.  */
      atomic_fetch_add_relaxed (&table->head->gc_cycle, 1);
      assert ((table->head->gc_cycle & 1) == 1);

      while (first <= last)
	{
	  if (mark[first])
	    {
	      ref_t *old = &table->head->array[first];
	      ref_t run = table->head->array[first];

	      assert (run != ENDREF);
	      do
		{
		  struct hashentry *runp = (struct hashentry *) (data + run);
		  struct datahead *dh
		    = (struct datahead *) (data + runp->packet);

		  if (! dh->usable)
		    {
		      /* We need the list only for debugging but it is
			 more costly to avoid creating the list than
			 doing it.  */
		      runp->dellist = head;
		      head = runp;

		      /* No need for an atomic operation, we have the
			 write lock.  */
		      --table->head->nentries;

		      run = *old = runp->next;
		    }
		  else
		    {
		      old = &runp->next;
		      run = runp->next;
		    }
		}
	      while (run != ENDREF);
	    }

	  ++first;
	}

      /* Now we are done modifying the data.  */
      atomic_fetch_add_relaxed (&table->head->gc_cycle, 1);
      assert ((table->head->gc_cycle & 1) == 0);

      /* It's all done.  */
      pthread_rwlock_unlock (&table->lock);

      /* Make sure the data is saved to disk.  */
      if (table->persistent)
	msync (table->head,
	       data + table->head->first_free - (char *) table->head,
	       MS_ASYNC);

      /* One extra pass if we do debugging.  */
      if (__glibc_unlikely (debug_level > 0))
	{
	  struct hashentry *runp = head;

	  while (runp != NULL)
	    {
	      char buf[INET6_ADDRSTRLEN];
	      const char *str;

	      if (runp->type == GETHOSTBYADDR || runp->type == GETHOSTBYADDRv6)
		{
		  inet_ntop (runp->type == GETHOSTBYADDR ? AF_INET : AF_INET6,
			     data + runp->key, buf, sizeof (buf));
		  str = buf;
		}
	      else
		str = data + runp->key;

	      dbg_log ("remove %s entry \"%s\"", serv2str[runp->type], str);

	      runp = runp->dellist;
	    }
	}
    }

  if (__glibc_unlikely (! mark_use_alloca))
    free (mark);

  /* Run garbage collection if any entry has been removed or replaced.  */
  if (any)
    gc (table);

  /* If there is no entry in the database and we therefore have no new
     timeout value, tell the caller to wake up in 24 hours.  */
  return next_timeout == NO_TIMEOUT ? 24 * 60 * 60 : next_timeout - now;
}
