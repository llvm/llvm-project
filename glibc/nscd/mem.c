/* Cache memory handling.
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
#include <error.h>
#include <fcntl.h>
#include <inttypes.h>
#include <libintl.h>
#include <limits.h>
#include <obstack.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/param.h>

#include "dbg_log.h"
#include "nscd.h"


static int
sort_he (const void *p1, const void *p2)
{
  struct hashentry *h1 = *(struct hashentry **) p1;
  struct hashentry *h2 = *(struct hashentry **) p2;

  if (h1 < h2)
    return -1;
  if (h1 > h2)
    return 1;
  return 0;
}


static int
sort_he_data (const void *p1, const void *p2)
{
  struct hashentry *h1 = *(struct hashentry **) p1;
  struct hashentry *h2 = *(struct hashentry **) p2;

  if (h1->packet < h2->packet)
    return -1;
  if (h1->packet > h2->packet)
    return 1;
  return 0;
}


/* Basic definitions for the bitmap implementation.  Only BITMAP_T
   needs to be changed to choose a different word size.  */
#define BITMAP_T uint8_t
#define BITS (CHAR_BIT * sizeof (BITMAP_T))
#define ALLBITS ((((BITMAP_T) 1) << BITS) - 1)
#define HIGHBIT (((BITMAP_T) 1) << (BITS - 1))


static void
markrange (BITMAP_T *mark, ref_t start, size_t len)
{
  /* Adjust parameters for block alignment.  */
  assert ((start & BLOCK_ALIGN_M1) == 0);
  start /= BLOCK_ALIGN;
  len = (len + BLOCK_ALIGN_M1) / BLOCK_ALIGN;

  size_t elem = start / BITS;

  if (start % BITS != 0)
    {
      if (start % BITS + len <= BITS)
	{
	  /* All fits in the partial byte.  */
	  mark[elem] |= (ALLBITS >> (BITS - len)) << (start % BITS);
	  return;
	}

      mark[elem++] |= ALLBITS << (start % BITS);
      len -= BITS - (start % BITS);
    }

  while (len >= BITS)
    {
      mark[elem++] = ALLBITS;
      len -= BITS;
    }

  if (len > 0)
    mark[elem] |= ALLBITS >> (BITS - len);
}


void
gc (struct database_dyn *db)
{
  /* We need write access.  */
  pthread_rwlock_wrlock (&db->lock);

  /* And the memory handling lock.  */
  pthread_mutex_lock (&db->memlock);

  /* We need an array representing the data area.  All memory
     allocation is BLOCK_ALIGN aligned so this is the level at which
     we have to look at the memory.  We use a mark and sweep algorithm
     where the marks are placed in this array.  */
  assert (db->head->first_free % BLOCK_ALIGN == 0);

  BITMAP_T *mark;
  bool mark_use_malloc;
  /* In prune_cache we are also using a dynamically allocated array.
     If the array in the caller is too large we have malloc'ed it.  */
  size_t stack_used = sizeof (bool) * db->head->module;
  if (__glibc_unlikely (stack_used > MAX_STACK_USE))
    stack_used = 0;
  size_t nmark = (db->head->first_free / BLOCK_ALIGN + BITS - 1) / BITS;
  size_t memory_needed = nmark * sizeof (BITMAP_T);
  if (__glibc_likely (stack_used + memory_needed <= MAX_STACK_USE))
    {
      mark = (BITMAP_T *) alloca_account (memory_needed, stack_used);
      mark_use_malloc = false;
      memset (mark, '\0', memory_needed);
    }
  else
    {
      mark = (BITMAP_T *) xcalloc (1, memory_needed);
      mark_use_malloc = true;
    }

  /* Create an array which can hold pointer to all the entries in hash
     entries.  */
  memory_needed = 2 * db->head->nentries * sizeof (struct hashentry *);
  struct hashentry **he;
  struct hashentry **he_data;
  bool he_use_malloc;
  if (__glibc_likely (stack_used + memory_needed <= MAX_STACK_USE))
    {
      he = alloca_account (memory_needed, stack_used);
      he_use_malloc = false;
    }
  else
    {
      he = xmalloc (memory_needed);
      he_use_malloc = true;
    }
  he_data = &he[db->head->nentries];

  size_t cnt = 0;
  for (size_t idx = 0; idx < db->head->module; ++idx)
    {
      ref_t *prevp = &db->head->array[idx];
      ref_t run = *prevp;

      while (run != ENDREF)
	{
	  assert (cnt < db->head->nentries);
	  he[cnt] = (struct hashentry *) (db->data + run);

	  he[cnt]->prevp = prevp;
	  prevp = &he[cnt]->next;

	  /* This is the hash entry itself.  */
	  markrange (mark, run, sizeof (struct hashentry));

	  /* Add the information for the data itself.  We do this
	     only for the one special entry marked with FIRST.  */
	  if (he[cnt]->first)
	    {
	      struct datahead *dh
		= (struct datahead *) (db->data + he[cnt]->packet);
	      markrange (mark, he[cnt]->packet, dh->allocsize);
	    }

	  run = he[cnt]->next;

	  ++cnt;
	}
    }
  assert (cnt == db->head->nentries);

  /* Sort the entries by the addresses of the referenced data.  All
     the entries pointing to the same DATAHEAD object will have the
     same key.  Stability of the sorting is unimportant.  */
  memcpy (he_data, he, cnt * sizeof (struct hashentry *));
  qsort (he_data, cnt, sizeof (struct hashentry *), sort_he_data);

  /* Sort the entries by their address.  */
  qsort (he, cnt, sizeof (struct hashentry *), sort_he);

#define obstack_chunk_alloc xmalloc
#define obstack_chunk_free free
  struct obstack ob;
  obstack_init (&ob);

  /* Determine the highest used address.  */
  size_t high = nmark;
  while (high > 0 && mark[high - 1] == 0)
    --high;

  /* No memory used.  */
  if (high == 0)
    {
      db->head->first_free = 0;
      goto out;
    }

  /* Determine the highest offset.  */
  BITMAP_T mask = HIGHBIT;
  ref_t highref = (high * BITS - 1) * BLOCK_ALIGN;
  while ((mark[high - 1] & mask) == 0)
    {
      mask >>= 1;
      highref -= BLOCK_ALIGN;
    }

  /* Now we can iterate over the MARK array and find bits which are not
     set.  These represent memory which can be recovered.  */
  size_t byte = 0;
  /* Find the first gap.  */
  while (byte < high && mark[byte] == ALLBITS)
    ++byte;

  if (byte == high
      || (byte == high - 1 && (mark[byte] & ~(mask | (mask - 1))) == 0))
    /* No gap.  */
    goto out;

  mask = 1;
  cnt = 0;
  while ((mark[byte] & mask) != 0)
    {
      ++cnt;
      mask <<= 1;
    }
  ref_t off_free = (byte * BITS + cnt) * BLOCK_ALIGN;
  assert (off_free <= db->head->first_free);

  struct hashentry **next_hash = he;
  struct hashentry **next_data = he_data;

  /* Skip over the hash entries in the first block which does not get
     moved.  */
  while (next_hash < &he[db->head->nentries]
	 && *next_hash < (struct hashentry *) (db->data + off_free))
    ++next_hash;

  while (next_data < &he_data[db->head->nentries]
	 && (*next_data)->packet < off_free)
    ++next_data;


  /* Now we start modifying the data.  Make sure all readers of the
     data are aware of this and temporarily don't use the data.  */
  atomic_fetch_add_relaxed (&db->head->gc_cycle, 1);
  assert ((db->head->gc_cycle & 1) == 1);


  /* We do not perform the move operations right away since the
     he_data array is not sorted by the address of the data.  */
  struct moveinfo
  {
    void *from;
    void *to;
    size_t size;
    struct moveinfo *next;
  } *moves = NULL;

  while (byte < high)
    {
      /* Search for the next filled block.  BYTE is the index of the
	 entry in MARK, MASK is the bit, and CNT is the bit number.
	 OFF_FILLED is the corresponding offset.  */
      if ((mark[byte] & ~(mask - 1)) == 0)
	{
	  /* No other bit set in the same element of MARK.  Search in the
	     following memory.  */
	  do
	    ++byte;
	  while (byte < high && mark[byte] == 0);

	  if (byte == high)
	    /* That was it.  */
	    break;

	  mask = 1;
	  cnt = 0;
	}
      /* Find the exact bit.  */
      while ((mark[byte] & mask) == 0)
	{
	  ++cnt;
	  mask <<= 1;
	}

      ref_t off_alloc = (byte * BITS + cnt) * BLOCK_ALIGN;
      assert (off_alloc <= db->head->first_free);

      /* Find the end of the used area.  */
      if ((mark[byte] & ~(mask - 1)) == (BITMAP_T) ~(mask - 1))
	{
	  /* All other bits set.  Search the next bytes in MARK.  */
	  do
	    ++byte;
	  while (byte < high && mark[byte] == ALLBITS);

	  mask = 1;
	  cnt = 0;
	}
      if (byte < high)
	{
	  /* Find the exact bit.  */
	  while ((mark[byte] & mask) != 0)
	    {
	      ++cnt;
	      mask <<= 1;
	    }
	}

      ref_t off_allocend = (byte * BITS + cnt) * BLOCK_ALIGN;
      assert (off_allocend <= db->head->first_free);
      /* Now we know that we can copy the area from OFF_ALLOC to
	 OFF_ALLOCEND (not included) to the memory starting at
	 OFF_FREE.  First fix up all the entries for the
	 displacement.  */
      ref_t disp = off_alloc - off_free;

      struct moveinfo *new_move;
      if (__builtin_expect (stack_used + sizeof (*new_move) <= MAX_STACK_USE,
			    1))
	new_move = alloca_account (sizeof (*new_move), stack_used);
      else
	new_move = obstack_alloc (&ob, sizeof (*new_move));
      new_move->from = db->data + off_alloc;
      new_move->to = db->data + off_free;
      new_move->size = off_allocend - off_alloc;
      /* Create a circular list to be always able to append at the end.  */
      if (moves == NULL)
	moves = new_move->next = new_move;
      else
	{
	  new_move->next = moves->next;
	  moves = moves->next = new_move;
	}

      /* The following loop will prepare to move this much data.  */
      off_free += off_allocend - off_alloc;

      while (off_alloc < off_allocend)
	{
	  /* Determine whether the next entry is for a hash entry or
	     the data.  */
	  if ((struct hashentry *) (db->data + off_alloc) == *next_hash)
	    {
	      /* Just correct the forward reference.  */
	      *(*next_hash++)->prevp -= disp;

	      off_alloc += ((sizeof (struct hashentry) + BLOCK_ALIGN_M1)
			    & ~BLOCK_ALIGN_M1);
	    }
	  else
	    {
	      assert (next_data < &he_data[db->head->nentries]);
	      assert ((*next_data)->packet == off_alloc);

	      struct datahead *dh = (struct datahead *) (db->data + off_alloc);
	      do
		{
		  assert ((*next_data)->key >= (*next_data)->packet);
		  assert ((*next_data)->key + (*next_data)->len
			  <= (*next_data)->packet + dh->allocsize);

		  (*next_data)->packet -= disp;
		  (*next_data)->key -= disp;
		  ++next_data;
		}
	      while (next_data < &he_data[db->head->nentries]
		     && (*next_data)->packet == off_alloc);

	      off_alloc += (dh->allocsize + BLOCK_ALIGN_M1) & ~BLOCK_ALIGN_M1;
	    }
	}
      assert (off_alloc == off_allocend);

      assert (off_alloc <= db->head->first_free);
      if (off_alloc == db->head->first_free)
	/* We are done, that was the last block.  */
	break;
    }
  assert (next_hash == &he[db->head->nentries]);
  assert (next_data == &he_data[db->head->nentries]);

  /* Now perform the actual moves.  */
  if (moves != NULL)
    {
      struct moveinfo *runp = moves->next;
      do
	{
	  assert ((char *) runp->to >= db->data);
	  assert ((char *) runp->to + runp->size
		  <= db->data  + db->head->first_free);
	  assert ((char *) runp->from >= db->data);
	  assert ((char *) runp->from + runp->size
		  <= db->data  + db->head->first_free);

	  /* The regions may overlap.  */
	  memmove (runp->to, runp->from, runp->size);
	  runp = runp->next;
	}
      while (runp != moves->next);

      if (__glibc_unlikely (debug_level >= 3))
	dbg_log (_("freed %zu bytes in %s cache"),
		 (size_t) (db->head->first_free
			   - ((char *) moves->to + moves->size - db->data)),
		 dbnames[db - dbs]);

      /* The byte past the end of the last copied block is the next
	 available byte.  */
      db->head->first_free = (char *) moves->to + moves->size - db->data;

      /* Consistency check.  */
      if (__glibc_unlikely (debug_level >= 3))
	{
	  for (size_t idx = 0; idx < db->head->module; ++idx)
	    {
	      ref_t run = db->head->array[idx];
	      size_t cnt = 0;

	      while (run != ENDREF)
		{
		  if (run + sizeof (struct hashentry) > db->head->first_free)
		    {
		      dbg_log ("entry %zu in hash bucket %zu out of bounds: "
			       "%" PRIu32 "+%zu > %zu\n",
			       cnt, idx, run, sizeof (struct hashentry),
			       (size_t) db->head->first_free);
		      break;
		    }

		  struct hashentry *he = (struct hashentry *) (db->data + run);

		  if (he->key + he->len > db->head->first_free)
		    dbg_log ("key of entry %zu in hash bucket %zu out of "
			     "bounds: %" PRIu32 "+%zu > %zu\n",
			     cnt, idx, he->key, (size_t) he->len,
			     (size_t) db->head->first_free);

		  if (he->packet + sizeof (struct datahead)
		      > db->head->first_free)
		    dbg_log ("packet of entry %zu in hash bucket %zu out of "
			     "bounds: %" PRIu32 "+%zu > %zu\n",
			     cnt, idx, he->packet, sizeof (struct datahead),
			     (size_t) db->head->first_free);
		  else
		    {
		      struct datahead *dh = (struct datahead *) (db->data
								 + he->packet);
		      if (he->packet + dh->allocsize
			  > db->head->first_free)
			dbg_log ("full key of entry %zu in hash bucket %zu "
				 "out of bounds: %" PRIu32 "+%zu > %zu",
				 cnt, idx, he->packet, (size_t) dh->allocsize,
				 (size_t) db->head->first_free);
		    }

		  run = he->next;
		  ++cnt;
		}
	    }
	}
    }

  /* Make sure the data on disk is updated.  */
  if (db->persistent)
    msync (db->head, db->data + db->head->first_free - (char *) db->head,
	   MS_ASYNC);


  /* Now we are done modifying the data.  */
  atomic_fetch_add_relaxed (&db->head->gc_cycle, 1);
  assert ((db->head->gc_cycle & 1) == 0);

  /* We are done.  */
 out:
  pthread_mutex_unlock (&db->memlock);
  pthread_rwlock_unlock (&db->lock);

  if (he_use_malloc)
    free (he);
  if (mark_use_malloc)
    free (mark);

  obstack_free (&ob, NULL);
}


void *
mempool_alloc (struct database_dyn *db, size_t len, int data_alloc)
{
  /* Make sure LEN is a multiple of our maximum alignment so we can
     keep track of used memory is multiples of this alignment value.  */
  if ((len & BLOCK_ALIGN_M1) != 0)
    len += BLOCK_ALIGN - (len & BLOCK_ALIGN_M1);

  if (data_alloc)
    pthread_rwlock_rdlock (&db->lock);

  pthread_mutex_lock (&db->memlock);

  assert ((db->head->first_free & BLOCK_ALIGN_M1) == 0);

  bool tried_resize = false;
  void *res;
 retry:
  res = db->data + db->head->first_free;

  if (__glibc_unlikely (db->head->first_free + len > db->head->data_size))
    {
      if (! tried_resize)
	{
	  /* Try to resize the database.  Grow size of 1/8th.  */
	  size_t oldtotal = (sizeof (struct database_pers_head)
			     + roundup (db->head->module * sizeof (ref_t),
					ALIGN)
			     + db->head->data_size);
	  size_t new_data_size = (db->head->data_size
				  + MAX (2 * len, db->head->data_size / 8));
	  size_t newtotal = (sizeof (struct database_pers_head)
			     + roundup (db->head->module * sizeof (ref_t), ALIGN)
			     + new_data_size);
	  if (newtotal > db->max_db_size)
	    {
	      new_data_size -= newtotal - db->max_db_size;
	      newtotal = db->max_db_size;
	    }

	  if (db->mmap_used && newtotal > oldtotal
	      /* We only have to adjust the file size.  The new pages
		 become magically available.  */
	      && TEMP_FAILURE_RETRY_VAL (posix_fallocate (db->wr_fd, oldtotal,
							  newtotal
							  - oldtotal)) == 0)
	    {
	      db->head->data_size = new_data_size;
	      tried_resize = true;
	      goto retry;
	    }
	}

      if (data_alloc)
	pthread_rwlock_unlock (&db->lock);

      if (! db->last_alloc_failed)
	{
	  dbg_log (_("no more memory for database '%s'"), dbnames[db - dbs]);

	  db->last_alloc_failed = true;
	}

      ++db->head->addfailed;

      /* No luck.  */
      res = NULL;
    }
  else
    {
      db->head->first_free += len;

      db->last_alloc_failed = false;

    }

  pthread_mutex_unlock (&db->memlock);

  return res;
}
