/* Implement simple hashing table with string based keys.
   Copyright (C) 1994-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Ulrich Drepper <drepper@gnu.ai.mit.edu>, October 1994.

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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/types.h>

#include <obstack.h>

#ifdef HAVE_VALUES_H
# include <values.h>
#endif

#include "simple-hash.h"

#define obstack_chunk_alloc malloc
#define obstack_chunk_free free

#ifndef BITSPERBYTE
# define BITSPERBYTE 8
#endif

#define hashval_t uint32_t
#include "hashval.h"

#include <programs/xmalloc.h>

typedef struct hash_entry
{
  unsigned long used;
  const void *key;
  size_t keylen;
  void *data;
  struct hash_entry *next;
}
hash_entry;

/* Prototypes for local functions.  */
static void insert_entry_2 (hash_table *htab, const void *key, size_t keylen,
			    unsigned long hval, size_t idx, void *data);
static size_t lookup (const hash_table *htab, const void *key, size_t keylen,
		      unsigned long int hval);
static int is_prime (unsigned long int candidate);


int
init_hash (hash_table *htab, unsigned long int init_size)
{
  /* We need the size to be a prime.  */
  init_size = next_prime (init_size);

  /* Initialize the data structure.  */
  htab->size = init_size;
  htab->filled = 0;
  htab->first = NULL;
  htab->table = (void *) xcalloc (init_size + 1, sizeof (hash_entry));
  if (htab->table == NULL)
    return -1;

  obstack_init (&htab->mem_pool);

  return 0;
}


int
delete_hash (hash_table *htab)
{
  free (htab->table);
  obstack_free (&htab->mem_pool, NULL);
  return 0;
}


int
insert_entry (hash_table *htab, const void *key, size_t keylen, void *data)
{
  unsigned long int hval = compute_hashval (key, keylen);
  hash_entry *table = (hash_entry *) htab->table;
  size_t idx = lookup (htab, key, keylen, hval);

  if (table[idx].used)
    /* We don't want to overwrite the old value.  */
    return -1;
  else
    {
      /* An empty bucket has been found.  */
      insert_entry_2 (htab, obstack_copy (&htab->mem_pool, key, keylen),
		      keylen, hval, idx, data);
      return 0;
    }
}

static void
insert_entry_2 (hash_table *htab, const void *key, size_t keylen,
		unsigned long int hval, size_t idx, void *data)
{
  hash_entry *table = (hash_entry *) htab->table;

  table[idx].used = hval;
  table[idx].key = key;
  table[idx].keylen = keylen;
  table[idx].data = data;

      /* List the new value in the list.  */
  if ((hash_entry *) htab->first == NULL)
    {
      table[idx].next = &table[idx];
      htab->first = &table[idx];
    }
  else
    {
      table[idx].next = ((hash_entry *) htab->first)->next;
      ((hash_entry *) htab->first)->next = &table[idx];
      htab->first = &table[idx];
    }

  ++htab->filled;
  if (100 * htab->filled > 75 * htab->size)
    {
      /* Table is filled more than 75%.  Resize the table.
	 Experiments have shown that for best performance, this threshold
	 must lie between 40% and 85%.  */
      unsigned long int old_size = htab->size;

      htab->size = next_prime (htab->size * 2);
      htab->filled = 0;
      htab->first = NULL;
      htab->table = (void *) xcalloc (1 + htab->size, sizeof (hash_entry));

      for (idx = 1; idx <= old_size; ++idx)
	if (table[idx].used)
	  insert_entry_2 (htab, table[idx].key, table[idx].keylen,
			  table[idx].used,
			  lookup (htab, table[idx].key, table[idx].keylen,
				  table[idx].used),
			  table[idx].data);

      free (table);
    }
}


int
find_entry (const hash_table *htab, const void *key, size_t keylen,
	    void **result)
{
  hash_entry *table = (hash_entry *) htab->table;
  size_t idx = lookup (htab, key, keylen, compute_hashval (key, keylen));

  if (table[idx].used == 0)
    return -1;

  *result = table[idx].data;
  return 0;
}


int
set_entry (hash_table *htab, const void *key, size_t keylen, void *newval)
{
  hash_entry *table = (hash_entry *) htab->table;
  size_t idx = lookup (htab, key, keylen, compute_hashval (key, keylen));

  if (table[idx].used == 0)
    return -1;

  table[idx].data = newval;
  return 0;
}


int
iterate_table (const hash_table *htab, void **ptr, const void **key,
	       size_t *keylen, void **data)
{
  if (*ptr == NULL)
    {
      if (htab->first == NULL)
	return -1;
      *ptr = (void *) ((hash_entry *) htab->first)->next;
    }
  else
    {
      if (*ptr == htab->first)
	return -1;
      *ptr = (void *) (((hash_entry *) *ptr)->next);
    }

  *key = ((hash_entry *) *ptr)->key;
  *keylen = ((hash_entry *) *ptr)->keylen;
  *data = ((hash_entry *) *ptr)->data;
  return 0;
}


/* References:
   [Aho,Sethi,Ullman] Compilers: Principles, Techniques and Tools, 1986
   [Knuth]	      The Art of Computer Programming, part3 (6.4) */

static size_t
lookup (const hash_table *htab, const void *key, size_t keylen,
	unsigned long int hval)
{
  unsigned long int hash;
  size_t idx;
  hash_entry *table = (hash_entry *) htab->table;

  /* First hash function: simply take the modul but prevent zero.  */
  hash = 1 + hval % htab->size;

  idx = hash;

  if (table[idx].used)
    {
      if (table[idx].used == hval && table[idx].keylen == keylen
	  && memcmp (table[idx].key, key, keylen) == 0)
	return idx;

      /* Second hash function as suggested in [Knuth].  */
      hash = 1 + hval % (htab->size - 2);

      do
	{
	  if (idx <= hash)
	    idx = htab->size + idx - hash;
	  else
	    idx -= hash;

	  /* If entry is found use it.  */
	  if (table[idx].used == hval && table[idx].keylen == keylen
	      && memcmp (table[idx].key, key, keylen) == 0)
	    return idx;
	}
      while (table[idx].used);
    }
  return idx;
}


unsigned long int
next_prime (unsigned long int seed)
{
  /* Make it definitely odd.  */
  seed |= 1;

  while (!is_prime (seed))
    seed += 2;

  return seed;
}


static int
is_prime (unsigned long int candidate)
{
  /* No even number and none less than 10 will be passed here.  */
  unsigned long int divn = 3;
  unsigned long int sq = divn * divn;

  while (sq < candidate && candidate % divn != 0)
    {
      ++divn;
      sq += 4 * divn;
      ++divn;
    }

  return candidate % divn != 0;
}
