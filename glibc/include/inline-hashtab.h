/* Fully-inline hash table, used mainly for managing TLS descriptors.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Alexandre Oliva  <aoliva@redhat.com>

   This file is derived from a 2003's version of libiberty's
   hashtab.c, contributed by Vladimir Makarov (vmakarov@cygnus.com),
   but with most adaptation points and support for deleting elements
   removed.

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

#ifndef INLINE_HASHTAB_H
# define INLINE_HASHTAB_H 1

struct hashtab
{
  /* Table itself.  */
  void **entries;

  /* Current size (in entries) of the hash table */
  size_t size;

  /* Current number of elements.  */
  size_t n_elements;

  /* Free function for the entries array.  This may vary depending on
     how early the array was allocated.  If it is NULL, then the array
     can't be freed.  */
  void (*free) (void *ptr);
};

inline static struct hashtab *
htab_create (void)
{
  struct hashtab *ht = malloc (sizeof (struct hashtab));

  if (! ht)
    return NULL;
  ht->size = 3;
  ht->entries = malloc (sizeof (void *) * ht->size);
  ht->free = __rtld_free;
  if (! ht->entries)
    {
      free (ht);
      return NULL;
    }

  ht->n_elements = 0;

  memset (ht->entries, 0, sizeof (void *) * ht->size);

  return ht;
}

/* This is only called from _dl_unmap, so it's safe to call
   free().  */
inline static void
htab_delete (struct hashtab *htab)
{
  int i;

  for (i = htab->size - 1; i >= 0; i--)
    free (htab->entries[i]);

  htab->free (htab->entries);
  free (htab);
}

/* Similar to htab_find_slot, but without several unwanted side effects:
    - Does not call htab->eq_f when it finds an existing entry.
    - Does not change the count of elements/searches/collisions in the
      hash table.
   This function also assumes there are no deleted entries in the table.
   HASH is the hash value for the element to be inserted.  */

inline static void **
find_empty_slot_for_expand (struct hashtab *htab, int hash)
{
  size_t size = htab->size;
  unsigned int index = hash % size;
  void **slot = htab->entries + index;
  int hash2;

  if (! *slot)
    return slot;

  hash2 = 1 + hash % (size - 2);
  for (;;)
    {
      index += hash2;
      if (index >= size)
	index -= size;

      slot = htab->entries + index;
      if (! *slot)
	return slot;
    }
}

/* The following function changes size of memory allocated for the
   entries and repeatedly inserts the table elements.  The occupancy
   of the table after the call will be about 50%.  Naturally the hash
   table must already exist.  Remember also that the place of the
   table entries is changed.  If memory allocation failures are allowed,
   this function will return zero, indicating that the table could not be
   expanded.  If all goes well, it will return a non-zero value.  */

inline static int
htab_expand (struct hashtab *htab, int (*hash_fn) (void *))
{
  void **oentries;
  void **olimit;
  void **p;
  void **nentries;
  size_t nsize;

  oentries = htab->entries;
  olimit = oentries + htab->size;

  /* Resize only when table after removal of unused elements is either
     too full or too empty.  */
  if (htab->n_elements * 2 > htab->size)
    nsize = _dl_higher_prime_number (htab->n_elements * 2);
  else
    nsize = htab->size;

  nentries = calloc (sizeof (void *), nsize);
  if (nentries == NULL)
    return 0;
  htab->entries = nentries;
  htab->size = nsize;

  p = oentries;
  do
    {
      if (*p)
	*find_empty_slot_for_expand (htab, hash_fn (*p))
	  = *p;

      p++;
    }
  while (p < olimit);

  /* Without recording the free corresponding to the malloc used to
     allocate the table, we couldn't tell whether this was allocated
     by the malloc() built into ld.so or the one in the main
     executable or libc.  Calling free() for something that was
     allocated by the early malloc(), rather than the final run-time
     malloc() could do Very Bad Things (TM).  We will waste memory
     allocated early as long as there's no corresponding free(), but
     this isn't so much memory as to be significant.  */

  htab->free (oentries);

  /* Use the free() corresponding to the malloc() above to free this
     up.  */
  htab->free = __rtld_free;

  return 1;
}

/* This function searches for a hash table slot containing an entry
   equal to the given element.  To delete an entry, call this with
   INSERT = 0, then call htab_clear_slot on the slot returned (possibly
   after doing some checks).  To insert an entry, call this with
   INSERT = 1, then write the value you want into the returned slot.
   When inserting an entry, NULL may be returned if memory allocation
   fails.  */

inline static void **
htab_find_slot (struct hashtab *htab, void *ptr, int insert,
		int (*hash_fn)(void *), int (*eq_fn)(void *, void *))
{
  unsigned int index;
  int hash, hash2;
  size_t size;
  void **entry;

  if (htab->size * 3 <= htab->n_elements * 4
      && htab_expand (htab, hash_fn) == 0)
    return NULL;

  hash = hash_fn (ptr);

  size = htab->size;
  index = hash % size;

  entry = &htab->entries[index];
  if (!*entry)
    goto empty_entry;
  else if (eq_fn (*entry, ptr))
    return entry;

  hash2 = 1 + hash % (size - 2);
  for (;;)
    {
      index += hash2;
      if (index >= size)
	index -= size;

      entry = &htab->entries[index];
      if (!*entry)
	goto empty_entry;
      else if (eq_fn (*entry, ptr))
	return entry;
    }

 empty_entry:
  if (!insert)
    return NULL;

  htab->n_elements++;
  return entry;
}

#endif /* INLINE_HASHTAB_H */
