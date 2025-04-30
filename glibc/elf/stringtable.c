/* String tables for ld.so.cache construction.  Implementation.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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
#include <error.h>
#include <ldconfig.h>
#include <libintl.h>
#include <stdlib.h>
#include <string.h>
#include <stringtable.h>

static void
stringtable_init (struct stringtable *table)
{
  table->count = 0;

  /* This needs to be a power of two.  128 is sufficient to keep track
     of 42 DSOs without resizing (assuming two strings per DSOs).
     glibc itself comes with more than 20 DSOs, so 64 would likely to
     be too small.  */
  table->allocated = 128;

  table->entries = xcalloc (table->allocated, sizeof (table->entries[0]));
}

/* 32-bit FNV-1a hash function.  */
static uint32_t
fnv1a (const char *string, size_t length)
{
  const unsigned char *p = (const unsigned char *) string;
  uint32_t hash = 2166136261U;
  for (size_t i = 0; i < length; ++i)
    {
      hash ^= p[i];
      hash *= 16777619U;
    }
  return hash;
}

/* Double the capacity of the hash table.  */
static void
stringtable_rehash (struct stringtable *table)
{
  /* This computation cannot overflow because the old total in-memory
     size of the hash table is larger than the computed value.  */
  uint32_t new_allocated = table->allocated * 2;
  struct stringtable_entry **new_entries
    = xcalloc (new_allocated, sizeof (table->entries[0]));

  uint32_t mask = new_allocated - 1;
  for (uint32_t i = 0; i < table->allocated; ++i)
    for (struct stringtable_entry *e = table->entries[i]; e != NULL; )
      {
        struct stringtable_entry *next = e->next;
        uint32_t hash = fnv1a (e->string, e->length);
        uint32_t new_index = hash & mask;
        e->next = new_entries[new_index];
        new_entries[new_index] = e;
        e = next;
      }

  free (table->entries);
  table->entries = new_entries;
  table->allocated = new_allocated;
}

struct stringtable_entry *
stringtable_add (struct stringtable *table, const char *string)
{
  /* Check for a zero-initialized table.  */
  if (table->allocated == 0)
    stringtable_init (table);

  size_t length = strlen (string);
  if (length > (1U << 30))
    error (EXIT_FAILURE, 0, _("String table string is too long"));
  uint32_t hash = fnv1a (string, length);

  /* Return a previously-existing entry.  */
  for (struct stringtable_entry *e
         = table->entries[hash & (table->allocated - 1)];
       e != NULL; e = e->next)
    if (e->length == length && memcmp (e->string, string, length) == 0)
      return e;

  /* Increase the size of the table if necessary.  Keep utilization
     below two thirds.  */
  if (table->count >= (1U << 30))
    error (EXIT_FAILURE, 0, _("String table has too many entries"));
  if (table->count * 3 > table->allocated * 2)
    stringtable_rehash (table);

  /* Add the new table entry.  */
  ++table->count;
  struct stringtable_entry *e
    = xmalloc (offsetof (struct stringtable_entry, string) + length + 1);
  uint32_t index = hash & (table->allocated - 1);
  e->next = table->entries[index];
  table->entries[index] = e;
  e->length = length;
  e->offset = 0;
  memcpy (e->string, string, length + 1);
  return e;
}

/* Sort reversed strings in reverse lexicographic order.  This is used
   for tail merging.  */
static int
finalize_compare (const void *l, const void *r)
{
  struct stringtable_entry *left = *(struct stringtable_entry **) l;
  struct stringtable_entry *right = *(struct stringtable_entry **) r;
  size_t to_compare;
  if (left->length < right->length)
    to_compare = left->length;
  else
    to_compare = right->length;
  for (size_t i = 1; i <= to_compare; ++i)
    {
      unsigned char lch = left->string[left->length - i];
      unsigned char rch = right->string[right->length - i];
      if (lch != rch)
        return rch - lch;
    }
  if (left->length == right->length)
    return 0;
  else if (left->length < right->length)
    /* Longer strings should come first.  */
    return 1;
  else
    return -1;
}

void
stringtable_finalize (struct stringtable *table,
                      struct stringtable_finalized *result)
{
  if (table->count == 0)
    {
      result->strings = xstrdup ("");
      result->size = 0;
      return;
    }

  /* Optimize the order of the strings.  */
  struct stringtable_entry **array = xcalloc (table->count, sizeof (*array));
  {
    size_t j = 0;
    for (uint32_t i = 0; i < table->allocated; ++i)
      for (struct stringtable_entry *e = table->entries[i]; e != NULL;
           e = e->next)
        {
          array[j] = e;
          ++j;
        }
    assert (j == table->count);
  }
  qsort (array, table->count, sizeof (*array), finalize_compare);

  /* Assign offsets, using tail merging (sharing suffixes) if possible.  */
  array[0]->offset = 0;
  for (uint32_t j = 1; j < table->count; ++j)
    {
      struct stringtable_entry *previous = array[j - 1];
      struct stringtable_entry *current = array[j];
      if (previous->length >= current->length
          && memcmp (&previous->string[previous->length - current->length],
                     current->string, current->length) == 0)
        current->offset = (previous->offset + previous->length
                           - current->length);
      else if (__builtin_add_overflow (previous->offset,
                                       previous->length + 1,
                                       &current->offset))
        error (EXIT_FAILURE, 0, _("String table is too large"));
    }

  /* Allocate the result string.  */
  {
    struct stringtable_entry *last = array[table->count - 1];
    if (__builtin_add_overflow (last->offset, last->length + 1,
                                &result->size))
      error (EXIT_FAILURE, 0, _("String table is too large"));
  }
  /* The strings are copied from the hash table, so the array is no
     longer needed.  */
  free (array);
  result->strings = xcalloc (result->size, 1);

  /* Copy the strings.  */
  for (uint32_t i = 0; i < table->allocated; ++i)
    for (struct stringtable_entry *e = table->entries[i]; e != NULL;
         e = e->next)
      if (result->strings[e->offset] == '\0')
        memcpy (&result->strings[e->offset], e->string, e->length + 1);
}
