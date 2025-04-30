/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Bruno Haible <haible@clisp.cons.org>, 2000.

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

#include <stdint.h>

/* Construction of sparse 3-level tables.
   See wchar-lookup.h or coll-lookup.h for their structure and the
   meaning of p and q.

   Before including this file, set
     TABLE        to the name of the structure to be defined
     ELEMENT      to the type of every entry
     DEFAULT      to the default value for empty entries
     ITERATE      if you want the TABLE_iterate function to be defined
     NO_ADD_LOCALE  if you don't want the add_locale_TABLE function
		    to be defined

   This will define

     struct TABLE;
     void TABLE_init (struct TABLE *t);
     ELEMENT TABLE_get (struct TABLE *t, uint32_t wc);
     void TABLE_add (struct TABLE *t, uint32_t wc, ELEMENT value);
     void TABLE_iterate (struct TABLE *t,
			 void (*fn) (uint32_t wc, ELEMENT value));
     void add_locale_TABLE (struct locale_file *file, struct TABLE *t);
*/

#define CONCAT(a,b) CONCAT1(a,b)
#define CONCAT1(a,b) a##b

struct TABLE
{
  /* Parameters.  */
  unsigned int p;
  unsigned int q;
  /* Working representation.  */
  size_t level1_alloc;
  size_t level1_size;
  uint32_t *level1;
  size_t level2_alloc;
  size_t level2_size;
  uint32_t *level2;
  size_t level3_alloc;
  size_t level3_size;
  ELEMENT *level3;
  /* Size of compressed representation.  */
  size_t result_size;
};

/* Initialize.  Assumes t->p and t->q have already been set.  */
static inline void
CONCAT(TABLE,_init) (struct TABLE *t)
{
  t->level1 = NULL;
  t->level1_alloc = t->level1_size = 0;
  t->level2 = NULL;
  t->level2_alloc = t->level2_size = 0;
  t->level3 = NULL;
  t->level3_alloc = t->level3_size = 0;
}

/* Marker for an empty slot.  This has the value 0xFFFFFFFF, regardless
   whether 'int' is 16 bit, 32 bit, or 64 bit.  */
#define EMPTY ((uint32_t) ~0)

/* Retrieve an entry.  */
static inline ELEMENT
__attribute ((always_inline))
CONCAT(TABLE,_get) (struct TABLE *t, uint32_t wc)
{
  uint32_t index1 = wc >> (t->q + t->p);
  if (index1 < t->level1_size)
    {
      uint32_t lookup1 = t->level1[index1];
      if (lookup1 != EMPTY)
	{
	  uint32_t index2 = ((wc >> t->p) & ((1 << t->q) - 1))
			    + (lookup1 << t->q);
	  uint32_t lookup2 = t->level2[index2];
	  if (lookup2 != EMPTY)
	    {
	      uint32_t index3 = (wc & ((1 << t->p) - 1))
				+ (lookup2 << t->p);
	      ELEMENT lookup3 = t->level3[index3];

	      return lookup3;
	    }
	}
    }
  return DEFAULT;
}

/* Add one entry.  */
static void
CONCAT(TABLE,_add) (struct TABLE *t, uint32_t wc, ELEMENT value)
{
  uint32_t index1 = wc >> (t->q + t->p);
  uint32_t index2 = (wc >> t->p) & ((1 << t->q) - 1);
  uint32_t index3 = wc & ((1 << t->p) - 1);
  size_t i, i1, i2;

  if (value == CONCAT(TABLE,_get) (t, wc))
    return;

  if (index1 >= t->level1_size)
    {
      if (index1 >= t->level1_alloc)
	{
	  size_t alloc = 2 * t->level1_alloc;
	  if (alloc <= index1)
	    alloc = index1 + 1;
	  t->level1 = (uint32_t *) xrealloc ((char *) t->level1,
					     alloc * sizeof (uint32_t));
	  t->level1_alloc = alloc;
	}
      while (index1 >= t->level1_size)
	t->level1[t->level1_size++] = EMPTY;
    }

  if (t->level1[index1] == EMPTY)
    {
      if (t->level2_size == t->level2_alloc)
	{
	  size_t alloc = 2 * t->level2_alloc + 1;
	  t->level2 = (uint32_t *) xrealloc ((char *) t->level2,
					     (alloc << t->q) * sizeof (uint32_t));
	  t->level2_alloc = alloc;
	}
      i1 = t->level2_size << t->q;
      i2 = (t->level2_size + 1) << t->q;
      for (i = i1; i < i2; i++)
	t->level2[i] = EMPTY;
      t->level1[index1] = t->level2_size++;
    }

  index2 += t->level1[index1] << t->q;

  if (t->level2[index2] == EMPTY)
    {
      if (t->level3_size == t->level3_alloc)
	{
	  size_t alloc = 2 * t->level3_alloc + 1;
	  t->level3 = (ELEMENT *) xrealloc ((char *) t->level3,
					    (alloc << t->p) * sizeof (ELEMENT));
	  t->level3_alloc = alloc;
	}
      i1 = t->level3_size << t->p;
      i2 = (t->level3_size + 1) << t->p;
      for (i = i1; i < i2; i++)
	t->level3[i] = DEFAULT;
      t->level2[index2] = t->level3_size++;
    }

  index3 += t->level2[index2] << t->p;

  t->level3[index3] = value;
}

#ifdef ITERATE
/* Apply a function to all entries in the table.  */
static void
CONCAT(TABLE,_iterate) (struct TABLE *t,
			void (*fn) (uint32_t wc, ELEMENT value))
{
  uint32_t index1;
  for (index1 = 0; index1 < t->level1_size; index1++)
    {
      uint32_t lookup1 = t->level1[index1];
      if (lookup1 != EMPTY)
	{
	  uint32_t lookup1_shifted = lookup1 << t->q;
	  uint32_t index2;
	  for (index2 = 0; index2 < (1 << t->q); index2++)
	    {
	      uint32_t lookup2 = t->level2[index2 + lookup1_shifted];
	      if (lookup2 != EMPTY)
		{
		  uint32_t lookup2_shifted = lookup2 << t->p;
		  uint32_t index3;
		  for (index3 = 0; index3 < (1 << t->p); index3++)
		    {
		      ELEMENT lookup3 = t->level3[index3 + lookup2_shifted];
		      if (lookup3 != DEFAULT)
			fn ((((index1 << t->q) + index2) << t->p) + index3,
			    lookup3);
		    }
		}
	    }
	}
    }
}
#endif

#ifndef NO_ADD_LOCALE
/* Finalize and shrink.  */
static void
CONCAT(add_locale_,TABLE) (struct locale_file *file, struct TABLE *t)
{
  size_t i, j, k;
  uint32_t reorder3[t->level3_size];
  uint32_t reorder2[t->level2_size];
  uint32_t level2_offset, level3_offset, last_offset;

  /* Uniquify level3 blocks.  */
  k = 0;
  for (j = 0; j < t->level3_size; j++)
    {
      for (i = 0; i < k; i++)
	if (memcmp (&t->level3[i << t->p], &t->level3[j << t->p],
		    (1 << t->p) * sizeof (ELEMENT)) == 0)
	  break;
      /* Relocate block j to block i.  */
      reorder3[j] = i;
      if (i == k)
	{
	  if (i != j)
	    memcpy (&t->level3[i << t->p], &t->level3[j << t->p],
		    (1 << t->p) * sizeof (ELEMENT));
	  k++;
	}
    }
  t->level3_size = k;

  for (i = 0; i < (t->level2_size << t->q); i++)
    if (t->level2[i] != EMPTY)
      t->level2[i] = reorder3[t->level2[i]];

  /* Uniquify level2 blocks.  */
  k = 0;
  for (j = 0; j < t->level2_size; j++)
    {
      for (i = 0; i < k; i++)
	if (memcmp (&t->level2[i << t->q], &t->level2[j << t->q],
		    (1 << t->q) * sizeof (uint32_t)) == 0)
	  break;
      /* Relocate block j to block i.  */
      reorder2[j] = i;
      if (i == k)
	{
	  if (i != j)
	    memcpy (&t->level2[i << t->q], &t->level2[j << t->q],
		    (1 << t->q) * sizeof (uint32_t));
	  k++;
	}
    }
  t->level2_size = k;

  for (i = 0; i < t->level1_size; i++)
    if (t->level1[i] != EMPTY)
      t->level1[i] = reorder2[t->level1[i]];

  /* Create and fill the resulting compressed representation.  */
  last_offset =
    5 * sizeof (uint32_t)
    + t->level1_size * sizeof (uint32_t)
    + (t->level2_size << t->q) * sizeof (uint32_t)
    + (t->level3_size << t->p) * sizeof (ELEMENT);
  t->result_size = LOCFILE_ALIGN_UP (last_offset);

  level2_offset =
    5 * sizeof (uint32_t)
    + t->level1_size * sizeof (uint32_t);
  level3_offset =
    5 * sizeof (uint32_t)
    + t->level1_size * sizeof (uint32_t)
    + (t->level2_size << t->q) * sizeof (uint32_t);

  start_locale_structure (file);
  add_locale_uint32 (file, t->q + t->p);
  add_locale_uint32 (file, t->level1_size);
  add_locale_uint32 (file, t->p);
  add_locale_uint32 (file, (1 << t->q) - 1);
  add_locale_uint32 (file, (1 << t->p) - 1);

  for (i = 0; i < t->level1_size; i++)
    add_locale_uint32
      (file,
       t->level1[i] == EMPTY
       ? 0
       : (t->level1[i] << t->q) * sizeof (uint32_t) + level2_offset);

  for (i = 0; i < (t->level2_size << t->q); i++)
    add_locale_uint32
      (file,
       t->level2[i] == EMPTY
       ? 0
       : (t->level2[i] << t->p) * sizeof (ELEMENT) + level3_offset);

  if (sizeof (ELEMENT) == 1)
    add_locale_raw_data (file, t->level3, t->level3_size << t->p);
  else if (sizeof (ELEMENT) == sizeof (uint32_t))
    add_locale_uint32_array (file, (uint32_t *) t->level3,
			     t->level3_size << t->p);
  else
    abort ();
  align_locale_data (file, LOCFILE_ALIGN);
  end_locale_structure (file);

  if (t->level1_alloc > 0)
    free (t->level1);
  if (t->level2_alloc > 0)
    free (t->level2);
  if (t->level3_alloc > 0)
    free (t->level3);
}
#endif

#undef EMPTY
#undef TABLE
#undef ELEMENT
#undef DEFAULT
#undef ITERATE
#undef NO_ADD_LOCALE
