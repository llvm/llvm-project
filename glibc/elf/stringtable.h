/* String tables for ld.so.cache construction.
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

#ifndef _STRINGTABLE_H
#define _STRINGTABLE_H

#include <stddef.h>
#include <stdint.h>

/* An entry in the string table.  Only the length and string fields are
   expected to be used outside the string table code.  */
struct stringtable_entry
{
  struct stringtable_entry *next; /* For collision resolution.  */
  uint32_t length;                /* Length of then string.  */
  uint32_t offset;                /* From start of finalized table.  */
  char string[];                  /* Null-terminated string.  */
};

/* A string table.  Zero-initialization produces a valid atable.  */
struct stringtable
{
  struct stringtable_entry **entries;  /* Array of hash table buckets.  */
  uint32_t count;                 /* Number of elements in the table.  */
  uint32_t allocated;             /* Length of the entries array.  */
};

/* Adds STRING to TABLE.  May return the address of an existing entry.  */
struct stringtable_entry *stringtable_add (struct stringtable *table,
                                           const char *string);

/* Result of stringtable_finalize.  SIZE bytes at STRINGS should be
   written to the file.  */
struct stringtable_finalized
{
  char *strings;
  size_t size;
};

/* Assigns offsets to string table entries and computes the serialized
   form of the string table.  */
void stringtable_finalize (struct stringtable *table,
                           struct stringtable_finalized *result);

/* Deallocate the string table (but not the TABLE pointer itself).
   (The table can be re-used for adding more strings without
   initialization.)  */
void stringtable_free (struct stringtable *table);

#endif /* _STRINGTABLE_H */
