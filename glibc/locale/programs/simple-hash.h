/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.org>, 1995.

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

#ifndef _SIMPLE_HASH_H
#define _SIMPLE_HASH_H

#include <inttypes.h>
#include <obstack.h>
#include <stdint.h>

typedef struct hash_table
{
  unsigned long int size;
  unsigned long int filled;
  void *first;
  void *table;
  struct obstack mem_pool;
}
hash_table;


extern int init_hash (hash_table *htab, unsigned long int init_size) __THROW;
extern int delete_hash (hash_table *htab) __THROW;
extern int insert_entry (hash_table *htab, const void *key, size_t keylen,
			 void *data) __THROW;
extern int find_entry (const hash_table *htab, const void *key, size_t keylen,
		       void **result) __THROW;
extern int set_entry (hash_table *htab, const void *key, size_t keylen,
		      void *newval) __THROW;

extern int iterate_table (const hash_table *htab, void **ptr,
			  const void **key, size_t *keylen, void **data)
     __THROW;

extern uint32_t compute_hashval (const void *key, size_t keylen)
     __THROW;
extern unsigned long int next_prime (unsigned long int seed) __THROW;

#endif /* simple-hash.h */
