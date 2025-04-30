/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "flang/ADT/hash.h"
#include "flang/Error/pgerror.h"
#include <stdlib.h>
#include <string.h>

#if UNIT_TESTING
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include "cmockery.h"
#endif /* UNIT_TESTING */

/** \file
 * \brief Open addressing, quadratically probed hash tables.
 *
 * All keys are kept in a single array with 2^n entries. A key is first looked
 * up at index hash(k) mod 2^k, then following a quadratic sequence of offsets.
 * The quadratic probing has decent locality of reference for the first few
 * probes.
 *
 * The search terminates when the key is found or a NULL is encountered. A
 * special ERASED marker is used to avoid severing probe sequences when erasing
 * keys.
 */

#ifdef HOST_WIN
#define ERASED (hash_key_t) ~0ULL
#define LONG long long
#else
#define ERASED (hash_key_t) ~0UL
#define LONG long
#endif
#define ISKEY(k) ((k) != NULL && (k) != ERASED)

struct hashset_ {
  hash_functions_t func;
  hash_key_t *table;

  /** Table size minus 1 is a bit mask. */
  unsigned mask;

  /** Number of ISKEY() entries in table. */
  unsigned entries;

  /** Number of ERASED entries in table. */
  unsigned erased;
};

/** \brief A hashmap adds a table of data pointers after the hashset table. */
struct hashmap_ {
  struct hashset_ set;
};

#define DTABLE(h) (((h)->table + 1) + (h)->mask)

/*
   Quadrating probing sequence.

   Given an initial hash value h, we probe at table indices

       h, h+2, h+1+2, h+1+2+3, ... ( mod 2^n ), or

       (i+1)i/2 ( mod 2^n ), for i = 0 .. 2^n.

   This sequence visits the entire table without duplicates. Suppose two
   indexes in the sequence are the same entry:

                (i+1)i/2 = (j+1)j/2  ( mod 2^n )
   ==>           i^2 + i = j^2 + j   ( mod 2^(n+1) )
   ==>  (i+j)(i-j) + i-j = 0         ( mod 2^(n+1) )
   ==>      (i+j+1)(i-j) = 0         ( mod 2^(n+1) )

   Look at the difference of the factors on the left-hand side:

         (i+j+1) - (i-j) = 2j + 1

   This is an odd number, so one of the factors must be odd and thus coprime
   with 2^(n+1). Therefore, 2^(n+1) must divide the other factor. The range of
   the factors can be determined from the ranges of i and j:

              1 <= i+j+1 <= 2^(n+1)-1, and
       -(2^n-1) <=  i-j  <= 2^n-1.

   The only way 2^(n+1) can divide one of the two factors is if 1 = j.

   q.e.d.
*/

/** \brief Search for key, return index that terminated the search. */
static unsigned
search(hashset_t h, hash_key_t key)
{
  unsigned p, s = 1;
  hash_equality_t eq = h->func.equals;

  assert(ISKEY(key), "Invalid key for hash", HKEY2INT(key), ERR_Fatal);

  p = h->func.hash(key) & h->mask;
  if (eq) {
    /* General case where we have to call eq() to determine if keys are
     * equivalent. */
    while (h->table[p]) {
      if (h->table[p] != ERASED && eq(key, h->table[p]))
        break;
      p = (p + s++) & h->mask;
    }
  } else {
    /* Optimized case when eq is NULL and we simply test for pointer
     * equality. */
    while (h->table[p]) {
      if (key == h->table[p])
        break;
      p = (p + s++) & h->mask;
    }
  }
  return p;
}

/** \brief Find insertion point for key. */
static unsigned
insertion_point(hashset_t h, hash_key_t key)
{
  unsigned p, s = 1;

  assert(ISKEY(key), "Invalid key for hash", HKEY2INT(key), ERR_Fatal);

  p = h->func.hash(key) & h->mask;
  while (ISKEY(h->table[p])) {
    p = (p + s++) & h->mask;
  }
  return p;
}

/*
   The number of NULL entries is kept above 1/8th of the whole table to avoid
   too long probe sequences. The table is rehashed before the NULL entries drop
   below that.

   The ideal table size is the smallest power of two that keeps the load factor
   below 2/3.
*/

#define NEED_REHASH(h) ((h)->entries + (h)->erased >= (h)->mask - (h)->mask / 8)
#define MINSIZE 16

static unsigned
mask_for_entries(unsigned n)
{
  unsigned m = (n + n / 2) | (MINSIZE - 1);

  /* Arithmetic overflow happens after we have billions of entries. */
  assert(m > n, "Hash table full", n, ERR_Fatal);

  /* Round up to the next power of two minus one. */
  m |= m >> 1;
  m |= m >> 2;
  m |= m >> 4;
  m |= m >> 8;
  m |= m >> 16;

  return m;
}

/* Compute h->mask from h->entries and allocate table and dtable. */
static void
alloc_tables(hashset_t h, int with_dtable)
{
  h->mask = mask_for_entries(h->entries);

  /* Allocate table + dtable in one calloc(). */
  if (with_dtable)
    h->table = (const void**) calloc(
        1 + (size_t)h->mask, sizeof(hash_key_t) + sizeof(hash_data_t));
  else
    h->table = (const void**) calloc(1 + (size_t)h->mask, sizeof(hash_key_t));
}

/* Resize and rehash table to get rid of ERASED entries.

   If not NULL, dtable points to a table of data entries that should be
   rearranged the same way as h->table.
 */
static void
rehash(hashset_t h, int with_dtable)
{
  hash_key_t *old_table = h->table;
  hash_data_t *old_dtable = DTABLE(h);
  size_t n, old_size = 1 + (size_t)h->mask;
  hash_data_t *new_dtable = NULL;

  alloc_tables(h, with_dtable);
  new_dtable = DTABLE(h);

  /* There will be no ERASED markers after the rehash. */
  h->erased = 0;

  for (n = 0; n < old_size; n++) {
    hash_key_t key = old_table[n];
    if (ISKEY(key)) {
      unsigned i = insertion_point(h, key);
      h->table[i] = key;
      if (with_dtable)
        new_dtable[i] = old_dtable[n];
    }
  }

  free((void*)old_table);
}

hashset_t
hashset_alloc(hash_functions_t f)
{
  hashset_t h = (hashset_t) calloc(1, sizeof(struct hashset_));
  h->func = f;
  alloc_tables(h, 0);
  return h;
}

hashmap_t
hashmap_alloc(hash_functions_t f)
{
  hashmap_t h = (hashmap_t) calloc(1, sizeof(struct hashmap_));
  h->set.func = f;
  alloc_tables(&h->set, 1);
  return h;
}

void
hashset_free(hashset_t h)
{
  free((void*)h->table);
  memset(h, 0, sizeof(*h));
  free((void*)h);
}

void
hashmap_free(hashmap_t h)
{
  free((void*)h->set.table);
  memset(h, 0, sizeof(*h));
  free((void*)h);
}

void
hashset_clear(hashset_t h)
{
  free((void*)h->table);
  h->entries = h->erased = 0;
  alloc_tables(h, 0);
}

void
hashmap_clear(hashmap_t h)
{
  free((void*)h->set.table);
  h->set.entries = h->set.erased = 0;
  alloc_tables(&h->set, 1);
}

unsigned
hashset_size(hashset_t h)
{
  return h->entries;
}

unsigned
hashmap_size(hashmap_t h)
{
  return h->set.entries;
}

hash_key_t
hashset_lookup(hashset_t h, hash_key_t key)
{
  return h->table[search(h, key)];
}

hash_key_t
hashmap_lookup(hashmap_t h, hash_key_t key, hash_data_t *data)
{
  unsigned i = search(&h->set, key);
  hash_key_t k = h->set.table[i];
  if (data && k)
    *data = DTABLE(&h->set)[i];
  return k;
}

void
hashset_insert(hashset_t h, hash_key_t key)
{
  unsigned i;

#if DEBUG
  assert(hashset_lookup(h, key) == NULL, "Duplicate hash key", 0, ERR_Fatal);
#endif

  if (NEED_REHASH(h))
    rehash(h, 0);

  i = insertion_point(h, key);
  if (h->table[i] == ERASED)
    h->erased--;
  h->table[i] = key;
  h->entries++;
}

void
hashmap_insert(hashmap_t h, hash_key_t key, hash_data_t data)
{
  unsigned i;

#if DEBUG
  assert(hashmap_lookup(h, key, NULL) == NULL, "Duplicate hash key", 0, ERR_Fatal);
#endif

  if (NEED_REHASH(&h->set))
    rehash(&h->set, 1);

  i = insertion_point(&h->set, key);
  if (h->set.table[i] == ERASED)
    h->set.erased--;
  h->set.table[i] = key;
  DTABLE(&h->set)[i] = data;
  h->set.entries++;
}

hash_key_t
hashset_replace(hashset_t h, hash_key_t key)
{
  unsigned i = search(h, key);
  hash_key_t old = h->table[i];

  if (ISKEY(old)) {
    h->table[i] = key;
    return old;
  }

  hashset_insert(h, key);
  return NULL;
}

hash_key_t
hashmap_replace(hashmap_t h, hash_key_t key, hash_data_t *data)
{
  unsigned i = search(&h->set, key);
  hash_key_t old = h->set.table[i];

  if (ISKEY(old)) {
    hash_data_t old_data = DTABLE(&h->set)[i];
    h->set.table[i] = key;
    DTABLE(&h->set)[i] = *data;
    *data = old_data;
    return old;
  }

  hashmap_insert(h, key, *data);
  return NULL;
}

/* We never rehash while erasing elements. The rehash() at insertion can also
 * shrink the table. */
hash_key_t
hashset_erase(hashset_t h, hash_key_t key)
{
  unsigned i = search(h, key);
  hash_key_t old = h->table[i];

  if (!old)
    return NULL;

  h->table[i] = ERASED;
  h->erased++;
  h->entries--;
  return old;
}

hash_key_t
hashmap_erase(hashmap_t h, hash_key_t key, hash_data_t *data)
{
  unsigned i = search(&h->set, key);
  hash_key_t old = h->set.table[i];

  if (!old)
    return NULL;

  h->set.table[i] = ERASED;
  h->set.erased++;
  h->set.entries--;
  if (data)
    *data = DTABLE(&h->set)[i];
  return old;
}

void
hashset_iterate(hashset_t h, void (*f)(hash_key_t, void *context),
                void *context)
{
  size_t n, size = 1 + (size_t)h->mask;

  for (n = 0; n < size; n++) {
    hash_key_t key = h->table[n];
    if (ISKEY(key))
      f(key, context);
  }
}


typedef struct  {
    hash_key_t table;
    hash_data_t dtable;
} sortmap_t;

static int
string_cmp(sortmap_t* a, sortmap_t* b)
{
  return strcmp((const char *)(a->table), (const char *)(b->table));
}

void
hashmap_sort(hashmap_t h, void (*f)(hash_key_t, hash_data_t, void *context),
                void *context)
{
  typedef int (*compare_func_t)(const void*, const void*);
  size_t n,i, size = 1 + (size_t)h->set.mask;
  size_t count = 0;
  hash_data_t *dtable = DTABLE(&h->set);
  sortmap_t *mysortmap;

  for (n = 0; n < size; n++) {
    hash_key_t key = h->set.table[n];
    if (ISKEY(key)) 
      count++;
  }

  mysortmap = (sortmap_t*) malloc(count * sizeof(sortmap_t));
 
  i = 0;
  for (n = 0; n < size; n++) {
    hash_key_t key = h->set.table[n];
    if (ISKEY(key)) {
      mysortmap[i].table =(hash_key_t)h->set.table[n];
      mysortmap[i].dtable = dtable[n];
      i++;
    } 
  }

  qsort(mysortmap, count, sizeof(sortmap_t), (compare_func_t)string_cmp);
  for (n = 0; n < count; n++) {
    f(mysortmap[n].table, mysortmap[n].dtable, context);
  }

  free(mysortmap);
}

void
hashmap_iterate(hashmap_t h, void (*f)(hash_key_t, hash_data_t, void *context),
                void *context)
{
  size_t n, size = 1 + (size_t)h->set.mask;
  hash_data_t *dtable = DTABLE(&h->set);

  for (n = 0; n < size; n++) {
    hash_key_t key = h->set.table[n];
    if (ISKEY(key))
      f(key, dtable[n], context);
  }
}

/* String hashing */
static hash_value_t
string_hash(hash_key_t key)
{
  const unsigned char *p = (const unsigned char *)key;
  hash_accu_t hacc = HASH_ACCU_INIT;
  for (; *p; p++)
    HASH_ACCU_ADD(hacc, *p);
  HASH_ACCU_FINISH(hacc);
  return HASH_ACCU_VALUE(hacc);
}

static int
string_equals(hash_key_t a, hash_key_t b)
{
  return strcmp((const char *)a, (const char *)b) == 0;
}

const hash_functions_t hash_functions_strings = {string_hash, string_equals};

/* Direct hashing. */
static hash_value_t
direct_hash(hash_key_t key)
{
  unsigned LONG k = (unsigned LONG)key;
  hash_accu_t hacc = HASH_ACCU_INIT;
  HASH_ACCU_ADD(hacc, k);
  /* On an LP64 system, this ignores the high 8 bits of k. That's OK if k
   * represents a pointer, since current 64-bit systems don't use the high
   * bits of addresses. A shift by 32 would be undefined when long is a
   * 32-bit type. */
  HASH_ACCU_ADD(hacc, k >> 24);
  HASH_ACCU_FINISH(hacc);
  return HASH_ACCU_VALUE(hacc);
}

const hash_functions_t hash_functions_direct = {direct_hash, NULL};

/* Everything below is only for testing the implementation. */
#if UNIT_TESTING

#include <stdio.h>

static void
table_size(void **state)
{
  assert_int_equal(mask_for_entries(0), 15);
  assert_int_equal(mask_for_entries(1), 15);
  assert_int_equal(mask_for_entries(10), 15);
  assert_int_equal(mask_for_entries(11), 31);
  assert_int_equal(mask_for_entries(20), 31);
  assert_int_equal(mask_for_entries(21), 31);
  assert_int_equal(mask_for_entries(22), 63);
  assert_int_equal(mask_for_entries(0x80000000), 0xffffffff);
  assert_int_equal(mask_for_entries(0xa0000000), 0xffffffff);
  /* 0xb0000000 will overflow and assert, but expect_assert_failure()
   * segfaults... */
}

static void
hash_funcs(void **state)
{
  assert_int_equal(string_hash(""), 0);
  assert_int_equal(string_hash("a"), 0xca2e9442);
  assert_int_equal(string_hash("A"), 0x820103f0);
  assert_int_equal(string_hash("foo"), 0x238678dd);

  assert_int_equal(direct_hash((hash_key_t)0), 0);
  assert_int_equal(direct_hash((hash_key_t)1), 0x20e9c0b3);
  assert_int_equal(direct_hash((hash_key_t)2), 0x41d38166);
}

static void
basic_direct(void **state)
{
  unsigned LONG i;
  hashset_t h = hashset_alloc(hash_functions_direct);
  assert_true(h != NULL);
  assert_int_equal(hashset_size(h), 0);
  assert_int_equal(h->mask, MINSIZE - 1);

  assert_int_equal(hashset_lookup(h, (hash_key_t)1), 0);
  assert_int_equal(hashset_lookup(h, (hash_key_t)2), 0);

  hashset_insert(h, (hash_key_t)1);
  assert_int_equal(hashset_lookup(h, (hash_key_t)1), 1);
  assert_int_equal(hashset_lookup(h, (hash_key_t)2), 0);

  assert_int_equal(hashset_replace(h, (hash_key_t)1), 1);
  assert_int_equal(hashset_replace(h, (hash_key_t)2), 0);

  for (i = 3; i <= 14; i++)
    hashset_insert(h, (hash_key_t)i);

  /* Table has 16 entries, 2 are NULL for exactly 7/8 load factor. */
  assert_int_equal(hashset_size(h), 14);
  assert_int_equal(h->mask, MINSIZE - 1);

  /* Trigger a rehash when we add the 15th element. */
  hashset_insert(h, (hash_key_t)15);
  assert_int_equal(hashset_size(h), 15);
  assert_int_equal(h->mask, 2 * MINSIZE - 1);

  /* Remove odd entries. */
  for (i = 1; i <= 15; i += 2)
    assert_int_equal(hashset_erase(h, (hash_key_t)i), i);
  assert_int_equal(hashset_size(h), 7);
  assert_int_equal(h->mask, 2 * MINSIZE - 1);
  assert_int_equal(h->erased, 8);

  /* Remove everything. */
  for (i = 1; i <= 15; i++)
    assert_int_equal(hashset_erase(h, (hash_key_t)i), i % 2 ? 0 : i);

  /* Set is empty, but hasn't rehashed yet. */
  assert_int_equal(hashset_size(h), 0);
  assert_int_equal(h->mask, 2 * MINSIZE - 1);
  assert_int_equal(h->erased, 15);

  for (i = 1; i <= 20; i++)
    assert_int_equal(hashset_erase(h, (hash_key_t)i), 0);
  assert_int_equal(hashset_size(h), 0);
  assert_int_equal(h->mask, 2 * MINSIZE - 1);

  /* Insert triggers a table shrink. Eventually. The exact timing depends on
   * the hash function behavior. */
  for (i = 1; i <= 10; i++)
    hashset_insert(h, (hash_key_t)(100 + i));
  assert_int_equal(h->mask, 2 * MINSIZE - 1);
  for (i = 1; i <= 10; i++)
    hashset_erase(h, (hash_key_t)(100 + i));
  assert_int_equal(h->mask, 2 * MINSIZE - 1);
  for (i = 1; i <= 10; i++)
    hashset_insert(h, (hash_key_t)(200 + i));
  assert_int_equal(h->mask, 2 * MINSIZE - 1);
  for (i = 1; i <= 10; i++)
    hashset_erase(h, (hash_key_t)(200 + i));
  assert_int_equal(h->mask, 2 * MINSIZE - 1);
  for (i = 1; i <= 10; i++)
    hashset_insert(h, (hash_key_t)(300 + i));
  assert_int_equal(h->mask, 2 * MINSIZE - 1);
  for (i = 1; i <= 10; i++)
    hashset_erase(h, (hash_key_t)(300 + i));
  assert_int_equal(h->mask, 2 * MINSIZE - 1);
  for (i = 1; i <= 10; i++)
    hashset_insert(h, (hash_key_t)(400 + i));
  assert_int_equal(h->erased, 0);
  assert_int_equal(h->mask, MINSIZE - 1);

  hashset_free(h);
}

static void
free_iterator(hash_key_t key, void *context)
{
  ++*(unsigned *)context;
  free((void *)key);
}

static void
basic_string(void **state)
{
  unsigned i;
  hashset_t h = hashset_alloc(hash_functions_strings);
  const char *strs[] = {"foo", "bar", "baz", "quux"};
  char buf[20];

  for (i = 0; i != 4; i++)
    hashset_insert(h, (hash_key_t)strs[i]);

  strcpy(buf, "foo");
  assert_int_equal(hashset_lookup(h, buf), strs[0]);
  strcpy(buf, "fooo");
  assert_int_not_equal(hashset_lookup(h, buf), strs[0]);

  for (i = 0; i < 1000; i++) {
    char *x = malloc(10);
    sprintf(x, "x%d", i);
    hashset_insert(h, x);
  }

  for (i = 0; i != 4; i++)
    hashset_erase(h, (hash_key_t)strs[i]);

  i = 0;
  assert_int_equal(hashset_size(h), 1000);
  hashset_iterate(h, free_iterator, &i);
  assert_int_equal(i, 1000);

  hashset_free(h);
}

static hash_value_t
worst_hash_ever(hash_key_t key)
{
  return 42;
}

static const hash_functions_t hash_functions_worst = {worst_hash_ever, NULL};

static void
worst_case(void **state)
{
  unsigned LONG i;
  hashset_t h = hashset_alloc(hash_functions_worst);

  for (i = 1; i <= 14; i++)
    hashset_insert(h, (hash_key_t)i);

  /* Table has 16 entries, 2 are NULL for exactly 7/8 load factor. */
  assert_int_equal(hashset_size(h), 14);
  assert_int_equal(h->mask, MINSIZE - 1);

  /* The bad hash function stresses the probing sequence to find the NULLS.
   * The insert and the lookup can both loop infinitely if their probing
   * sequence doesn't completely cover the table. */
  for (i = 15; i <= 1000; i++)
    hashset_insert(h, (hash_key_t)i);

  for (i = 1; i <= 2000; i += 100)
    assert_int_equal(hashset_lookup(h, (hash_key_t)i), i <= 1000 ? i : 0);

  hashset_free(h);
}

static void
basic_map(void **state)
{
  hashmap_t h = hashmap_alloc(hash_functions_direct);
  hash_data_t datum;
  const char *s1 = "foo", *s2 = "bar";
  unsigned LONG i;

  assert_int_equal(hashmap_size(h), 0);

  /* Lookup nonexistent data with NULL data pointer. */
  assert_int_equal(hashmap_lookup(h, (hash_key_t)1, NULL), 0);

  /* Lookup nonexistent data with non-NULL data pointer. */
  datum = &h;
  assert_int_equal(hashmap_lookup(h, (hash_key_t)1, &datum), 0);
  assert_int_equal(datum, &h);

  hashmap_insert(h, (hash_key_t)1, (hash_data_t)s1);
  hashmap_insert(h, (hash_key_t)2, (hash_data_t)s2);

  /* We support lookup with NULL data pointer. */
  assert_int_equal(hashmap_lookup(h, (hash_key_t)1, NULL), 1);

  datum = (hash_data_t)s2;
  assert_int_equal(hashmap_lookup(h, (hash_key_t)1, &datum), 1);
  assert_int_equal(datum, s1);

  /* Replace where no previous key existed. */
  datum = (hash_data_t)s2;
  assert_int_equal(hashmap_replace(h, (hash_key_t)3, &datum), 0);
  assert_int_equal(datum, s2);

  /* Replace previous key. Should return previous data. */
  datum = (hash_data_t)s1;
  assert_int_equal(hashmap_replace(h, (hash_key_t)3, &datum), 3);
  assert_int_equal(datum, s2);

  /* Force rehash. Verify that data entries are copied correctly. */
  for (i = 1; i < 100; i++) {
    datum = (hash_data_t)(i % 7);
    hashmap_replace(h, (hash_key_t)i, &datum);
  }
  for (i = 1; i < 100; i++) {
    assert_int_equal(hashmap_lookup(h, (hash_key_t)i, &datum), i);
    assert_int_equal(datum, i % 7);
  }

  hashmap_free(h);
}

int
main()
{
  const UnitTest tests[] = {
      unit_test(table_size),   unit_test(hash_funcs), unit_test(basic_direct),
      unit_test(basic_string), unit_test(worst_case), unit_test(basic_map),
  };
  return run_tests(tests);
}

#endif /* UNIT_TESTING */
