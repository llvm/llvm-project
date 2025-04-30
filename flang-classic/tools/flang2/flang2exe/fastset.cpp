/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Implementation of fast set representation for small integers
 *
 * The fast set operations with O(1) time are implemented with small inline
 * functions in fastset.h.  This file contains those that need not be fast or
 * that have linear time complexity, at worst.
 */

#ifndef UNITTEST
#include "gbldefs.h"
#include "error.h"
#else
#define assert(cond,message,val,severity) ((cond)?(void)0:(void)abort())
#endif
#include <stdlib.h>
#include <string.h>
#include "fastset.h"
#include "go.h"

/** \brief Allocations are rounded up to amortize the allocation time. */
#define FASTSET_GRANULARITY 64

/** \brief Verifies that each member of the set corresponds to a valid index
 * entry. */
void
fastset_check(const fastset *set)
{
  if (set) {
    unsigned j;
    for (j = 0; j < set->members; ++j) {
      CHECK(set->member[j] < set->limit);
      CHECK(set->index[set->member[j]] == j);
    }
  }
}

/** \brief Callback for fastset_map() as a means for printing sets. */
static void *
print_int(void *p, int x)
{
  dbgprintf(" %d", x);
  return p;
}

/** \brief Debug printing of a set. */
void
fastset_dbgprintf(const fastset *set)
{
  fastset_map(set, print_int, NULL);
}

/** \brief Releases storage occupied by a set. */
void
fastset_free(fastset *set)
{
  free(set->member);
  free(set->index);
#if DEBUG+0
  memset(set, 0x55, sizeof *set);
#endif
}

void
fastset_reinit(fastset *set)
{
  fastset_free(set);
  fastset_init(set);
}

void
fastset_resize(fastset *set, unsigned limit)
{
  limit += FASTSET_GRANULARITY - 1;
  limit &= -FASTSET_GRANULARITY;
  if (limit > set->limit) {
    unsigned i, n, *new_member, *new_index;

    /* Do *not* use realloc, since it takes time O(set->limit) and we want
       to take time O(set->members) when !DEBUG. */

    /* Index array is redundant, so it can be freed now. */
    free(set->index);

    /* Reallocate the member array. */
    n = set->members;
    new_member = (unsigned *) malloc(limit * sizeof *set->member);
    assert(new_member, "fastset_resize: out of memory", limit, ERR_Fatal);
    memcpy(new_member, set->member, n * sizeof *set->member);
    free(set->member);

    /* Allocate the index array and rebuild the index. */
    new_index = (unsigned *) malloc(limit * sizeof *set->index);
    assert(new_index, "fastset_resize: out of memory", limit, ERR_Fatal);
#if DEBUG+0
    /* It's not necessary to initialize any of this allocated storage
     * for correctness or safety, but the uninitialized index[] array
     * produces warnings from valgrind.  So overwrite it in DEBUG
     * mode.  The member[] entries will not be overindexed past
     * "->members", so leave it uninitialized.
     *
     * Unfortunately this takes time O(set->limit).
     */
    memset(new_index, 0xaa, limit * sizeof *set->index);
#endif
    for (i = 0; i < n; ++i)
      new_index[new_member[i]] = i;

    set->member = new_member;
    set->index = new_index;
    set->limit = limit;
  }
}

/** \brief Set union, destructive.
 *
 * Just call fastset_add() for every member of the second set argument.
 */
void
fastset_union(fastset *xs, const fastset *ys)
{
  unsigned j;
  for (j = 0; j < ys->members; ++j)
    fastset_add(xs, ys->member[j]);
}

/** Internal utility: keep a set's member list contiguous by moving its
 * last member into a vacated position.
 */
INLINE static void
fastset_internal_drop_at(fastset *set, unsigned idx)
{
  unsigned last = set->member[--set->members];
  set->member[set->index[last] = idx] = last;
}

/** \brief Set difference, destructive.
 *
 * Algorithm depends on the number of members in the two sets.
 */
void
fastset_difference(fastset *xs, const fastset *ys)
{
  unsigned j = 0;
  if (xs->members < ys->members) {
    /* xs is small - sweep it and test membership in ys */
    while (j < xs->members) {
      if (fastset_contains(ys, xs->member[j]))
        fastset_internal_drop_at(xs, j);
      else
        ++j;
    }
  } else if (xs == ys) {
    fastset_vacate(xs);
  } else {
    /* ys is small - delete its members from xs */
    while (j < ys->members)
      fastset_remove(xs, ys->member[j++]);
  }
}

/** \brief Set intersection, destructive.
 *
 * Algorithm depends on the number of members in the two sets.
 */
void
fastset_intersection(fastset *xs, const fastset *ys)
{
  unsigned j;
  if (xs->members <= ys->members) {
    /* xs is small - sweep it and test membership in ys */
    if (xs != ys) {
      for (j = 0; j < xs->members;) {
        if (fastset_contains(ys, xs->member[j]))
          ++j;
        else
          fastset_internal_drop_at(xs, j);
      }
    }
  } else {
    /* ys is small - sweep it and move any common members to the front
     * of xs->member[]
     */
    unsigned keep = 0;
    for (j = 0; j < ys->members; ++j) {
      unsigned y = ys->member[j];
      if (y < xs->limit) {
        unsigned idx = xs->index[y];
        if (idx < xs->members && xs->member[idx] == y) {
          unsigned tmp = xs->member[keep];
          xs->member[xs->index[y] = keep++] = y;
          xs->member[xs->index[tmp] = idx] = tmp;
        }
      }
    }
    xs->members = keep;
  }
}

/** \brief Apply a function to each member of a set. */
void *
fastset_map(const fastset *xs, void *f(void *, int), void *pointer)
{
  if (xs) {
    int j, count = xs->members;
    for (j = 0; j < count; ++j)
      pointer = f(pointer, xs->member[j]);
  }
  return pointer;
}

void
fastset_unit_tests(void)
{
#ifdef UNITTEST
  fastset a, b, c, empty;
  int j;

  fastset_init(&a);
  fastset_init(&b);
  fastset_init(&c);
  fastset_init(&empty);

  CHECK(fastset_members(&a) == 0);
  CHECK(fastset_is_empty(&a));
  fastset_check(&a);

  fastset_vacate(&a);
  CHECK(fastset_members(&a) == 0);
  CHECK(fastset_is_empty(&a));
  fastset_check(&a);
  j = fastset_pop(&a);
  CHECK(j == -1);

  fastset_add(&a, 0);
  CHECK(fastset_members(&a) == 1);
  CHECK(!fastset_is_empty(&a));
  fastset_check(&a);
  CHECK(fastset_contains(&a, 0));

  fastset_vacate(&a);
  CHECK(fastset_members(&a) == 0);
  CHECK(fastset_is_empty(&a));
  fastset_check(&a);
  j = fastset_pop(&a);
  CHECK(j == -1);

  for (j = 0; j < 1000; j += 2)
    fastset_add(&a, j);
  CHECK(fastset_members(&a) == 500);
  CHECK(!fastset_is_empty(&a));
  fastset_check(&a);
  CHECK(!fastset_contains(&a, -1));
  for (j = 0; j < 1000; j += 2)
    CHECK(fastset_contains(&a, j));
  for (j = 1; j < 1000; j += 2)
    CHECK(!fastset_contains(&a, j));
  for (j = 0; j < 1000; j += 4)
    fastset_remove(&a, j);
  CHECK(fastset_members(&a) == 250);
  CHECK(!fastset_is_empty(&a));
  fastset_check(&a);
  for (j = 0; j < 1000; ++j)
    if ((j % 4) == 2)
      CHECK(fastset_contains(&a, j));
    else
      CHECK(!fastset_contains(&a, j));

  for (j = 0; j < 250; ++j) {
    int k = fastset_pop(&a);
    CHECK(k >= 0);
  }
  fastset_check(&a);
  j = fastset_pop(&a);
  CHECK(j == -1);
  CHECK(fastset_members(&a) == 0);
  CHECK(fastset_is_empty(&a));

  for (j = 0; j < 1000; ++j) {
    if ((j % 3) == 0)
      fastset_add(&a, j);
    else
      fastset_add(&b, j);
    fastset_add(&c, j);
  }
  CHECK(fastset_members(&a) == 334);
  CHECK(!fastset_is_empty(&a));
  CHECK(fastset_members(&b) == 666);
  CHECK(!fastset_is_empty(&b));
  CHECK(fastset_members(&c) == 1000);
  CHECK(!fastset_is_empty(&c));
  fastset_check(&a);
  fastset_check(&b);
  fastset_check(&c);
  for (j = 0; j < 1000; ++j) {
    if ((j % 3) == 0) {
      CHECK(fastset_contains(&a, j));
      CHECK(!fastset_contains(&b, j));
    } else {
      CHECK(!fastset_contains(&a, j));
      CHECK(fastset_contains(&b, j));
    }
    CHECK(fastset_contains(&c, j));
  }

  fastset_union(&a, &empty);
  CHECK(fastset_members(&a) == 334);
  CHECK(!fastset_is_empty(&a));
  fastset_check(&a);
  for (j = 0; j < 1000; ++j)
    if ((j % 3) == 0)
      CHECK(fastset_contains(&a, j));
    else
      CHECK(!fastset_contains(&a, j));

  fastset_union(&a, &b);
  CHECK(fastset_members(&a) == 1000);
  CHECK(!fastset_is_empty(&a));
  fastset_check(&a);
  for (j = 0; j < 1000; ++j)
    CHECK(fastset_contains(&a, j));

  fastset_difference(&a, &b);
  CHECK(fastset_members(&a) == 334);
  CHECK(!fastset_is_empty(&a));
  fastset_check(&a);
  for (j = 0; j < 1000; ++j)
    if ((j % 3) == 0)
      CHECK(fastset_contains(&a, j));
    else
      CHECK(!fastset_contains(&a, j));

  fastset_difference(&empty, &a);
  CHECK(fastset_members(&empty) == 0);
  CHECK(fastset_is_empty(&empty));
  fastset_check(&empty);

  fastset_difference(&a, &empty);
  CHECK(fastset_members(&a) == 334);
  CHECK(!fastset_is_empty(&a));
  fastset_check(&a);
  for (j = 0; j < 1000; ++j)
    if ((j % 3) == 0)
      CHECK(fastset_contains(&a, j));
    else
      CHECK(!fastset_contains(&a, j));

  fastset_intersection(&a, &b);
  CHECK(fastset_members(&a) == 0);
  CHECK(fastset_is_empty(&a));
  fastset_check(&a);

  fastset_intersection(&b, &c);
  CHECK(fastset_members(&b) == 666);
  CHECK(!fastset_is_empty(&b));
  fastset_check(&b);
  for (j = 0; j < 1000; ++j)
    if ((j % 3) != 0)
      CHECK(fastset_contains(&b, j));

  fastset_intersection(&b, &empty);
  CHECK(fastset_members(&b) == 0);
  CHECK(fastset_is_empty(&b));
  fastset_check(&b);

  fastset_free(&empty);
  fastset_free(&c);
  fastset_free(&b);
  fastset_free(&a);
#endif
}
