/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Fast sets of small nonnegative integers.
 *
 * Element membership testing and removal are constant-time operations, as is
 * emptying a set.  Adding a member to a set is constant-time unless memory
 * allocation is required.  Element traversal has time complexity linear in the
 * number elements.  Set union is O(elements of second set); set difference and
 * set intersection are O(elements of smallest set).
 *
 * Always use fastset_init() after declaring or allocating a new set, and
 * always use fastset_free() when the set is destroyed.
 *
 * The data structure is an example of a space/time trade-off being
 * made entirely in favor of time.  These sets should not be used for values
 * larger than a few thousand.  (For sets with really small maximum values,
 * consider using bitsets; for sets with larger maximum values, try a 3-2 tree.)
 *
 * The idea for this fast set data structure is attributed by Preston Briggs
 * in his register assignment thesis to himself and Linda Torczon.
 */

#ifndef SHARED_FASTSET_H_
#define SHARED_FASTSET_H_

#include "universal.h"
#include <stdlib.h>

/** \brief The representation of a fastset should be considered private. */
typedef struct fastset fastset;

/** \brief Validation (asserts if set is bad) */
void fastset_check(const fastset *set);

/** \brief Debug printing */
void fastset_dbgprintf(const fastset *set);

/* Initialization and destruction.  N.b.: fastset_free() must be called exactly
 * once on any fastset that has been initialized; it does not free() the set
 * object itself, as that may be a local variable; and the set must not be used
 * thereafter.
 */

/// \brief FIXME
void fastset_free(fastset *set); /* free set contents (but not set itself) */
/// \brief FIXME
void fastset_reinit(fastset *set);
/// \brief FIXME
void fastset_resize(fastset *set, unsigned limit_hint);


/* Basic set operations.  These modify their first arguments in place. */
/// \brief FIXME
void fastset_union(fastset *xs, const fastset *ys);
/// \brief FIXME
void fastset_difference(fastset *xs, const fastset *ys);
/// \brief FIXME
void fastset_intersection(fastset *xs, const fastset *ys);

/** Apply 'f' to the supplied pointer and each member of 'xs'.  The pointer
 * result from each invocation is passed as the pointer argument to the next.
 */
void *fastset_map(const fastset *xs, void *f(void *, int), void *pointer);

/// \brief FIXME
void fastset_unit_tests(void);

/**
 * \brief Implementation
 *
 * Sets are represented by a pair of arrays.  The first, set->member[],
 * contains the distinct elements of the set in arbitrary order in its
 * leading set->members positions, and garbage afterwards.  For each
 * element x in a set, *both* of these predicates must be true:
 * 1) 0 <= set->index[x] < set->members, and
 * 2) set->member[set->index[x]] == x
 *
 * For all values x that are *not* in a set, either x >= set->limit, or
 * set->index[x] >= set->members, or set->member[set->index[x]] != x.
 *
 * Unsigned integers are used internally to reduce needless comparisons
 * to zero.
 *
 * The predicate fastset_contains(), q.v. below, demonstrates all of the
 * conditions that hold when a value is an element of a fastset.  The
 * value must be small enough to be used as an offset into set->index[],
 * and the index value at that position must be small enough to be used
 * as an offset into the list of distinct set values stored contiguously
 * in the leading elements of the set->member[] array.
 *
 * Note that neither of the set->member[] or set->index[] array contents
 * need be initialized, and that a set can be emptied in constant time by
 * setting set->members = 0 as in fastset_vacate().  It is okay if the values
 * in set->index[] are garbage or if multiple values point to the same
 * offset in set->member[].
 *
 */
struct fastset {
  unsigned members, limit, *member, *index;
};

/// \brief FIXME
INLINE static void
fastset_init(fastset *set)
{
  set->member = set->index = NULL;
  set->members = set->limit = 0;
}

/// \brief FIXME
INLINE static void
fastset_vacate(fastset *set)
{
  set->members = 0;
}

/// \brief FIXME
INLINE static int
fastset_members(const fastset *set)
{
  return set->members;
}

/// \brief FIXME
INLINE static int
fastset_limit(const fastset *set)
{
  return set->limit;
}

/// \brief FIXME
INLINE static int
fastset_is_empty(const fastset *set)
{
  return set->members == 0;
}

/// \brief FIXME
INLINE static int
fastset_get(const fastset *set, int idx)
{
  return set->member[idx];
}

/// \brief FIXME
INLINE static int
fastset_pop(fastset *set)
{
  return set->members == 0 ? -1 : set->member[--set->members];
}

/** Predicate: is a particular value a current member of a set?
 * Any integer value "x" may be tested.
 */
INLINE static int
fastset_contains(const fastset *set, int x)
{
  unsigned idx;
  return (unsigned)x < set->limit && (idx = set->index[x]) < set->members &&
         set->member[idx] == (unsigned)x;
}

/// \brief FIXME
INLINE static int
fastset_member_index(const fastset *set, int x)
{
  unsigned idx;
  if ((unsigned)x < set->limit && (idx = set->index[x]) < set->members &&
      set->member[idx] == (unsigned)x)
    return idx;
  return -1;
}

/** Adds a value to a set, if it is distinct from current members. */
INLINE static void
fastset_add(fastset *set, unsigned x)
{
  if (x >= set->limit)
    fastset_resize(set, x + 1);
  else {
    unsigned idx = set->index[x];
    if (idx < set->members && set->member[idx] == x)
      return;
  }
  set->member[set->index[x] = set->members++] = x;
}

/** Removes a value from a set, if present.  "x" can be any integer, since
 * out-of-range values can't be in the set anyway.
 */
INLINE static void
fastset_remove(fastset *set, int x)
{
  if ((unsigned)x < set->limit) {
    unsigned idx = set->index[x];
    if (idx < set->members && set->member[idx] == (unsigned)x) {
      /* x is a member of the set; remove it and fill the vacated
       * position with the last set member
       */
      unsigned last = set->member[--set->members];
      set->member[set->index[last] = idx] = last;
    }
  }
}

#endif /* SHARED_FASTSET_H_*/
