/* Test program for tsearch et al.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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

#ifndef _GNU_SOURCE
# define _GNU_SOURCE	1
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <search.h>
#include <tst-stack-align.h>
#include <support/check.h>

#define SEED 0
#define BALANCED 1
#define PASSES 100

#if BALANCED
#include <math.h>
#define SIZE 1000
#else
#define SIZE 100
#endif

enum order
{
  ascending,
  descending,
  randomorder
};

enum action
{
  build,
  build_and_del,
  delete,
  find
};

/* Set to 1 if a test is flunked.  */
static int error = 0;

/* The keys we add to the tree.  */
static int x[SIZE];

/* Pointers into the key array, possibly permutated, to define an order
   for insertion/removal.  */
static int y[SIZE];

/* Flags set for each element visited during a tree walk.  */
static int z[SIZE];

/* Depths for all the elements, to check that the depth is constant for
   all three visits.  */
static int depths[SIZE];

/* Maximum depth during a tree walk.  */
static int max_depth;

static int stack_align_check[2];

/* Used to compare walk traces between the two implementations.  */
struct walk_trace_element
{
  const void *key;
  VISIT which;
  int depth;
};
#define DYNARRAY_STRUCT walk_trace_list
#define DYNARRAY_ELEMENT struct walk_trace_element
#define DYNARRAY_PREFIX walk_trace_
#define DYNARRAY_INITIAL_SIZE 0
#include <malloc/dynarray-skeleton.c>
static struct walk_trace_list walk_trace;

/* Compare two keys.  */
static int
cmp_fn (const void *a, const void *b)
{
  if (!stack_align_check[0])
    stack_align_check[0] = TEST_STACK_ALIGN () ? -1 : 1;
  return *(const int *) a - *(const int *) b;
}

/* Permute an array of integers.  */
static void
memfry (int *string)
{
  int i;

  for (i = 0; i < SIZE; ++i)
    {
      int32_t j;
      int c;

      j = random () % SIZE;

      c = string[i];
      string[i] = string[j];
      string[j] = c;
    }
}

struct twalk_with_twalk_r_closure
{
  void (*action) (const void *, VISIT, int);
  int depth;
};

static void
twalk_with_twalk_r_action (const void *nodep, VISIT which, void *closure0)
{
  struct twalk_with_twalk_r_closure *closure = closure0;

  switch (which)
    {
    case leaf:
      closure->action (nodep, which, closure->depth);
      break;
    case preorder:
      closure->action (nodep, which, closure->depth);
      ++closure->depth;
      break;
    case postorder:
      /* The preorder action incremented the depth.  */
      closure->action (nodep, which, closure->depth - 1);
      break;
    case endorder:
      --closure->depth;
      closure->action (nodep, which, closure->depth);
      break;
    }
}

static void
twalk_with_twalk_r (const void *root,
		    void (*action) (const void *, VISIT, int))
{
  struct twalk_with_twalk_r_closure closure = { action, 0 };
  twalk_r (root, twalk_with_twalk_r_action, &closure);
  TEST_COMPARE (closure.depth, 0);
}

static void
walk_action (const void *nodep, const VISIT which, const int depth)
{
  int key = **(int **) nodep;

  walk_trace_add (&walk_trace,
		  (struct walk_trace_element) { nodep, which, depth });

  if (!stack_align_check[1])
    stack_align_check[1] = TEST_STACK_ALIGN () ? -1 : 1;

  if (depth > max_depth)
    max_depth = depth;
  if (which == leaf || which == preorder)
    {
      ++z[key];
      depths[key] = depth;
    }
  else
    {
      if (depths[key] != depth)
	{
	  fputs ("Depth for one element is not constant during tree walk.\n",
		 stdout);
	}
    }
}

static void
walk_tree_with (void *root, int expected_count,
		void (*walk) (const void *,
			      void (*) (const void *, VISIT, int)))
{
  int i;

  memset (z, 0, sizeof z);
  max_depth = 0;

  walk (root, walk_action);
  for (i = 0; i < expected_count; ++i)
    if (z[i] != 1)
      {
	fputs ("Node was not visited.\n", stdout);
	error = 1;
      }

#if BALANCED
  if (max_depth > log (expected_count) * 2 + 2)
#else
  if (max_depth > expected_count)
#endif
    {
      fputs ("Depth too large during tree walk.\n", stdout);
      error = 1;
    }
}

static void
walk_tree (void *root, int expected_count)
{
  walk_trace_clear (&walk_trace);
  walk_tree_with (root, expected_count, twalk);
  TEST_VERIFY (!walk_trace_has_failed (&walk_trace));
  size_t first_list_size;
  struct walk_trace_element *first_list
    = walk_trace_finalize (&walk_trace, &first_list_size);
  TEST_VERIFY_EXIT (first_list != NULL);

  walk_tree_with (root, expected_count, twalk_with_twalk_r);
  TEST_VERIFY (!walk_trace_has_failed (&walk_trace));

  /* Compare the two traces.  */
  TEST_COMPARE (first_list_size, walk_trace_size (&walk_trace));
  for (size_t i = 0; i < first_list_size && i < walk_trace_size (&walk_trace);
       ++i)
    {
      TEST_VERIFY (first_list[i].key == walk_trace_at (&walk_trace, i)->key);
      TEST_COMPARE (first_list[i].which, walk_trace_at (&walk_trace, i)->which);
      TEST_COMPARE (first_list[i].depth, walk_trace_at (&walk_trace, i)->depth);
    }

  walk_trace_free (&walk_trace);
}

/* Perform an operation on a tree.  */
static void
mangle_tree (enum order how, enum action what, void **root, int lag)
{
  int i;

  if (how == randomorder)
    {
      for (i = 0; i < SIZE; ++i)
	y[i] = i;
      memfry (y);
    }

  for (i = 0; i < SIZE + lag; ++i)
    {
      void *elem;
      int j, k;

      switch (how)
	{
	case randomorder:
	  if (i >= lag)
	    k = y[i - lag];
	  else
	    /* Ensure that the array index is within bounds.  */
	    k = y[(SIZE - i - 1 + lag) % SIZE];
	  j = y[i % SIZE];
	  break;

	case ascending:
	  k = i - lag;
	  j = i;
	  break;

	case descending:
	  k = SIZE - i - 1 + lag;
	  j = SIZE - i - 1;
	  break;

	default:
	  /* This never should happen, but gcc isn't smart enough to
	     recognize it.  */
	  abort ();
	}

      switch (what)
	{
	case build_and_del:
	case build:
	  if (i < SIZE)
	    {
	      if (tfind (x + j, (void *const *) root, cmp_fn) != NULL)
		{
		  fputs ("Found element which is not in tree yet.\n", stdout);
		  error = 1;
		}
	      elem = tsearch (x + j, root, cmp_fn);
	      if (elem == 0
		  || tfind (x + j, (void *const *) root, cmp_fn) == NULL)
		{
		  fputs ("Couldn't find element after it was added.\n",
			 stdout);
		  error = 1;
		}
	    }

	  if (what == build || i < lag)
	    break;

	  j = k;
	  /* fall through */

	case delete:
	  elem = tfind (x + j, (void *const *) root, cmp_fn);
	  if (elem == NULL || tdelete (x + j, root, cmp_fn) == NULL)
	    {
	      fputs ("Error deleting element.\n", stdout);
	      error = 1;
	    }
	  break;

	case find:
	  if (tfind (x + j, (void *const *) root, cmp_fn) == NULL)
	    {
	      fputs ("Couldn't find element after it was added.\n", stdout);
	      error = 1;
	    }
	  break;

	}
    }
}


static int
do_test (void)
{
  int total_error = 0;
  static char state[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
  void *root = NULL;
  int i, j;

  initstate (SEED, state, 8);

  for (i = 0; i < SIZE; ++i)
    x[i] = i;

  /* Do this loop several times to get different permutations for the
     random case.  */
  fputs ("Series I\n", stdout);
  for (i = 0; i < PASSES; ++i)
    {
      fprintf (stdout, "Pass %d... ", i + 1);
      fflush (stdout);
      error = 0;

      mangle_tree (ascending, build, &root, 0);
      mangle_tree (ascending, find, &root, 0);
      mangle_tree (descending, find, &root, 0);
      mangle_tree (randomorder, find, &root, 0);
      walk_tree (root, SIZE);
      mangle_tree (ascending, delete, &root, 0);

      mangle_tree (ascending, build, &root, 0);
      walk_tree (root, SIZE);
      mangle_tree (descending, delete, &root, 0);

      mangle_tree (ascending, build, &root, 0);
      walk_tree (root, SIZE);
      mangle_tree (randomorder, delete, &root, 0);

      mangle_tree (descending, build, &root, 0);
      mangle_tree (ascending, find, &root, 0);
      mangle_tree (descending, find, &root, 0);
      mangle_tree (randomorder, find, &root, 0);
      walk_tree (root, SIZE);
      mangle_tree (descending, delete, &root, 0);

      mangle_tree (descending, build, &root, 0);
      walk_tree (root, SIZE);
      mangle_tree (descending, delete, &root, 0);

      mangle_tree (descending, build, &root, 0);
      walk_tree (root, SIZE);
      mangle_tree (randomorder, delete, &root, 0);

      mangle_tree (randomorder, build, &root, 0);
      mangle_tree (ascending, find, &root, 0);
      mangle_tree (descending, find, &root, 0);
      mangle_tree (randomorder, find, &root, 0);
      walk_tree (root, SIZE);
      mangle_tree (randomorder, delete, &root, 0);

      for (j = 1; j < SIZE; j *= 2)
	{
	  mangle_tree (randomorder, build_and_del, &root, j);
	}

      fputs (error ? " failed!\n" : " ok.\n", stdout);
      total_error |= error;
    }

  fputs ("Series II\n", stdout);
  for (i = 1; i < SIZE; i *= 2)
    {
      fprintf (stdout, "For size %d... ", i);
      fflush (stdout);
      error = 0;

      mangle_tree (ascending, build_and_del, &root, i);
      mangle_tree (descending, build_and_del, &root, i);
      mangle_tree (ascending, build_and_del, &root, i);
      mangle_tree (descending, build_and_del, &root, i);
      mangle_tree (ascending, build_and_del, &root, i);
      mangle_tree (descending, build_and_del, &root, i);
      mangle_tree (ascending, build_and_del, &root, i);
      mangle_tree (descending, build_and_del, &root, i);

      fputs (error ? " failed!\n" : " ok.\n", stdout);
      total_error |= error;
    }

  for (i = 0; i < 2; ++i)
    if (stack_align_check[i] == 0)
      {
        printf ("stack alignment check %d not run\n", i);
        total_error |= 1;
      }
    else if (stack_align_check[i] != 1)
      {
        printf ("stack insufficiently aligned in check %d\n", i);
        total_error |= 1;
      }

  return total_error;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
