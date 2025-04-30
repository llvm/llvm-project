/* Test allocation failures with dynamic arrays.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

/* This test is separate from tst-dynarray because it cannot run under
   valgrind.  */

#include "tst-dynarray-shared.h"

#include <mcheck.h>
#include <stdio.h>
#include <support/check.h>
#include <support/support.h>
#include <support/xunistd.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

/* Data structure to fill up the heap.  */
struct heap_filler
{
  struct heap_filler *next;
};

/* Allocate objects until the heap is full.  */
static struct heap_filler *
fill_heap (void)
{
  size_t pad = 4096;
  struct heap_filler *head = NULL;
  while (true)
    {
      struct heap_filler *new_head = malloc (sizeof (*new_head) + pad);
      if (new_head == NULL)
        {
          if (pad > 0)
            {
              /* Try again with smaller allocations.  */
              pad = 0;
              continue;
            }
          else
            break;
        }
      new_head->next = head;
      head = new_head;
    }
  return head;
}

/* Free the heap-filling allocations, so that we can continue testing
   and detect memory leaks elsewhere.  */
static void
free_fill_heap (struct heap_filler *head)
{
  while (head != NULL)
    {
      struct heap_filler *next = head->next;
      free (head);
      head = next;
    }
}

/* Check allocation failures for int arrays (without an element free
   function).  */
static void
test_int_fail (void)
{
  /* Exercise failure in add/emplace.

     do_add: Use emplace (false) or add (true) to add elements.
     do_finalize: Perform finalization at the end (instead of free).  */
  for (int do_add = 0; do_add < 2; ++do_add)
    for (int do_finalize = 0; do_finalize < 2; ++do_finalize)
      {
        struct dynarray_int dyn;
        dynarray_int_init (&dyn);
        size_t count = 0;
        while (true)
          {
            if (do_add)
              {
                dynarray_int_add (&dyn, 0);
                if (dynarray_int_has_failed (&dyn))
                  break;
              }
            else
              {
                int *place = dynarray_int_emplace (&dyn);
                if (place == NULL)
                  break;
                TEST_VERIFY_EXIT (!dynarray_int_has_failed (&dyn));
                *place = 0;
              }
            ++count;
          }
        printf ("info: %s: failure after %zu elements\n", __func__, count);
        TEST_VERIFY_EXIT (dynarray_int_has_failed (&dyn));
        if (do_finalize)
          {
            struct int_array result = { (int *) (uintptr_t) -1, -1 };
            TEST_VERIFY_EXIT (!dynarray_int_finalize (&dyn, &result));
            TEST_VERIFY_EXIT (result.array == (int *) (uintptr_t) -1);
            TEST_VERIFY_EXIT (result.length == (size_t) -1);
          }
        else
          dynarray_int_free (&dyn);
        CHECK_INIT_STATE (int, &dyn);
      }

  /* Exercise failure in finalize.  */
  for (int do_add = 0; do_add < 2; ++do_add)
    {
      struct dynarray_int dyn;
      dynarray_int_init (&dyn);
      for (unsigned int i = 0; i < 10000; ++i)
        {
          if (do_add)
            {
              dynarray_int_add (&dyn, i);
              TEST_VERIFY_EXIT (!dynarray_int_has_failed (&dyn));
            }
          else
            {
              int *place = dynarray_int_emplace (&dyn);
              TEST_VERIFY_EXIT (place != NULL);
              *place = i;
            }
        }
      TEST_VERIFY_EXIT (!dynarray_int_has_failed (&dyn));
      struct heap_filler *heap_filler = fill_heap ();
      struct int_array result = { (int *) (uintptr_t) -1, -1 };
      TEST_VERIFY_EXIT (!dynarray_int_finalize (&dyn, &result));
      TEST_VERIFY_EXIT (result.array == (int *) (uintptr_t) -1);
      TEST_VERIFY_EXIT (result.length == (size_t) -1);
      CHECK_INIT_STATE (int, &dyn);
      free_fill_heap (heap_filler);
    }

  /* Exercise failure in resize.  */
  {
    struct dynarray_int dyn;
    dynarray_int_init (&dyn);
    struct heap_filler *heap_filler = fill_heap ();
    TEST_VERIFY (!dynarray_int_resize (&dyn, 1000));
    TEST_VERIFY (dynarray_int_has_failed (&dyn));
    free_fill_heap (heap_filler);

    dynarray_int_init (&dyn);
    TEST_VERIFY (dynarray_int_resize (&dyn, 1));
    heap_filler = fill_heap ();
    TEST_VERIFY (!dynarray_int_resize (&dyn, 1000));
    TEST_VERIFY (dynarray_int_has_failed (&dyn));
    free_fill_heap (heap_filler);

    dynarray_int_init (&dyn);
    TEST_VERIFY (dynarray_int_resize (&dyn, 1000));
    heap_filler = fill_heap ();
    TEST_VERIFY (!dynarray_int_resize (&dyn, 2000));
    TEST_VERIFY (dynarray_int_has_failed (&dyn));
    free_fill_heap (heap_filler);
  }
}

/* Check allocation failures for char * arrays (which automatically
   free the pointed-to strings).  */
static void
test_str_fail (void)
{
  /* Exercise failure in add/emplace.

     do_add: Use emplace (false) or add (true) to add elements.
     do_finalize: Perform finalization at the end (instead of free).  */
  for (int do_add = 0; do_add < 2; ++do_add)
    for (int do_finalize = 0; do_finalize < 2; ++do_finalize)
      {
        struct dynarray_str dyn;
        dynarray_str_init (&dyn);
        size_t count = 0;
        while (true)
          {
            char **place;
            if (do_add)
              {
                dynarray_str_add (&dyn, NULL);
                if (dynarray_str_has_failed (&dyn))
                  break;
                else
                  place = dynarray_str_at (&dyn, dynarray_str_size (&dyn) - 1);
              }
            else
              {
                place = dynarray_str_emplace (&dyn);
                if (place == NULL)
                  break;
              }
            TEST_VERIFY_EXIT (!dynarray_str_has_failed (&dyn));
            TEST_VERIFY_EXIT (*place == NULL);
            *place = strdup ("placeholder");
            if (*place == NULL)
              {
                /* Second loop to wait for failure of
                   dynarray_str_emplace.  */
                while (true)
                  {
                    if (do_add)
                      {
                        dynarray_str_add (&dyn, NULL);
                        if (dynarray_str_has_failed (&dyn))
                          break;
                      }
                    else
                      {
                        char **place = dynarray_str_emplace (&dyn);
                        if (place == NULL)
                          break;
                        TEST_VERIFY_EXIT (!dynarray_str_has_failed (&dyn));
                        *place = NULL;
                      }
                    ++count;
                  }
                break;
              }
            ++count;
          }
        printf ("info: %s: failure after %zu elements\n", __func__, count);
        TEST_VERIFY_EXIT (dynarray_str_has_failed (&dyn));
        if (do_finalize)
          {
            struct str_array result = { (char **) (uintptr_t) -1, -1 };
            TEST_VERIFY_EXIT (!dynarray_str_finalize (&dyn, &result));
            TEST_VERIFY_EXIT (result.array == (char **) (uintptr_t) -1);
            TEST_VERIFY_EXIT (result.length == (size_t) -1);
          }
        else
          dynarray_str_free (&dyn);
        TEST_VERIFY_EXIT (!dynarray_str_has_failed (&dyn));
        TEST_VERIFY_EXIT (dyn.u.dynarray_header.array == dyn.scratch);
        TEST_VERIFY_EXIT (dynarray_str_size (&dyn) == 0);
        TEST_VERIFY_EXIT (dyn.u.dynarray_header.allocated > 0);
      }

  /* Exercise failure in finalize.  */
  for (int do_add = 0; do_add < 2; ++do_add)
    {
      struct dynarray_str dyn;
      dynarray_str_init (&dyn);
      for (unsigned int i = 0; i < 1000; ++i)
        {
          if (do_add)
            dynarray_str_add (&dyn, xstrdup ("placeholder"));
          else
            {
              char **place = dynarray_str_emplace (&dyn);
              TEST_VERIFY_EXIT (place != NULL);
              TEST_VERIFY_EXIT (*place == NULL);
              *place = xstrdup ("placeholder");
            }
        }
      TEST_VERIFY_EXIT (!dynarray_str_has_failed (&dyn));
      struct heap_filler *heap_filler = fill_heap ();
      struct str_array result = { (char **) (uintptr_t) -1, -1 };
      TEST_VERIFY_EXIT (!dynarray_str_finalize (&dyn, &result));
      TEST_VERIFY_EXIT (result.array == (char **) (uintptr_t) -1);
      TEST_VERIFY_EXIT (result.length == (size_t) -1);
      TEST_VERIFY_EXIT (!dynarray_str_has_failed (&dyn));
      TEST_VERIFY_EXIT (dyn.u.dynarray_header.array == dyn.scratch);
      TEST_VERIFY_EXIT (dynarray_str_size (&dyn) == 0);
      TEST_VERIFY_EXIT (dyn.u.dynarray_header.allocated > 0);
      free_fill_heap (heap_filler);
    }

  /* Exercise failure in resize.  */
  {
    struct dynarray_str dyn;
    dynarray_str_init (&dyn);
    struct heap_filler *heap_filler = fill_heap ();
    TEST_VERIFY (!dynarray_str_resize (&dyn, 1000));
    TEST_VERIFY (dynarray_str_has_failed (&dyn));
    free_fill_heap (heap_filler);

    dynarray_str_init (&dyn);
    TEST_VERIFY (dynarray_str_resize (&dyn, 1));
    *dynarray_str_at (&dyn, 0) = xstrdup ("allocated");
    heap_filler = fill_heap ();
    TEST_VERIFY (!dynarray_str_resize (&dyn, 1000));
    TEST_VERIFY (dynarray_str_has_failed (&dyn));
    free_fill_heap (heap_filler);

    dynarray_str_init (&dyn);
    TEST_VERIFY (dynarray_str_resize (&dyn, 1000));
    *dynarray_str_at (&dyn, 0) = xstrdup ("allocated");
    heap_filler = fill_heap ();
    TEST_VERIFY (!dynarray_str_resize (&dyn, 2000));
    TEST_VERIFY (dynarray_str_has_failed (&dyn));
    free_fill_heap (heap_filler);
  }
}

/* Test if mmap can allocate a page.  This is necessary because
   setrlimit does not fail even if it reduces the RLIMIT_AS limit
   below what is currently needed by the process.  */
static bool
mmap_works (void)
{
  void *ptr =  mmap (NULL, 1, PROT_READ | PROT_WRITE,
                     MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  if (ptr == MAP_FAILED)
    return false;
  xmunmap (ptr, 1);
  return true;
}

/* Set the RLIMIT_AS limit to the value in *LIMIT.  */
static void
xsetrlimit_as (const struct rlimit *limit)
{
  if (setrlimit (RLIMIT_AS, limit) != 0)
    FAIL_EXIT1 ("setrlimit (RLIMIT_AS, %lu): %m",
                (unsigned long) limit->rlim_cur);
}

/* Approximately this many bytes can be allocated after
   reduce_rlimit_as has run.  */
enum { as_limit_reserve = 2 * 1024 * 1024 };

/* Limit the size of the process, so that memory allocation in
   allocate_thread will eventually fail, without impacting the entire
   system.  By default, a dynamic limit which leaves room for 2 MiB is
   activated.  The TEST_RLIMIT_AS environment variable overrides
   it.  */
static void
reduce_rlimit_as (void)
{
  struct rlimit limit;
  if (getrlimit (RLIMIT_AS, &limit) != 0)
    FAIL_EXIT1 ("getrlimit (RLIMIT_AS) failed: %m");

  /* Use the TEST_RLIMIT_AS setting if available.  */
  {
    long target = 0;
    const char *variable = "TEST_RLIMIT_AS";
    const char *target_str = getenv (variable);
    if (target_str != NULL)
      {
        target = atoi (target_str);
        if (target <= 0)
          FAIL_EXIT1 ("invalid %s value: \"%s\"", variable, target_str);
        printf ("info: setting RLIMIT_AS to %ld MiB\n", target);
        target *= 1024 * 1024;      /* Convert to megabytes.  */
        limit.rlim_cur = target;
        xsetrlimit_as (&limit);
        return;
      }
  }

  /* Otherwise, try to find the limit with a binary search.  */
  unsigned long low = 1 << 20;
  limit.rlim_cur = low;
  xsetrlimit_as (&limit);

  /* Find working upper limit.  */
  unsigned long high = 1 << 30;
  while (true)
    {
      limit.rlim_cur = high;
      xsetrlimit_as (&limit);
      if (mmap_works ())
        break;
      if (2 * high < high)
        FAIL_EXIT1 ("cannot find upper AS limit");
      high *= 2;
    }

  /* Perform binary search.  */
  while ((high - low) > 128 * 1024)
    {
      unsigned long middle = (low + high) / 2;
      limit.rlim_cur = middle;
      xsetrlimit_as (&limit);
      if (mmap_works ())
        high = middle;
      else
        low = middle;
    }

  unsigned long target = high + as_limit_reserve;
  limit.rlim_cur = target;
  xsetrlimit_as (&limit);
  printf ("info: RLIMIT_AS limit: %lu bytes\n", target);
}

static int
do_test (void)
{
  mtrace ();
  reduce_rlimit_as ();
  test_int_fail ();
  test_str_fail ();
  return 0;
}

#define TIMEOUT 90
#include <support/test-driver.c>
