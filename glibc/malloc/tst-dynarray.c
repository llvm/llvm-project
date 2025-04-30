/* Test for dynamic arrays.
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

#include "tst-dynarray-shared.h"

#include <errno.h>
#include <stdint.h>

#define DYNARRAY_STRUCT dynarray_long
#define DYNARRAY_ELEMENT long
#define DYNARRAY_PREFIX dynarray_long_
#define DYNARRAY_ELEMENT_INIT(e) (*(e) = 17)
#include <malloc/dynarray-skeleton.c>

struct long_array
{
  long *array;
  size_t length;
};

#define DYNARRAY_STRUCT dynarray_long_noscratch
#define DYNARRAY_ELEMENT long
#define DYNARRAY_PREFIX dynarray_long_noscratch_
#define DYNARRAY_ELEMENT_INIT(e) (*(e) = 23)
#define DYNARRAY_FINAL_TYPE struct long_array
#define DYNARRAY_INITIAL_SIZE 0
#include <malloc/dynarray-skeleton.c>

#define DYNARRAY_STRUCT zstr
#define DYNARRAY_ELEMENT char
#define DYNARRAY_PREFIX zstr_
#define DYNARRAY_INITIAL_SIZE 128
#include <malloc/dynarray-skeleton.c>

#include <malloc.h>
#include <mcheck.h>
#include <stdint.h>
#include <support/check.h>
#include <support/support.h>

enum { max_count = 20 };

/* Test dynamic arrays with int elements (no automatic deallocation
   for elements).  */
static void
test_int (void)
{
  /* Empty array.  */
  {
    struct dynarray_int dyn;
    dynarray_int_init (&dyn);
    CHECK_EMPTY (int, &dyn);
  }

  /* Empty array with finalization.  */
  {
    struct dynarray_int dyn;
    dynarray_int_init (&dyn);
    CHECK_INIT_STATE (int, &dyn);
    struct int_array result = { (int *) (uintptr_t) -1, -1 };
    TEST_VERIFY_EXIT (dynarray_int_finalize (&dyn, &result));
    CHECK_INIT_STATE (int, &dyn);
    TEST_VERIFY_EXIT (result.array == NULL);
    TEST_VERIFY_EXIT (result.length == 0);
  }

  /* Non-empty array tests.

     do_add: Switch between emplace (false) and add (true).
     do_finalize: Perform finalize call at the end.
     do_clear: Perform clear call at the end.
     do_remove_last: Perform remove_last call after adding elements.
     count: Number of elements added to the array.  */
  for (int do_add = 0; do_add < 2; ++do_add)
    for (int do_finalize = 0; do_finalize < 2; ++do_finalize)
      for (int do_clear = 0; do_clear < 2; ++do_clear)
        for (int do_remove_last = 0; do_remove_last < 2; ++do_remove_last)
          for (unsigned int count = 0; count < max_count; ++count)
            {
              if (do_remove_last && count == 0)
                continue;
              unsigned int base = count * count;
              struct dynarray_int dyn;
              dynarray_int_init (&dyn);
              for (unsigned int i = 0; i < count; ++i)
                {
                  if (do_add)
                    dynarray_int_add (&dyn, base + i);
                  else
                    {
                      int *place = dynarray_int_emplace (&dyn);
                      TEST_VERIFY_EXIT (place != NULL);
                      *place = base + i;
                    }
                  TEST_VERIFY_EXIT (!dynarray_int_has_failed (&dyn));
                  TEST_VERIFY_EXIT (dynarray_int_size (&dyn) == i + 1);
                  TEST_VERIFY_EXIT (dynarray_int_size (&dyn)
                                    <= dyn.u.dynarray_header.allocated);
                }
              TEST_VERIFY_EXIT (dynarray_int_size (&dyn) == count);
              TEST_VERIFY_EXIT (count <= dyn.u.dynarray_header.allocated);
              if (count > 0)
                {
                  TEST_VERIFY (dynarray_int_begin (&dyn)
                               == dynarray_int_at (&dyn, 0));
                  TEST_VERIFY (dynarray_int_end (&dyn)
                               == dynarray_int_at (&dyn, count - 1) + 1);
                }
              unsigned final_count;
              bool heap_array = dyn.u.dynarray_header.array != dyn.scratch;
              if (do_remove_last)
                {
                  dynarray_int_remove_last (&dyn);
                  if (count == 0)
                    final_count = 0;
                  else
                    final_count = count - 1;
                }
              else
                final_count = count;
              if (final_count > 0)
                {
                  TEST_VERIFY (dynarray_int_begin (&dyn)
                               == dynarray_int_at (&dyn, 0));
                  TEST_VERIFY (dynarray_int_end (&dyn)
                               == dynarray_int_at (&dyn, final_count - 1) + 1);
                }
              if (do_clear)
                {
                  dynarray_int_clear (&dyn);
                  final_count = 0;
                }
              TEST_VERIFY_EXIT (!dynarray_int_has_failed (&dyn));
              TEST_VERIFY_EXIT ((dyn.u.dynarray_header.array != dyn.scratch)
                                == heap_array);
              TEST_VERIFY_EXIT (dynarray_int_size (&dyn) == final_count);
              TEST_VERIFY_EXIT (dyn.u.dynarray_header.allocated
				>= final_count);
              if (!do_clear)
                for (unsigned int i = 0; i < final_count; ++i)
                  TEST_VERIFY_EXIT (*dynarray_int_at (&dyn, i) == base + i);
              if (do_finalize)
                {
                  struct int_array result = { (int *) (uintptr_t) -1, -1 };
                  TEST_VERIFY_EXIT (dynarray_int_finalize (&dyn, &result));
                  CHECK_INIT_STATE (int, &dyn);
                  TEST_VERIFY_EXIT (result.length == final_count);
                  if (final_count == 0)
                    TEST_VERIFY_EXIT (result.array == NULL);
                  else
                    {
                      TEST_VERIFY_EXIT (result.array != NULL);
                      TEST_VERIFY_EXIT (result.array != (int *) (uintptr_t) -1);
                      TEST_VERIFY_EXIT
                        (malloc_usable_size (result.array)
                         >= final_count * sizeof (result.array[0]));
                      for (unsigned int i = 0; i < final_count; ++i)
                        TEST_VERIFY_EXIT (result.array[i] == base + i);
                      free (result.array);
                    }
                }
              else /* !do_finalize */
                {
                  dynarray_int_free (&dyn);
                  CHECK_INIT_STATE (int, &dyn);
                }
            }
}

/* Test dynamic arrays with char * elements (with automatic
   deallocation of the pointed-to strings).  */
static void
test_str (void)
{
  /* Empty array.  */
  {
    struct dynarray_str dyn;
    dynarray_str_init (&dyn);
    CHECK_EMPTY (str, &dyn);
  }

  /* Empty array with finalization.  */
  {
    struct dynarray_str dyn;
    dynarray_str_init (&dyn);
    TEST_VERIFY_EXIT (!dynarray_str_has_failed (&dyn));
    struct str_array result = { (char **) (uintptr_t) -1, -1 };
    TEST_VERIFY_EXIT (dynarray_str_finalize (&dyn, &result));
    CHECK_INIT_STATE (str, &dyn);
    TEST_VERIFY_EXIT (result.array == NULL);
    TEST_VERIFY_EXIT (result.length == 0);
  }

  /* Non-empty array tests.

     do_add: Switch between emplace (false) and add (true).
     do_finalize: Perform finalize call at the end.
     do_clear: Perform clear call at the end.
     do_remove_last: Perform remove_last call after adding elements.
     count: Number of elements added to the array.  */
  for (int do_add = 0; do_add < 2; ++do_add)
    for (int do_finalize = 0; do_finalize < 2; ++do_finalize)
      for (int do_clear = 0; do_clear < 2; ++do_clear)
        for (int do_remove_last = 0; do_remove_last < 2; ++do_remove_last)
          for (unsigned int count = 0; count < max_count; ++count)
            {
              if (do_remove_last && count == 0)
                continue;
              unsigned int base = count * count;
              struct dynarray_str dyn;
              dynarray_str_init (&dyn);
              for (unsigned int i = 0; i < count; ++i)
                {
                  char *item = xasprintf ("%d", base + i);
                  if (do_add)
                    dynarray_str_add (&dyn, item);
                  else
                    {
                      char **place = dynarray_str_emplace (&dyn);
                      TEST_VERIFY_EXIT (place != NULL);
                      TEST_VERIFY_EXIT (*place == NULL);
                      *place = item;
                    }
                  TEST_VERIFY_EXIT (!dynarray_str_has_failed (&dyn));
                  TEST_VERIFY_EXIT (dynarray_str_size (&dyn) == i + 1);
                  TEST_VERIFY_EXIT (dynarray_str_size (&dyn)
                                    <= dyn.u.dynarray_header.allocated);
                }
              TEST_VERIFY_EXIT (dynarray_str_size (&dyn) == count);
              TEST_VERIFY_EXIT (count <= dyn.u.dynarray_header.allocated);
              if (count > 0)
                {
                  TEST_VERIFY (dynarray_str_begin (&dyn)
                               == dynarray_str_at (&dyn, 0));
                  TEST_VERIFY (dynarray_str_end (&dyn)
                               == dynarray_str_at (&dyn, count - 1) + 1);
                }
              unsigned final_count;
              bool heap_array = dyn.u.dynarray_header.array != dyn.scratch;
              if (do_remove_last)
                {
                  dynarray_str_remove_last (&dyn);
                  if (count == 0)
                    final_count = 0;
                  else
                    final_count = count - 1;
                }
              else
                final_count = count;
              if (final_count > 0)
                {
                  TEST_VERIFY (dynarray_str_begin (&dyn)
                               == dynarray_str_at (&dyn, 0));
                  TEST_VERIFY (dynarray_str_end (&dyn)
                               == dynarray_str_at (&dyn, final_count - 1) + 1);
                }
              if (do_clear)
                {
                  dynarray_str_clear (&dyn);
                  final_count = 0;
                }
              TEST_VERIFY_EXIT (!dynarray_str_has_failed (&dyn));
              TEST_VERIFY_EXIT ((dyn.u.dynarray_header.array != dyn.scratch)
                                == heap_array);
              TEST_VERIFY_EXIT (dynarray_str_size (&dyn) == final_count);
              TEST_VERIFY_EXIT (dyn.u.dynarray_header.allocated
				>= final_count);
              if (!do_clear)
                for (unsigned int i = 0; i < count - do_remove_last; ++i)
                  {
                    char *expected = xasprintf ("%d", base + i);
                    const char *actual = *dynarray_str_at (&dyn, i);
                    TEST_VERIFY_EXIT (strcmp (actual, expected) == 0);
                    free (expected);
                  }
              if (do_finalize)
                {
                  struct str_array result = { (char **) (uintptr_t) -1, -1 };
                  TEST_VERIFY_EXIT (dynarray_str_finalize (&dyn, &result));
                  CHECK_INIT_STATE (str, &dyn);
                  TEST_VERIFY_EXIT (result.length == final_count);
                  if (final_count == 0)
                    TEST_VERIFY_EXIT (result.array == NULL);
                  else
                    {
                      TEST_VERIFY_EXIT (result.array != NULL);
                      TEST_VERIFY_EXIT (result.array
                                        != (char **) (uintptr_t) -1);
                      TEST_VERIFY_EXIT (result.length
                                        == count - do_remove_last);
                      TEST_VERIFY_EXIT
                        (malloc_usable_size (result.array)
                         >= final_count * sizeof (result.array[0]));
                      for (unsigned int i = 0; i < count - do_remove_last; ++i)
                        {
                          char *expected = xasprintf ("%d", base + i);
                          char *actual = result.array[i];
                          TEST_VERIFY_EXIT (strcmp (actual, expected) == 0);
                          free (expected);
                          free (actual);
                        }
                      free (result.array);
                    }
                }
              else /* !do_finalize */
                {
                  dynarray_str_free (&dyn);
                  CHECK_INIT_STATE (str, &dyn);
                }
            }

  /* Test resizing.  */
  {
    enum { count = 2131 };
    struct dynarray_str dyn;
    dynarray_str_init (&dyn);

    /* From length 0 to length 1.  */
    TEST_VERIFY (dynarray_str_resize (&dyn, 1));
    TEST_VERIFY (dynarray_str_size (&dyn) == 1);
    TEST_VERIFY (*dynarray_str_at (&dyn, 0) == NULL);
    *dynarray_str_at (&dyn, 0) = xstrdup ("allocated");
    dynarray_str_free (&dyn);

    /* From length 0 to length 1 and 2.  */
    TEST_VERIFY (dynarray_str_resize (&dyn, 1));
    TEST_VERIFY (dynarray_str_size (&dyn) == 1);
    TEST_VERIFY (*dynarray_str_at (&dyn, 0) == NULL);
    *dynarray_str_at (&dyn, 0) = xstrdup ("allocated0");
    TEST_VERIFY (dynarray_str_resize (&dyn, 2));
    TEST_VERIFY (dynarray_str_size (&dyn) == 2);
    TEST_VERIFY (strcmp (*dynarray_str_at (&dyn, 0), "allocated0") == 0);
    TEST_VERIFY (*dynarray_str_at (&dyn, 1) == NULL);
    *dynarray_str_at (&dyn, 1) = xstrdup ("allocated1");
    TEST_VERIFY (dynarray_str_resize (&dyn, count));
    TEST_VERIFY (dynarray_str_size (&dyn) == count);
    TEST_VERIFY (strcmp (*dynarray_str_at (&dyn, 0), "allocated0") == 0);
    TEST_VERIFY (strcmp (*dynarray_str_at (&dyn, 1), "allocated1") == 0);
    for (int i = 2; i < count; ++i)
      TEST_VERIFY (*dynarray_str_at (&dyn, i) == NULL);
    *dynarray_str_at (&dyn, count - 1) = xstrdup ("allocated2");
    TEST_VERIFY (dynarray_str_resize (&dyn, 3));
    TEST_VERIFY (strcmp (*dynarray_str_at (&dyn, 0), "allocated0") == 0);
    TEST_VERIFY (strcmp (*dynarray_str_at (&dyn, 1), "allocated1") == 0);
    TEST_VERIFY (*dynarray_str_at (&dyn, 2) == NULL);
    dynarray_str_free (&dyn);
  }
}

/* Verify that DYNARRAY_ELEMENT_INIT has an effect.  */
static void
test_long_init (void)
{
  enum { count = 2131 };
  {
    struct dynarray_long dyn;
    dynarray_long_init (&dyn);
    for (int i = 0; i < count; ++i)
      {
        long *place = dynarray_long_emplace (&dyn);
        TEST_VERIFY_EXIT (place != NULL);
        TEST_VERIFY (*place == 17);
      }
    TEST_VERIFY (dynarray_long_size (&dyn) == count);
    for (int i = 0; i < count; ++i)
      TEST_VERIFY (*dynarray_long_at (&dyn, i) == 17);
    dynarray_long_free (&dyn);

    TEST_VERIFY (dynarray_long_resize (&dyn, 1));
    TEST_VERIFY (dynarray_long_size (&dyn) == 1);
    TEST_VERIFY (*dynarray_long_at (&dyn, 0) == 17);
    *dynarray_long_at (&dyn, 0) = 18;
    dynarray_long_free (&dyn);
    TEST_VERIFY (dynarray_long_resize (&dyn, 1));
    TEST_VERIFY (dynarray_long_size (&dyn) == 1);
    TEST_VERIFY (*dynarray_long_at (&dyn, 0) == 17);
    TEST_VERIFY (dynarray_long_resize (&dyn, 2));
    TEST_VERIFY (dynarray_long_size (&dyn) == 2);
    TEST_VERIFY (*dynarray_long_at (&dyn, 0) == 17);
    TEST_VERIFY (*dynarray_long_at (&dyn, 1) == 17);
    *dynarray_long_at (&dyn, 0) = 18;
    TEST_VERIFY (dynarray_long_resize (&dyn, count));
    TEST_VERIFY (dynarray_long_size (&dyn) == count);
    TEST_VERIFY (*dynarray_long_at (&dyn, 0) == 18);
    for (int i = 1; i < count; ++i)
      TEST_VERIFY (*dynarray_long_at (&dyn, i) == 17);
    dynarray_long_free (&dyn);
  }

  /* Similar, but without an on-stack scratch region
     (DYNARRAY_INITIAL_SIZE is 0).  */
  {
    struct dynarray_long_noscratch dyn;
    dynarray_long_noscratch_init (&dyn);
    struct long_array result;
    TEST_VERIFY_EXIT (dynarray_long_noscratch_finalize (&dyn, &result));
    TEST_VERIFY (result.array == NULL);
    TEST_VERIFY (result.length == 0);

    /* Test with one element.  */
    {
      long *place = dynarray_long_noscratch_emplace (&dyn);
      TEST_VERIFY_EXIT (place != NULL);
      TEST_VERIFY (*place == 23);
    }
    TEST_VERIFY (dynarray_long_noscratch_size (&dyn) == 1);
    TEST_VERIFY (*dynarray_long_noscratch_at (&dyn, 0) == 23);
    TEST_VERIFY_EXIT (dynarray_long_noscratch_finalize (&dyn, &result));
    TEST_VERIFY_EXIT (result.array != NULL);
    TEST_VERIFY (result.length == 1);
    TEST_VERIFY (result.array[0] == 23);
    free (result.array);

    for (int i = 0; i < count; ++i)
      {
        long *place = dynarray_long_noscratch_emplace (&dyn);
        TEST_VERIFY_EXIT (place != NULL);
        TEST_VERIFY (*place == 23);
        if (i == 0)
          *place = 29;
      }
    TEST_VERIFY (dynarray_long_noscratch_size (&dyn) == count);
    TEST_VERIFY (*dynarray_long_noscratch_at (&dyn, 0) == 29);
    for (int i = 1; i < count; ++i)
      TEST_VERIFY (*dynarray_long_noscratch_at (&dyn, i) == 23);
    TEST_VERIFY_EXIT (dynarray_long_noscratch_finalize (&dyn, &result));
    TEST_VERIFY_EXIT (result.array != NULL);
    TEST_VERIFY (result.length == count);
    TEST_VERIFY (result.array[0] == 29);
    for (int i = 1; i < count; ++i)
      TEST_VERIFY (result.array[i] == 23);
    free (result.array);

    TEST_VERIFY (dynarray_long_noscratch_resize (&dyn, 1));
    TEST_VERIFY (dynarray_long_noscratch_size (&dyn) == 1);
    TEST_VERIFY (*dynarray_long_noscratch_at (&dyn, 0) == 23);
    *dynarray_long_noscratch_at (&dyn, 0) = 24;
    dynarray_long_noscratch_free (&dyn);
    TEST_VERIFY (dynarray_long_noscratch_resize (&dyn, 1));
    TEST_VERIFY (dynarray_long_noscratch_size (&dyn) == 1);
    TEST_VERIFY (*dynarray_long_noscratch_at (&dyn, 0) == 23);
    TEST_VERIFY (dynarray_long_noscratch_resize (&dyn, 2));
    TEST_VERIFY (dynarray_long_noscratch_size (&dyn) == 2);
    TEST_VERIFY (*dynarray_long_noscratch_at (&dyn, 0) == 23);
    TEST_VERIFY (*dynarray_long_noscratch_at (&dyn, 1) == 23);
    *dynarray_long_noscratch_at (&dyn, 0) = 24;
    TEST_VERIFY (dynarray_long_noscratch_resize (&dyn, count));
    TEST_VERIFY (dynarray_long_noscratch_size (&dyn) == count);
    TEST_VERIFY (*dynarray_long_noscratch_at (&dyn, 0) == 24);
    for (int i = 1; i < count; ++i)
      TEST_VERIFY (*dynarray_long_noscratch_at (&dyn, i) == 23);
    dynarray_long_noscratch_free (&dyn);
  }
}

/* Test overflow in resize.  */
static void
test_long_overflow (void)
{
  {
    struct dynarray_long dyn;
    dynarray_long_init (&dyn);
    errno = EINVAL;
    TEST_VERIFY (!dynarray_long_resize
                 (&dyn, (SIZE_MAX / sizeof (long)) + 1));
    TEST_VERIFY (errno == ENOMEM);
    TEST_VERIFY (dynarray_long_has_failed (&dyn));
  }

  {
    struct dynarray_long_noscratch dyn;
    dynarray_long_noscratch_init (&dyn);
    errno = EINVAL;
    TEST_VERIFY (!dynarray_long_noscratch_resize
                 (&dyn, (SIZE_MAX / sizeof (long)) + 1));
    TEST_VERIFY (errno == ENOMEM);
    TEST_VERIFY (dynarray_long_noscratch_has_failed (&dyn));
  }
}

/* Test NUL-terminated string construction with the add function and
   the simple finalize function.  */
static void
test_zstr (void)
{
  /* Totally empty string (no NUL termination).  */
  {
    struct zstr s;
    zstr_init (&s);
    char *result = zstr_finalize (&s, NULL);
    TEST_VERIFY (result == NULL);
    TEST_VERIFY (zstr_size (&s) == 0);
    size_t length = 1;
    result = zstr_finalize (&s, &length);
    TEST_VERIFY (result == NULL);
    TEST_VERIFY (length == 0);
    TEST_VERIFY (zstr_size (&s) == 0);
  }

  /* Empty string.  */
  {
    struct zstr s;
    zstr_init (&s);
    zstr_add (&s, '\0');
    char *result = zstr_finalize (&s, NULL);
    TEST_VERIFY_EXIT (result != NULL);
    TEST_VERIFY (*result == '\0');
    TEST_VERIFY (zstr_size (&s) == 0);
    free (result);

    zstr_add (&s, '\0');
    size_t length = 1;
    result = zstr_finalize (&s, &length);
    TEST_VERIFY_EXIT (result != NULL);
    TEST_VERIFY (*result == '\0');
    TEST_VERIFY (length == 1);
    TEST_VERIFY (zstr_size (&s) == 0);
    free (result);
  }

  /* A few characters.  */
  {
    struct zstr s;
    zstr_init (&s);
    zstr_add (&s, 'A');
    zstr_add (&s, 'b');
    zstr_add (&s, 'c');
    zstr_add (&s, '\0');
    char *result = zstr_finalize (&s, NULL);
    TEST_VERIFY_EXIT (result != NULL);
    TEST_VERIFY (strcmp (result, "Abc") == 0);
    TEST_VERIFY (zstr_size (&s) == 0);
    free (result);

    zstr_add (&s, 'X');
    zstr_add (&s, 'y');
    zstr_add (&s, 'z');
    zstr_add (&s, '\0');
    size_t length = 1;
    result = zstr_finalize (&s, &length);
    TEST_VERIFY_EXIT (result != NULL);
    TEST_VERIFY (strcmp (result, "Xyz") == 0);
    TEST_VERIFY (length == 4);
    TEST_VERIFY (zstr_size (&s) == 0);
    free (result);
  }
}

static int
do_test (void)
{
  mtrace ();
  test_int ();
  test_str ();
  test_long_init ();
  test_long_overflow ();
  test_zstr ();
  return 0;
}

#include <support/test-driver.c>
