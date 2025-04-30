/* Test the allocate_once function.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <allocate_once.h>
#include <mcheck.h>
#include <string.h>
#include <support/check.h>
#include <support/support.h>

/* Allocate a new string.  */
static void *
allocate_string (void *closure)
{
  return xstrdup (closure);
}

/* Allocation and deallocation functions which are not expected to be
   called.  */

static void *
allocate_not_called (void *closure)
{
  FAIL_EXIT1 ("allocation function called unexpectedly (%p)", closure);
}

static void
deallocate_not_called (void *closure, void *ptr)
{
  FAIL_EXIT1 ("deallocate function called unexpectedly (%p, %p)",
              closure, ptr);
}

/* Counter for various function calls.  */
static int function_called;

/* An allocation function which returns NULL and records that it has
   been called.  */
static void *
allocate_return_null (void *closure)
{
  /* The function should only be called once.  */
  TEST_COMPARE (function_called, 0);
  ++function_called;
  return NULL;
}


/* The following is used to check the retry logic, by causing a fake
   race condition.  */
static void *fake_race_place;
static char fake_race_region[3]; /* To obtain unique addresses.  */

static void *
fake_race_allocate (void *closure)
{
  TEST_VERIFY (closure == &fake_race_region[0]);
  TEST_COMPARE (function_called, 0);
  ++function_called;
  /* Fake allocation by another thread.  */
  fake_race_place = &fake_race_region[1];
  return &fake_race_region[2];
}

static void
fake_race_deallocate (void *closure, void *ptr)
{
  /* Check that the pointer returned from fake_race_allocate is
     deallocated (and not the one stored in fake_race_place).  */
  TEST_VERIFY (ptr == &fake_race_region[2]);

  TEST_VERIFY (fake_race_place == &fake_race_region[1]);
  TEST_VERIFY (closure == &fake_race_region[0]);
  TEST_COMPARE (function_called, 1);
  ++function_called;
}

/* Similar to fake_race_allocate, but expects to be paired with free
   as the deallocation function.  */
static void *
fake_race_allocate_for_free (void *closure)
{
  TEST_VERIFY (closure == &fake_race_region[0]);
  TEST_COMPARE (function_called, 0);
  ++function_called;
  /* Fake allocation by another thread.  */
  fake_race_place = &fake_race_region[1];
  return xstrdup ("to be freed");
}

static int
do_test (void)
{
  mtrace ();

  /* Simple allocation.  */
  void *place1 = NULL;
  char *string1 = allocate_once (&place1, allocate_string,
                                   deallocate_not_called,
                                   (char *) "test string 1");
  TEST_VERIFY_EXIT (string1 != NULL);
  TEST_VERIFY (strcmp ("test string 1", string1) == 0);
  /* Second call returns the first pointer, without calling any
     callbacks.  */
  TEST_VERIFY (string1
               == allocate_once (&place1, allocate_not_called,
                                 deallocate_not_called,
                                 (char *) "test string 1a"));

  /* Different place should result in another call.  */
  void *place2 = NULL;
  char *string2 = allocate_once (&place2, allocate_string,
                                 deallocate_not_called,
                                 (char *) "test string 2");
  TEST_VERIFY_EXIT (string2 != NULL);
  TEST_VERIFY (strcmp ("test string 2", string2) == 0);
  TEST_VERIFY (string1 != string2);

  /* Check error reporting (NULL return value from the allocation
     function).  */
  void *place3 = NULL;
  char *string3 = allocate_once (&place3, allocate_return_null,
                                 deallocate_not_called, NULL);
  TEST_VERIFY (string3 == NULL);
  TEST_COMPARE (function_called, 1);

  /* Check that the deallocation function is called if the race is
     lost.  */
  function_called = 0;
  TEST_VERIFY (allocate_once (&fake_race_place,
                              fake_race_allocate,
                              fake_race_deallocate,
                              &fake_race_region[0])
               == &fake_race_region[1]);
  TEST_COMPARE (function_called, 2);
  function_called = 3;
  TEST_VERIFY (allocate_once (&fake_race_place,
                              fake_race_allocate,
                              fake_race_deallocate,
                              &fake_race_region[0])
               == &fake_race_region[1]);
  TEST_COMPARE (function_called, 3);

  /* Similar, but this time rely on that free is called.  */
  function_called = 0;
  fake_race_place = NULL;
  TEST_VERIFY (allocate_once (&fake_race_place,
                                fake_race_allocate_for_free,
                                NULL,
                                &fake_race_region[0])
               == &fake_race_region[1]);
  TEST_COMPARE (function_called, 1);
  function_called = 3;
  TEST_VERIFY (allocate_once (&fake_race_place,
                              fake_race_allocate_for_free,
                              NULL,
                              &fake_race_region[0])
               == &fake_race_region[1]);
  TEST_COMPARE (function_called, 3);

  free (place2);
  free (place1);

  return 0;
}

#include <support/test-driver.c>
