/* Test recursive dlopen using malloc hooks.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <stdbool.h>
#include <stdalign.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>

#define DSO "moddummy1.so"
#define FUNC "dummy1"

#define DSO1 "moddummy2.so"
#define FUNC1 "dummy2"

/* Result of the called function.  */
int func_result;

/* Call function func_name in DSO dso_name via dlopen.  */
void
call_func (const char *dso_name, const char *func_name)
{
  int ret;
  void *dso;
  int (*func) (void);
  char *err;

  /* Open the DSO.  */
  dso = dlopen (dso_name, RTLD_NOW|RTLD_GLOBAL);
  if (dso == NULL)
    {
      err = dlerror ();
      fprintf (stderr, "%s\n", err);
      exit (1);
    }
  /* Clear any errors.  */
  dlerror ();

  /* Lookup func.  */
  func = (int (*) (void)) dlsym (dso, func_name);
  if (func == NULL)
    {
      err = dlerror ();
      if (err != NULL)
        {
	  fprintf (stderr, "%s\n", err);
	  exit (1);
        }
    }
  /* Call func.  */
  func_result = (*func) ();

  /* Close the library and look for errors too.  */
  ret = dlclose (dso);
  if (ret != 0)
    {
      err = dlerror ();
      fprintf (stderr, "%s\n", err);
      exit (1);
    }

}

/* If true, call another function from malloc.  */
static bool call_function;

/* Set to true to indicate that the interposed malloc was called.  */
static bool interposed_malloc_called;

/* Interposed malloc which optionally calls another function.  */
void *
malloc (size_t size)
{
  interposed_malloc_called = true;
  static void *(*original_malloc) (size_t);

  if (original_malloc == NULL)
    {
      static bool in_initialization;
      if (in_initialization)
	{
	  const char *message
	    = "error: malloc called recursively during initialization\n";
	  (void) write (STDOUT_FILENO, message, strlen (message));
	  _exit (2);
	}
      in_initialization = true;

      original_malloc
	= (__typeof (original_malloc)) dlsym (RTLD_NEXT, "malloc");
      if (original_malloc == NULL)
	{
	  const char *message
	    = "error: dlsym for malloc failed\n";
	  (void) write (STDOUT_FILENO, message, strlen (message));
	  _exit (2);
	}
    }

  if (call_function)
    {
      call_function = false;
      call_func (DSO1, FUNC1);
      call_function = true;
    }
  return original_malloc (size);
}

static int
do_test (void)
{
  /* Ensure initialization.  */
  {
    void *volatile ptr = malloc (1);
    free (ptr);
  }

  if (!interposed_malloc_called)
    {
      printf ("error: interposed malloc not called during initialization\n");
      return 1;
    }

  call_function = true;

  /* Bug 17702 fixes two things:
       * A recursive dlopen unmapping the ld.so.cache.
       * An assertion that _r_debug is RT_CONSISTENT at entry to dlopen.
     We can only test the latter. Testing the former requires modifying
     ld.so.conf to cache the dummy libraries, then running ldconfig,
     then run the test. If you do all of that (and glibc's test
     infrastructure doesn't support that yet) then the test will
     SEGFAULT without the fix. If you don't do that, then the test
     will abort because of the assert described in detail below.  */
  call_func (DSO, FUNC);

  call_function = false;

  /* The function dummy2() is called by the malloc hook. Check to
     see that it was called. This ensures the second recursive
     dlopen happened and we called the function in that library.
     Before the fix you either get a SIGSEGV when accessing mmap'd
     ld.so.cache data or an assertion failure about _r_debug not
     beint RT_CONSISTENT.  We don't test for the SIGSEGV since it
     would require finding moddummy1 or moddummy2 in the cache and
     we don't have any infrastructure to test that, but the _r_debug
     assertion triggers.  */
  printf ("Returned result is %d\n", func_result);
  if (func_result <= 0)
    {
      printf ("FAIL: Function call_func() not called.\n");
      exit (1);
    }

  printf ("PASS: Function call_func() called more than once.\n");
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
