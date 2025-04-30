/* Test DTV size oveflow when pthread_create reuses old DTV and TLS is
   used by dlopened shared object.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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
#include <stdint.h>
#include <dlfcn.h>
#include <assert.h>
#include <pthread.h>

/* The choices of thread count, and file counts are arbitary.
   The point is simply to run enough threads that an exiting
   thread has it's stack reused by another thread at the same
   time as new libraries have been loaded.  */
#define DSO_SHARED_FILES 20
#define DSO_OPEN_THREADS 20
#define DSO_EXEC_THREADS 2

/* Used to make sure that only one thread is calling dlopen and dlclose
   at a time.  */
pthread_mutex_t g_lock;

typedef void (*function) (void);

void *
dso_invoke(void *dso_fun)
{
  function *fun_vec = (function *) dso_fun;
  int dso;

  for (dso = 0; dso < DSO_SHARED_FILES; dso++)
    (*fun_vec[dso]) ();

  pthread_exit (NULL);
}

void *
dso_process (void * p)
{
  void *handle[DSO_SHARED_FILES];
  function fun_vec[DSO_SHARED_FILES];
  char dso_path[DSO_SHARED_FILES][100];
  int dso;
  int t = (int) (uintptr_t) p;

  /* Open DSOs and get a function.  */
  for (dso = 0; dso < DSO_SHARED_FILES; dso++)
    {
      sprintf (dso_path[dso], "tst-stack4mod-%i-%i.so", t, dso);

      pthread_mutex_lock (&g_lock);

      handle[dso] = dlopen (dso_path[dso], RTLD_NOW);
      assert (handle[dso]);

      fun_vec[dso] = (function) dlsym (handle[dso], "function");
      assert (fun_vec[dso]);

      pthread_mutex_unlock (&g_lock);
    }

  /* Spawn workers.  */
  pthread_t thread[DSO_EXEC_THREADS];
  int i, ret;
  uintptr_t result = 0;
  for (i = 0; i < DSO_EXEC_THREADS; i++)
    {
      pthread_mutex_lock (&g_lock);
      ret = pthread_create (&thread[i], NULL, dso_invoke, (void *) fun_vec);
      if (ret != 0)
	{
	  printf ("pthread_create failed: %d\n", ret);
	  result = 1;
	}
      pthread_mutex_unlock (&g_lock);
    }

  if (!result)
    for (i = 0; i < DSO_EXEC_THREADS; i++)
      {
	ret = pthread_join (thread[i], NULL);
	if (ret != 0)
	  {
	    printf ("pthread_join failed: %d\n", ret);
	    result = 1;
	  }
      }

  /* Close all DSOs.  */
  for (dso = 0; dso < DSO_SHARED_FILES; dso++)
    {
      pthread_mutex_lock (&g_lock);
      dlclose (handle[dso]);
      pthread_mutex_unlock (&g_lock);
    }

  /* Exit.  */
  pthread_exit ((void *) result);
}

static int
do_test (void)
{
  pthread_t thread[DSO_OPEN_THREADS];
  int i,j;
  int ret;
  int result = 0;

  pthread_mutex_init (&g_lock, NULL);

  /* 100 is arbitrary here and is known to trigger PR 13862.  */
  for (j = 0; j < 100; j++)
    {
      for (i = 0; i < DSO_OPEN_THREADS; i++)
	{
	  ret = pthread_create (&thread[i], NULL, dso_process,
				(void *) (uintptr_t) i);
	  if (ret != 0)
	    {
	      printf ("pthread_create failed: %d\n", ret);
	      result = 1;
	    }
	}

      if (result)
	break;

      for (i = 0; i < DSO_OPEN_THREADS; i++)
	{
	  ret = pthread_join (thread[i], NULL);
	  if (ret != 0)
	    {
	      printf ("pthread_join failed: %d\n", ret);
	      result = 1;
	    }
	}
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#define TIMEOUT 100
#include "../test-skeleton.c"
