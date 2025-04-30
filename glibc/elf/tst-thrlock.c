#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gnu/lib-names.h>

static void *
tf (void *arg)
{
  void *h = dlopen (LIBM_SO, RTLD_LAZY);
  if (h == NULL)
    {
      printf ("dlopen failed: %s\n", dlerror ());
      exit (1);
    }
  if (dlsym (h, "sin") == NULL)
    {
      printf ("dlsym failed: %s\n", dlerror ());
      exit (1);
    }
  if (dlclose (h) != 0)
    {
      printf ("dlclose failed: %s\n", dlerror ());
      exit (1);
    }
  return NULL;
}


static int
do_test (void)
{
#define N 10
  pthread_t th[N];
  for (int i = 0; i < N; ++i)
    {
      int e = pthread_create (&th[i], NULL, tf, NULL);
      if (e != 0)
	{
	  printf ("pthread_create failed with %d (%s)\n", e, strerror (e));
	  return 1;
	}
    }
  for (int i = 0; i < N; ++i)
    {
      void *res;
      int e = pthread_join (th[i], &res);
      if (e != 0 || res != NULL)
	{
	  puts ("thread failed");
	  return 1;
	}
    }
  return 0;
}

#include <support/test-driver.c>
