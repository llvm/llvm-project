#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

static int
do_test (void)
{
  void *handle;
  int (*test) (int);
  int res;

  handle = dlopen ("modstatic.so", RTLD_LAZY);
  if (handle == NULL)
    {
      printf ("%s\n", dlerror ());
      exit(1);
    }

  test = dlsym (handle, "test");
  if (test == NULL)
    {
      printf ("%s\n", dlerror ());
      exit(1);
    }

  res = test (2);
  if (res != 4)
    {
      printf ("Got %i, expected 4\n", res);
      exit (1);
    }

  dlclose (handle);
  return 0;
}

#define TEST_FUNCTION   do_test ()
#include "../test-skeleton.c"
