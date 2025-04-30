#include <dlfcn.h>
#include <stdio.h>

static int
do_test (void)
{
  void *handle = dlopen ("$ORIGIN/bug-atexit3-lib.so", RTLD_LAZY);
  if (handle == NULL)
    {
      printf ("dlopen failed: %s\n", dlerror ());
      return 1;
    }
  dlclose (handle);
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
