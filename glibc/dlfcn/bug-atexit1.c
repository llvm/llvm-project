/* Derived from a test case in
   https://sourceware.org/bugzilla/show_bug.cgi?id=1158.  */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

static int
do_test (void)
{
  for (int i = 0; i < 2; ++i)
    {
      void *dso = dlopen ("$ORIGIN/bug-atexit1-lib.so", RTLD_NOW);
      void (*fn) (void) = (void (*) (void)) dlsym (dso, "foo");
      fn ();
      dlclose (dso);
      puts ("round done");
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
