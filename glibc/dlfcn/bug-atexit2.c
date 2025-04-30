/* Derived from a test case in
   https://sourceware.org/bugzilla/show_bug.cgi?id=1158.  */
#include <dlfcn.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

static int next = 3;

static void
f1 (void)
{
  puts ("f1");
  if (next-- != 1)
    _exit (1);
}

static void
f2 (void)
{
  puts ("f2");
  if (next-- != 2)
    _exit (1);
}

static void
f3 (void)
{
  puts ("f3");
  if (next-- != 3)
    _exit (1);
}

static int
do_test (void)
{
  atexit (f1);

  void *dso = dlopen ("$ORIGIN/bug-atexit2-lib.so", RTLD_NOW);
  void (*fn) (void) = (void (*) (void)) dlsym (dso, "foo");
  fn ();

  atexit (f2);

  dlclose (dso);

  atexit (f3);

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
