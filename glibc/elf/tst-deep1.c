#include <dlfcn.h>
#include <stdio.h>

int
xyzzy (void)
{
  printf ("%s:%s\n", __FILE__, __func__);
  return 21;
}

int
back (void)
{
  printf ("%s:%s\n", __FILE__, __func__);
  return 1;
}

extern int foo (void);

static int
do_test (void)
{
  void *p = dlopen ("$ORIGIN/tst-deep1mod2.so", RTLD_LAZY|RTLD_DEEPBIND);

  int (*f) (void) = dlsym (p, "bar");
  if (f == NULL)
    {
      puts (dlerror ());
      return 1;
    }

  return foo () + f ();
}

#include <support/test-driver.c>
