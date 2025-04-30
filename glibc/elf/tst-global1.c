#include <dlfcn.h>
#include <stdio.h>

static int
do_test (void)
{
  void *h1 = dlopen ("$ORIGIN/testobj6.so", RTLD_GLOBAL|RTLD_LAZY);
  if (h1 == NULL)
    {
      puts ("cannot open testobj6");
      return 1;
    }

  void *h2 = dlopen ("$ORIGIN/testobj2.so",
		     RTLD_GLOBAL|RTLD_DEEPBIND|RTLD_LAZY);
  if (h2 == NULL)
    {
      puts ("cannot open testobj2");
      return 1;
    }

  dlclose (h1);

  void (*f) (void) = dlsym (h2, "p");
  if (f == NULL)
    {
      puts ("cannot find p");
      return 1;
    }

  f ();

  dlclose (h2);

  return 0;
}

#include <support/test-driver.c>
