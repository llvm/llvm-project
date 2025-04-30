#include <config.h>
#include <dlfcn.h>
#include <stdio.h>

extern int var;

static int
do_test (void)
{
  var = 1;

  void *h = dlopen ("tst-unique2mod2.so", RTLD_LAZY);
  if (h == NULL)
    {
      puts ("cannot load tst-unique2mod2");
      return 1;
    }
  int (*f) (int *) = dlsym (h, "f");
  if (f == NULL)
    {
      puts ("cannot locate f in tst-unique2mod2");
      return 1;
    }
  return f (&var);
}

#include <support/test-driver.c>
