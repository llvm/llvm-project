#include <config.h>
#include <dlfcn.h>
#include <stdio.h>
#include <sys/mman.h>

static int
do_test (void)
{
  void *h1 = dlopen ("tst-unique1mod1.so", RTLD_LAZY);
  if (h1 == NULL)
    {
      puts ("cannot load tst-unique1mod1");
      return 1;
    }
  int *(*f1) (void) = dlsym (h1, "f");
  if (f1 == NULL)
    {
      puts ("cannot locate f in tst-unique1mod1");
      return 1;
    }
  void *h2 = dlopen ("tst-unique1mod2.so", RTLD_LAZY);
  if (h2 == NULL)
    {
      puts ("cannot load tst-unique1mod2");
      return 1;
    }
  int (*f2) (int *) = dlsym (h2, "f");
  if (f2 == NULL)
    {
      puts ("cannot locate f in tst-unique1mod2");
      return 1;
    }
  if (f2 (f1 ()))
    {
      puts ("f from tst-unique1mod2 failed");
      return 1;
    }
  dlclose (h2);
  dlclose (h1);
  mmap (NULL, 1024 * 1024 * 16, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  h2 = dlopen ("tst-unique1mod2.so", RTLD_LAZY);
  if (h2 == NULL)
    {
      puts ("cannot load tst-unique1mod2");
      return 1;
    }
  f2 = dlsym (h2, "f");
  if (f2 == NULL)
    {
      puts ("cannot locate f in tst-unique1mod2");
      return 1;
    }
  h1 = dlopen ("tst-unique1mod1.so", RTLD_LAZY);
  if (h1 == NULL)
    {
      puts ("cannot load tst-unique1mod1");
      return 1;
    }
  f1 = dlsym (h1, "f");
  if (f1 == NULL)
    {
      puts ("cannot locate f in tst-unique1mod1");
      return 1;
    }
  if (f2 (f1 ()))
    {
      puts ("f from tst-unique1mod2 failed");
      return 1;
    }
  return 0;
}

#include <support/test-driver.c>
