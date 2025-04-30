#include <dlfcn.h>
#include <stdio.h>

int
main (void)
{
  void *h = dlopen ("$ORIGIN/unload7mod1.so", RTLD_LAZY);
  if (h == NULL)
    {
      puts ("dlopen unload7mod1.so failed");
      return 1;
    }

  int (*fn) (void);
  fn = dlsym (h, "foo");
  if (fn == NULL)
    {
      puts ("dlsym failed");
      return 1;
    }

  int ret = 0;
  if (fn () == 0)
    ++ret;

  void *h2 = dlopen ("$ORIGIN/unload7mod2.so", RTLD_LAZY);
  if (h2 == NULL)
    {
      puts ("dlopen unload7mod2.so failed");
      return 1;
    }
  dlclose (h2);

  if (fn () == 0)
    ++ret;

  dlclose (h);
  return ret;
}
