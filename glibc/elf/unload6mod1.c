#include <dlfcn.h>
#include <stdio.h>

int
foo (int i)
{
  void *h = dlopen ("unload6mod2.so", RTLD_LAZY);
  if (h == NULL)
    {
      puts ("dlopen unload6mod2.so failed");
      return 1;
    }

  dlclose (h);
  return i + 8;
}
