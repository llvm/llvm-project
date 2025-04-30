#include <dlfcn.h>
#include <stdio.h>

int
main (void)
{
  void *h = dlopen ("$ORIGIN/unload8mod1.so", RTLD_LAZY);
  if (h == NULL)
    {
      puts ("dlopen unload8mod1.so failed");
      return 1;
    }

  void *h2 = dlopen ("$ORIGIN/unload8mod1x.so", RTLD_LAZY);
  if (h2 == NULL)
    {
      puts ("dlopen unload8mod1x.so failed");
      return 1;
    }
  dlclose (h2);

  int (*mod1) (void) = dlsym (h, "mod1");
  if (mod1 == NULL)
    {
      puts ("dlsym failed");
      return 1;
    }

  mod1 ();
  dlclose (h);

  return 0;
}
