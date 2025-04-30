#include <dlfcn.h>
#include <stdio.h>

int
main (void)
{
  void *g = dlopen ("unload3mod1.so", RTLD_GLOBAL | RTLD_NOW);
  void *h = dlopen ("unload3mod2.so", RTLD_GLOBAL | RTLD_NOW);
  if (g == NULL || h == NULL)
    {
      printf ("dlopen unload3mod{1,2}.so failed: %p %p\n", g, h);
      return 1;
    }
  dlopen ("unload3mod4.so", RTLD_GLOBAL | RTLD_NOW);
  dlclose (h);
  dlclose (g);

  g = dlopen ("unload3mod3.so", RTLD_GLOBAL | RTLD_NOW);
  h = dlopen ("unload3mod4.so", RTLD_GLOBAL | RTLD_NOW);
  if (g == NULL || h == NULL)
    {
      printf ("dlopen unload3mod{3,4}.so failed: %p %p\n", g, h);
      return 1;
    }

  int (*fn) (int);
  fn = dlsym (h, "bar");
  if (fn == NULL)
    {
      puts ("dlsym failed");
      return 1;
    }

  int val = fn (16);
  if (val != 24)
    {
      printf ("bar returned %d != 24\n", val);
      return 1;
    }

  return 0;
}
