#include <dlfcn.h>
#include <stdio.h>

int
main (void)
{
  void *h = dlopen ("unload6mod1.so", RTLD_LAZY);
  if (h == NULL)
    {
      puts ("dlopen unload6mod1.so failed");
      return 1;
    }

  int (*fn) (int);
  fn = dlsym (h, "foo");
  if (fn == NULL)
    {
      puts ("dlsym failed");
      return 1;
    }

  int val = fn (16);
  if (val != 24)
    {
      printf ("foo returned %d != 24\n", val);
      return 1;
    }

  return 0;
}
