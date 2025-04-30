#include <dlfcn.h>
#include <stdio.h>
#include <malloc.h>

int
main (void)
{
#ifdef M_PERTURB
  mallopt (M_PERTURB, 0xaa);
#endif

  void *h;
  int (*fn) (int);
  h = dlopen ("unload4mod1.so", RTLD_LAZY);
  if (h == NULL)
    {
      puts ("1st dlopen failed");
      return 1;
    }
  fn = dlsym (h, "foo");
  if (fn == NULL)
    {
      puts ("dlsym failed");
      return 1;
    }
  int n = fn (10);
  if (n != 28)
    {
      printf ("foo (10) returned %d != 28\n", n);
      return 1;
    }
  dlclose (h);
  h = dlopen ("unload4mod3.so", RTLD_LAZY);
  fn = dlsym (h, "mod3fn2");
  if (fn == NULL)
    {
      puts ("second dlsym failed");
      return 1;
    }
  n = fn (10);
  if (n != 22)
    {
      printf ("mod3fn2 (10) returned %d != 22\n", n);
      return 1;
    }
  dlclose (h);
  return 0;
}
