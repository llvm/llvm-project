#include <dlfcn.h>
#include <stdio.h>

int
main (void)
{
  void *h = dlopen ("firstobj.so", RTLD_LAZY);
  void *f;
  if (! h)
    {
      printf ("cannot find firstobj.so: %s\n", dlerror ());
      return 1;
    }
  f = dlsym (h, "foo");
  if (! f)
    {
      printf ("cannot find symbol foo: %s\n", dlerror ());
      return 2;
    }
  ((void (*) (void)) f) ();
  return 0;
}
