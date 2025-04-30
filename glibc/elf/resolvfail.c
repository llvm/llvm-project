#include <dlfcn.h>
#include <stdio.h>

static const char obj[] = "testobj1.so";

int
main (void)
{
  void *d = dlopen (obj, RTLD_LAZY);
  int n;

  if (d == NULL)
    {
      printf ("cannot load %s: %s\n", obj, dlerror ());
      return 1;
    }

  for (n = 0; n < 10000; ++n)
    if (dlsym (d, "does not exist") != NULL)
      {
	puts ("dlsym() did not fail");
	return 1;
      }
    else if (dlerror () == NULL)
      {
	puts ("dlerror() didn't return a string");
	return 1;
      }

  return 0;
}
