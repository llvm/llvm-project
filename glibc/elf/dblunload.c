#include <dlfcn.h>
#include <mcheck.h>
#include <stdio.h>
#include <stdlib.h>


int
main (void)
{
  void *p1;
  void *p2;
  int (*fp) (void);
  int result;

  mtrace ();

  p1 = dlopen ("dblloadmod1.so", RTLD_LAZY);
  if (p1 == NULL)
    {
      printf ("cannot load dblloadmod1.so: %s\n", dlerror ());
      exit (EXIT_FAILURE);
    }

  p2 = dlopen ("dblloadmod2.so", RTLD_LAZY);
  if (p2 == NULL)
    {
      printf ("cannot load dblloadmod2.so: %s\n", dlerror ());
      exit (EXIT_FAILURE);
    }

  if (dlclose (p1) != 0)
    {
      printf ("error while closing dblloadmod1.so: %s\n", dlerror ());
      exit (EXIT_FAILURE);
    }

  fp = dlsym (p2, "xyzzy");
  if (fp == NULL)
    {
      printf ("cannot get function \"xyzzy\": %s\n", dlerror ());
      exit (EXIT_FAILURE);
    }

  result = fp ();

  if (dlclose (p2) != 0)
    {
      printf ("error while closing dblloadmod2.so: %s\n", dlerror ());
      exit (EXIT_FAILURE);
    }

  return result;
}
