#include <dlfcn.h>
#include <error.h>
#include <mcheck.h>
#include <stdio.h>
#include <stdlib.h>

int
main (void)
{
  void *su[5];
  void *h;
  int n;

  mtrace ();

  if ((su[0] = dlopen ("testobj1.so", RTLD_GLOBAL | RTLD_NOW)) == NULL
      || (su[1] = dlopen ("testobj2.so", RTLD_GLOBAL | RTLD_NOW)) == NULL
      || (su[2] = dlopen ("testobj3.so", RTLD_GLOBAL | RTLD_NOW)) == NULL
      || (su[3] = dlopen ("testobj4.so", RTLD_GLOBAL | RTLD_NOW)) == NULL
      || (su[4] = dlopen ("testobj5.so", RTLD_GLOBAL | RTLD_NOW)) == NULL)
    error (EXIT_FAILURE, 0, "failed to load shared object: %s", dlerror ());

  h = dlopen ("failobj.so", RTLD_GLOBAL | RTLD_NOW);

  printf ("h = %p, %s\n", h, h == NULL ? "ok" : "fail");

  for (n = 0; n < 5; ++n)
    if (dlclose (su[n]) != 0)
      {
	printf ("failed to unload su[%d]: %s\n", n, dlerror ());
	exit (EXIT_FAILURE);
      }

  return h != NULL;
}

extern int foo (int a);
int
foo (int a)
{
  return 10;
}
