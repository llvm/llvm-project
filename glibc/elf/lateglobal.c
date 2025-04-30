#include <dlfcn.h>
#include <mcheck.h>
#include <stdio.h>
#include <stdlib.h>

int
main (void)
{
  void *h[2];
  int fail;
  int (*fp) (void);

  mtrace ();

  h[0] = dlopen ("ltglobmod1.so", RTLD_LAZY);
  if (h[0] == NULL)
    {
      printf ("%s: cannot open %s: %s",
	      __FUNCTION__, "ltglobmod1.so", dlerror ());
      exit (EXIT_FAILURE);
    }
  h[1] = dlopen ("ltglobmod2.so", RTLD_LAZY);
  if (h[1] == NULL)
    {
      printf ("%s: cannot open %s: %s",
	      __FUNCTION__, "ltglobmod2.so", dlerror ());
      exit (EXIT_FAILURE);
    }

  puts ("loaded \"ltglobmod1.so\" without RTLD_GLOBAL");

  fp = dlsym (h[1], "foo");
  if (fp == NULL)
    {
      printf ("cannot get address of `foo': %s", dlerror ());
      exit (EXIT_FAILURE);
    }

  fail = fp ();

  puts ("back in main");

  dlclose (h[1]);
  dlclose (h[0]);

  return fail;
}
