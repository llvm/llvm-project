#include <dlfcn.h>
#include <mcheck.h>
#include <stdio.h>
#include <stdlib.h>

int
main (void)
{
  int i;
  void *h1, *h2;

  mtrace ();

  for (i = 0; i < 3; i++)
    {
      h1 = dlopen ("reldep4mod1.so", RTLD_NOW | RTLD_GLOBAL);
      if (h1 == NULL)
	{
	  printf ("cannot open reldep4mod1.so: %s\n", dlerror ());
	  exit (1);
	}
      h2 = dlopen ("reldep4mod2.so", RTLD_NOW | RTLD_GLOBAL);
      if (h2 == NULL)
	{
	  printf ("cannot open reldep4mod2.so: %s\n", dlerror ());
	  exit (1);
	}
      if (dlclose (h1) != 0)
	{
	  printf ("closing h1 failed: %s\n", dlerror ());
	  exit (1);
	}
      if (dlclose (h2) != 0)
	{
	  printf ("closing h2 failed: %s\n", dlerror ());
	  exit (1);
	}
    }
  return 0;
}
