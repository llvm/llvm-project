/* Check unloading modules with data in static TLS block.  */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


static int
do_test (void)
{
  for (int i = 0; i < 1000;)
    {
      printf ("round %d\n",++i);

      void *h = dlopen ("$ORIGIN/tst-tlsmod13a.so", RTLD_LAZY);
      if (h == NULL)
	{
	  printf ("cannot load: %s\n", dlerror ());
	  exit (1);
	}

      dlclose (h);
    }

  return 0;
}

#include <support/test-driver.c>
