#include <dlfcn.h>
#include <stdlib.h>
#include <stdio.h>

static int
do_test (void)
{
  char modname[sizeof "tst-tlsmod18aXX.so"];
  void *h[20];
  for (int i = 0; i < 20; i++)
    {
      snprintf (modname, sizeof modname, "tst-tlsmod18a%d.so", i);
      h[i] = dlopen (modname, RTLD_LAZY);
      if (h[i] == NULL)
	{
	  printf ("unexpectedly failed to open %s", modname);
	  exit (1);
	}
    }

  for (int i = 0; i < 20; i++)
    {
      int (*fp) (void) = (int (*) (void)) dlsym (h[i], "test");
      if (fp == NULL)
	{
	  printf ("cannot find test in tst-tlsmod18a%d.so", i);
	  exit (1);
	}

      if (fp ())
	exit (1);
    }

  return 0;
}

#include <support/test-driver.c>
