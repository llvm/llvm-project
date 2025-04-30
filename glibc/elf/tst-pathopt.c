#include <dlfcn.h>
#include <mcheck.h>
#include <stdio.h>
#include <stdlib.h>


static int
do_test (void)
{
  void *h;
  int (*fp) (int);
  int result;

  mtrace ();

  h = dlopen ("renamed.so", RTLD_LAZY);
  if (h == NULL)
    {
      printf ("failed to load \"%s\": %s\n", "renamed.so", dlerror ());
      exit (1);
    }

  fp = dlsym (h, "in_renamed");
  if (fp == NULL)
    {
      printf ("lookup of \"%s\" failed: %s\n", "in_renamed", dlerror ());
      exit (1);
    }

  result = fp (10);

  if (dlclose (h) != 0)
    {
      printf ("failed to close \"%s\": %s\n", "renamed.so", dlerror ());
      exit (1);
    }

  return result;
}

#include <support/test-driver.c>
