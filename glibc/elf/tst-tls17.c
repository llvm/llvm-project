#include <dlfcn.h>
#include <stdlib.h>
#include <stdio.h>

static int
do_test (void)
{
  void *h = dlopen ("tst-tlsmod17b.so", RTLD_LAZY);
  if (h == NULL)
    {
      puts ("unexpectedly failed to open tst-tlsmod17b.so");
      exit (1);
    }

  int (*fp) (void) = (int (*) (void)) dlsym (h, "tlsmod17b");
  if (fp == NULL)
    {
      puts ("cannot find tlsmod17b");
      exit (1);
    }

  if (fp ())
    exit (1);

  return 0;
}

#include <support/test-driver.c>
