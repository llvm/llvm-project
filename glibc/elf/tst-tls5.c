#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>


static int
do_test (void)
{
  static const char modname[] = "tst-tlsmod2.so";
  int result = 0;
  int *foop;
  int *foop2;
  int (*fp) (int, int *);
  void *h;

  h = dlopen (modname, RTLD_LAZY);
  if (h == NULL)
    {
      printf ("cannot open '%s': %s\n", modname, dlerror ());
      exit (1);
    }

  foop = dlsym (h, "foo");
  if (foop == NULL)
    {
      printf ("cannot get symbol 'foo': %s\n", dlerror ());
      exit (1);
    }

  *foop = 42;

  fp = dlsym (h, "in_dso");
  if (fp == NULL)
    {
      printf ("cannot get symbol 'in_dso': %s\n", dlerror ());
      exit (1);
    }

  result |= fp (42, foop);

  foop2 = dlsym (h, "foo");
  if (foop2 == NULL)
    {
      printf ("cannot get symbol 'foo' the second time: %s\n", dlerror ());
      exit (1);
    }

  if (foop != foop2)
    {
      puts ("address of 'foo' different the second time");
      result = 1;
    }
  else if (*foop != 16)
    {
      puts ("foo != 16");
      result = 1;
    }

  dlclose (h);

  return result;
}


#include <support/test-driver.c>
