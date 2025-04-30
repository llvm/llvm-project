#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

extern int call_me (void);

int
call_me (void)
{
  void *h;
  int (*fp) (void);
  int res;

  h = dlopen ("reldepmod1.so", RTLD_LAZY);
  if (h == NULL)
    {
      printf ("cannot open reldepmod1.so in %s: %s\n", __FILE__, dlerror ());
      exit (1);
    }

  fp = dlsym (h, "foo");
  if (fp == NULL)
    {
      printf ("cannot get address of foo in global scope: %s\n", dlerror ());
      exit (1);
    }

  res = fp () - 42;

  if (dlclose (h) != 0)
    {
      printf ("failure when closing h in %s: %s\n", __FILE__, dlerror ());
      exit (1);
    }

  return res;
}
