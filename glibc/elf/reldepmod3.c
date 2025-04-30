#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

extern int call_me (void);

int
call_me (void)
{
  int (*fp) (void);

  fp = dlsym (RTLD_DEFAULT, "foo");
  if (fp == NULL)
    {
      printf ("cannot get address of foo in global scope: %s\n", dlerror ());
      exit (1);
    }

  return fp () - 42;
}
