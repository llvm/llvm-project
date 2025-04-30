#include <dlfcn.h>
#include <stdlib.h>
#include <stdio.h>

#include "testobj.h"

int
obj2func1 (int a __attribute__ ((unused)))
{
  return 43;
}

int
obj2func2 (int a)
{
  return obj1func1 (a) + 10;
}

int
preload (int a)
{
  int (*fp) (int) = dlsym (RTLD_NEXT, "preload");
  if (fp != NULL)
    return fp (a) + 10;
  return 10;
}

void
p (void)
{
  puts ("hello world");
}
