#include <dlfcn.h>
#include <stdlib.h>

#include "testobj.h"


int
obj3func1 (int a __attribute__ ((unused)))
{
  return 44;
}

int
obj3func2 (int a)
{
  return foo (a) + 42;
}

int
preload (int a)
{
  int (*fp) (int) = dlsym (RTLD_NEXT, "preload");
  if (fp != NULL)
    return fp (a) + 10;
  return 10;
}
