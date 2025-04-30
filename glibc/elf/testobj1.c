#include <dlfcn.h>
#include <stdlib.h>

#include "testobj.h"

int
obj1func1 (int a __attribute__ ((unused)))
{
  return 42;
}

int
obj1func2 (int a)
{
  return foo (a) + 10;
}

int
preload (int a)
{
  int (*fp) (int) = dlsym (RTLD_NEXT, "preload");
  if (fp != NULL)
    return fp (a) + 10;
  return 10;
}
