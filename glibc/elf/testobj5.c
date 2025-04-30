#include <dlfcn.h>
#include <stdlib.h>

#include "testobj.h"


int
obj5func1 (int a __attribute__ ((unused)))
{
  return 66;
}

int
obj5func2 (int a)
{
  return foo (a) + 44;
}

int
preload (int a)
{
  int (*fp) (int) = dlsym (RTLD_NEXT, "preload");
  if (fp != NULL)
    return fp (a) + 10;
  return 10;
}
