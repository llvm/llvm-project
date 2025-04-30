#include <dlfcn.h>
#include <stdlib.h>

#include "testobj.h"

int
obj4func1 (int a __attribute__ ((unused)))
{
  return 55;
}

int
obj4func2 (int a)
{
  return foo (a) + 43;
}

int
preload (int a)
{
  int (*fp) (int) = dlsym (RTLD_NEXT, "preload");
  if (fp != NULL)
    return fp (a) + 10;
  return 10;
}
