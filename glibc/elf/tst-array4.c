#include <dlfcn.h>

#define main array1_main
#include "tst-array1.c"
#undef main

int
main (void)
{
  void *handle = dlopen ("tst-array2dep.so", RTLD_LAZY);

  array1_main ();

  if (handle != NULL)
    dlclose (handle);

  return 0;
}
