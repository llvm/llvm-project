#include <dlfcn.h>
#include <stdio.h>

int
foo (int i)
{
  if (dlsym (RTLD_DEFAULT, "unload7_nonexistent_symbol") == NULL)
    return 1;
  puts ("dlsym returned non-NULL");
  return 0;
}
