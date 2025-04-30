#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

void
mod3_fini2 (void)
{
}

void
mod3_fini (void)
{
  mod3_fini2 ();
}

void
mod3 (void)
{
  void *h = dlopen ("$ORIGIN/unload8mod2.so", RTLD_LAZY);
  if (h == NULL)
    {
      puts ("dlopen unload8mod2.so failed");
      exit (1);
    }

  atexit (mod3_fini);
}
