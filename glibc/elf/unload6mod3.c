#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>

static void *h;

static void __attribute__((constructor))
mod3init (void)
{
  h = dlopen ("unload6mod1.so", RTLD_LAZY);
  if (h == NULL)
    {
      puts ("dlopen unload6mod1.so failed");
      fflush (stdout);
      _exit (1);
    }
}

static void __attribute__((destructor))
mod3fini (void)
{
  dlclose (h);
}
