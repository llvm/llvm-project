#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>

static void *h;

static void __attribute__((constructor))
mod2init (void)
{
  h = dlopen ("unload6mod3.so", RTLD_LAZY);
  if (h == NULL)
    {
      puts ("dlopen unload6mod3.so failed");
      fflush (stdout);
      _exit (1);
    }
}

static void __attribute__((destructor))
mod2fini (void)
{
  dlclose (h);
}
