#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

int
main (void)
{
  void *h1;
  void *h2;
  void *mod1_bar, *mod2_bar;

  h1 = dlopen ("reldep7mod1.so", RTLD_GLOBAL | RTLD_LAZY);
  if (h1 == NULL)
    {
      printf ("cannot open reldep7mod1.so: %s\n", dlerror ());
      exit (1);
    }

  h2 = dlopen ("reldep7mod2.so", RTLD_GLOBAL | RTLD_LAZY);
  if (h2 == NULL)
    {
      printf ("cannot open reldep7mod1.so: %s\n", dlerror ());
      exit (1);
    }

  mod1_bar = dlsym (h1, "mod1_bar");
  if (mod1_bar == NULL)
    {
      printf ("cannot get address of \"mod1_bar\": %s\n", dlerror ());
      exit (1);
    }

  mod2_bar = dlsym (h2, "mod2_bar");
  if (mod2_bar == NULL)
    {
      printf ("cannot get address of \"mod2_bar\": %s\n", dlerror ());
      exit (1);
    }

  printf ("%d\n", ((int (*) (void)) mod1_bar) ());
  printf ("%d\n", ((int (*) (void)) mod2_bar) ());

  if (dlclose (h1) != 0)
    {
      printf ("closing h1 failed: %s\n", dlerror ());
      exit (1);
    }

  printf ("%d\n", ((int (*) (void)) mod2_bar) ());

  if (dlclose (h2) != 0)
    {
      printf ("closing h2 failed: %s\n", dlerror ());
      exit (1);
    }

  return 0;
}
