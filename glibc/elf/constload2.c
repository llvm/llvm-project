#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

extern int bar (void);
extern int baz (void);
extern int foo (void);
extern void __attribute__ ((__constructor__)) init (void);

void *h;

int
foo (void)
{
  return 42 + bar ();
}

int
baz (void)
{
  return -21;
}


void
__attribute__ ((__constructor__))
init (void)
{
  h = dlopen ("constload3.so", RTLD_GLOBAL | RTLD_LAZY);
  if (h == NULL)
    {
      puts ("failed to load constload3");
      exit (1);
    }
  else
    puts ("succeeded loading constload3");
}

static void
__attribute__ ((__destructor__))
fini (void)
{
  if (dlclose (h) != 0)
    {
      puts ("failed to unload constload3");
      exit (1);
    }
  else
    puts ("succeeded unloading constload3");
}
