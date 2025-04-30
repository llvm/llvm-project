#include <dlfcn.h>
#include <mcheck.h>
#include <stdio.h>
#include <stdlib.h>

typedef int (*fn)(void);
#define CHUNKS 1024
#define REPEAT 64

int
main (void)
{
  void *h1;
  void *h2;
  fn **foopp;
  fn bar, baz;
  int i, j;
  int n;
  void *allocs[REPEAT][CHUNKS];

  mtrace ();

  /* Open the two objects.  */
  h1 = dlopen ("reldep6mod3.so", RTLD_LAZY);
  if (h1 == NULL)
    {
      printf ("cannot open reldep6mod3.so: %s\n", dlerror ());
      exit (1);
    }

  foopp = dlsym (h1, "foopp");
  if (foopp == NULL)
    {
      printf ("cannot get address of \"foopp\": %s\n", dlerror ());
      exit (1);
    }
  n = (**foopp) ();
  if (n != 20)
    {
      printf ("(**foopp)() return %d, not return 20\n", n);
      exit (1);
    }

  h2 = dlopen ("reldep6mod4.so", RTLD_LAZY);
  if (h2 == NULL)
    {
      printf ("cannot open reldep6mod4.so: %s\n", dlerror ());
      exit (1);
    }

  baz = dlsym (h2, "baz");
  if (baz == NULL)
    {
      printf ("cannot get address of \"baz\": %s\n", dlerror ());
      exit (1);
    }
  if (baz () != 31)
    {
      printf ("baz() did not return 31\n");
      exit (1);
    }

  if (dlclose (h1) != 0)
    {
      printf ("closing h1 failed: %s\n", dlerror ());
      exit (1);
    }

  /* Clobber memory.  */
  for (i = 0; i < REPEAT; ++i)
    for (j = 0; j < CHUNKS; ++j)
      allocs[i][j] = calloc (1, j + 1);

  bar = dlsym (h2, "bar");
  if (bar == NULL)
    {
      printf ("cannot get address of \"bar\": %s\n", dlerror ());
      exit (1);
    }
  if (bar () != 40)
    {
      printf ("bar() did not return 40\n");
      exit (1);
    }

  baz = dlsym (h2, "baz");
  if (baz == NULL)
    {
      printf ("cannot get address of \"baz\": %s\n", dlerror ());
      exit (1);
    }
  if (baz () != 31)
    {
      printf ("baz() did not return 31\n");
      exit (1);
    }

  for (i = 0; i < REPEAT; ++i)
    for (j = 0; j < CHUNKS; ++j)
      free (allocs[i][j]);

  if (dlclose (h2) != 0)
    {
      printf ("closing h2 failed: %s\n", dlerror ());
      exit (1);
    }

  return 0;
}
