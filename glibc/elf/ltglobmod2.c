#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

extern int bar (void);
extern int foo (void);

int
foo (void)
{
  void *h;
  int res;

  /* Load ltglobalmod1 in the global namespace.  */
  h = dlopen ("ltglobmod1.so", RTLD_GLOBAL | RTLD_LAZY);
  if (h == NULL)
    {
      printf ("%s: cannot open %s: %s",
	      __FUNCTION__, "ltglobmod1.so", dlerror ());
      exit (EXIT_FAILURE);
    }

  /* Call bar.  This is undefined in the DSO.  */
  puts ("about to call `bar'");
  fflush (stdout);
  res = bar ();

  printf ("bar returned %d\n", res);

  dlclose (h);

  return res != 42;
}
