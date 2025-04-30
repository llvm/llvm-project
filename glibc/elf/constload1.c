#include <dlfcn.h>
#include <errno.h>
#include <error.h>
#include <mcheck.h>
#include <stdio.h>
#include <stdlib.h>

int
main (void)
{
  int (*foo) (void);
  void *h;
  int ret;

  mtrace ();

  h = dlopen ("constload2.so", RTLD_LAZY | RTLD_GLOBAL);
  if (h == NULL)
    error (EXIT_FAILURE, errno, "cannot load module \"constload2.so\"");
  foo = dlsym (h, "foo");
  ret = foo ();
  /* Note that the following dlclose() call cannot unload the objects.
     Due to the introduced relocation dependency constload2.so depends
     on constload3.so and the dependencies of constload2.so on constload3.so
     is not visible to ld.so since it's done using dlopen().  */
  if (dlclose (h) != 0)
    {
      puts ("failed to close");
      exit (EXIT_FAILURE);
    }
  return ret;
}
