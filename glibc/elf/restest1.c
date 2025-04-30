#include <dlfcn.h>
#include <error.h>
#include <mcheck.h>
#include <stdio.h>
#include <stdlib.h>

int
main (void)
{
  void *h1;
  int (*fp1) (int);
  void *h2;
  int (*fp2) (int);
  int res1;
  int res2;

  mtrace ();

  h1 = dlopen ("testobj1.so", RTLD_LAZY);
  if (h1 == NULL)
    error (EXIT_FAILURE, 0, "while loading `%s': %s", "testobj1.so",
	   dlerror ());

  h2 = dlopen ("testobj1_1.so", RTLD_LAZY);
  if (h1 == NULL)
    error (EXIT_FAILURE, 0, "while loading `%s': %s", "testobj1_1.so",
	   dlerror ());

  fp1 = dlsym (h1, "obj1func1");
  if (fp1 == NULL)
    error (EXIT_FAILURE, 0, "getting `obj1func1' in `%s': %s",
	   "testobj1.so", dlerror ());

  fp2 = dlsym (h2, "obj1func1");
  if (fp2 == NULL)
    error (EXIT_FAILURE, 0, "getting `obj1func1' in `%s': %s",
	   "testobj1_1.so", dlerror ());

  res1 = fp1 (10);
  res2 = fp2 (10);
  printf ("fp1(10) = %d\nfp2(10) = %d\n", res1, res2);

  if (dlclose (h1) != 0)
    error (EXIT_FAILURE, 0, "cannot close testobj1.so: %s\n", dlerror ());
  if (dlclose (h2) != 0)
    error (EXIT_FAILURE, 0, "cannot close testobj1_1.so: %s\n", dlerror ());

  return res1 != 42 || res2 != 72;
}


extern int foo (int a);
int
foo (int a)
{
  return a + 10;
}
