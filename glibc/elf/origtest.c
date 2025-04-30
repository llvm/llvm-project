#include <dlfcn.h>
#include <error.h>
#include <stdio.h>
#include <stdlib.h>

int
main (void)
{
  void *h;
  int (*fp) (int);
  int res;

  h = dlopen ("${ORIGIN}/testobj1.so", RTLD_LAZY);
  if (h == NULL)
    error (EXIT_FAILURE, 0, "while loading `%s': %s", "testobj1.so",
	   dlerror ());

  fp = dlsym (h, "obj1func1");
  if (fp == NULL)
    error (EXIT_FAILURE, 0, "getting `obj1func1' in `%s': %s",
	   "testobj1.so", dlerror ());

  res = fp (10);
  printf ("fp(10) = %d\n", res);

  if (dlclose (h) != 0)
    error (EXIT_FAILURE, 0, "while close `%s': %s",
	   "testobj1.so", dlerror ());

  return res != 42;
}


extern int foo (int a);
int
foo (int a)
{
  return a + 10;
}
