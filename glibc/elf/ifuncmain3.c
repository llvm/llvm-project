/* Test STT_GNU_IFUNC symbols with dlopen:

   1. Direct function call.
   2. Function pointer.
   3. Visibility with override.
 */

#include <dlfcn.h>
#include <stdlib.h>
#include <stdio.h>

typedef int (*foo_p) (void);

int
__attribute__ ((noinline))
foo (void)
{
  return -30;
}

int
__attribute__ ((noinline))
foo_hidden (void)
{
  return -20;
}

int
__attribute__ ((noinline))
foo_protected (void)
{
  return -40;
}

int
main (void)
{
  foo_p p;
  foo_p (*f) (void);
  int *ret;

  void *h = dlopen ("ifuncmod3.so", RTLD_LAZY);
  if (h == NULL)
    {
      printf ("cannot load: %s\n", dlerror ());
      return 1;
    }

  p = dlsym (h, "foo");
  if (p == NULL)
    {
      printf ("symbol not found: %s\n", dlerror ());
      return 1;
    }
  if ((*p) () != -1)
    abort ();

  f = dlsym (h, "get_foo_p");
  if (f == NULL)
    {
      printf ("symbol not found: %s\n", dlerror ());
      return 1;
    }

  ret = dlsym (h, "ret_foo");
  if (ret == NULL)
    {
      printf ("symbol not found: %s\n", dlerror ());
      return 1;
    }

  p = (*f) ();
  if (p != foo)
    abort ();
  if (foo () != -30)
    abort ();
  if (*ret != -30 || (*p) () != *ret)
    abort ();

  f = dlsym (h, "get_foo_hidden_p");
  if (f == NULL)
    {
      printf ("symbol not found: %s\n", dlerror ());
      return 1;
    }

  ret = dlsym (h, "ret_foo_hidden");
  if (ret == NULL)
    {
      printf ("symbol not found: %s\n", dlerror ());
      return 1;
    }

  p = (*f) ();
  if (foo_hidden () != -20)
    abort ();
  if (*ret != 1 || (*p) () != *ret)
    abort ();

  f = dlsym (h, "get_foo_protected_p");
  if (f == NULL)
    {
      printf ("symbol not found: %s\n", dlerror ());
      return 1;
    }

  ret = dlsym (h, "ret_foo_protected");
  if (ret == NULL)
    {
      printf ("symbol not found: %s\n", dlerror ());
      return 1;
    }

  p = (*f) ();
  if (p == foo_protected)
    abort ();
  if (foo_protected () != -40)
    abort ();
  if (*ret != 0 || (*p) () != *ret)
    abort ();

  if (dlclose (h) != 0)
    {
      printf ("cannot close: %s\n", dlerror ());
      return 1;
    }

  return 0;
}
