// BZ 12453
#include <stdio.h>
#include <dlfcn.h>


static int
do_test (void)
{
  void* dl = dlopen ("tst-tls19mod1.so", RTLD_LAZY | RTLD_GLOBAL);
  if (dl == NULL)
    {
      printf ("Error loading tst-tls19mod1.so: %s\n", dlerror ());
      return 1;
    }

  int (*fn) (void) = dlsym (dl, "foo");
  if (fn == NULL)
    {
      printf("Error obtaining symbol foo\n");
      return 1;
    }

  return fn ();
}

#include <support/test-driver.c>
