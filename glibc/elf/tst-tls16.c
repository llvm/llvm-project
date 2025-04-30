#include <dlfcn.h>
#include <stdlib.h>
#include <stdio.h>

static int
do_test (void)
{
  void *h = dlopen ("tst-tlsmod16a.so", RTLD_LAZY | RTLD_GLOBAL);
  if (h == NULL)
    {
      puts ("unexpectedly failed to open tst-tlsmod16a.so");
      exit (1);
    }

  void *p = dlsym (h, "tlsvar");

  /* This dlopen should indeed fail, because tlsvar was assigned to
     dynamic TLS, and the new module requests it to be in static TLS.
     However, there's a possibility that dlopen succeeds if the
     variable is, for whatever reason, assigned to static TLS, or if
     the module fails to require static TLS, or even if TLS is not
     supported.  */
  h = dlopen ("tst-tlsmod16b.so", RTLD_NOW | RTLD_GLOBAL);
  if (h == NULL)
    {
      return 0;
    }

  puts ("unexpectedly succeeded to open tst-tlsmod16b.so");


  void *(*fp) (void) = (void *(*) (void)) dlsym (h, "in_dso");
  if (fp == NULL)
    {
      puts ("cannot find in_dso");
      exit (1);
    }

  /* If the dlopen passes, at least make sure the address returned by
     dlsym is the same as that returned by the initial-exec access.
     If the variable was assigned to dynamic TLS during dlsym, this
     portion will fail.  */
  if (fp () != p)
    {
      puts ("returned values do not match");
      exit (1);
    }

  return 0;
}

#include <support/test-driver.c>
