#include <dlfcn.h>
#include <stdio.h>


int call_puts;

static int
do_test (void)
{
  call_puts = 1;

  void *h1 = dlopen ("$ORIGIN/order2mod1.so", RTLD_LAZY | RTLD_GLOBAL);
  if (h1 == NULL)
    {
      puts ("cannot load order2mod1");
      return 1;
    }
  void *h2 = dlopen ("$ORIGIN/order2mod2.so", RTLD_LAZY);
  if (h2 == NULL)
    {
      puts ("cannot load order2mod2");
      return 1;
    }
  if (dlclose (h1) != 0)
    {
      puts ("dlclose order2mod1 failed");
      return 1;
    }
  if (dlclose (h2) != 0)
    {
      puts ("dlclose order2mod2 failed");
      return 1;
    }
  return 0;
}

#include <support/test-driver.c>

static void
__attribute__ ((destructor))
fini (void)
{
  if (call_puts)
    puts ("5");
}
