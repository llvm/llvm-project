#include <dlfcn.h>

static int
do_test (void)
{
  void *h = dlopen("$ORIGIN/tst-auditmod9b.so", RTLD_LAZY);
  int (*fp)(void) = dlsym(h, "f");
  return fp() - 1;
}

#include <support/test-driver.c>
