#include "../dlfcn/dlfcn.h"
#include <stdio.h>
#include <stdlib.h>
#include <gnu/lib-names.h>

static int
do_test (void)
{
  int result = 0;

  printf ("\nOpening pthread library.\n");
  void *pthread = dlopen (LIBPTHREAD_SO, RTLD_LAZY);

  /* This is a test for correct DF_1_NODELETE clearing when dlopen failure
     happens.  We should clear DF_1_NODELETE for failed library only, because
     doing this for others (e.g. libpthread) might cause them to be unloaded,
     that may lead to some global references (e.g. __rtld_lock_unlock) to be
     broken.  The dlopen should fail because of undefined symbols in shared
     library, that cause DF_1_NODELETE to be cleared.  For libpthread, this
     flag should be set, because if not, SIGSEGV will happen in dlclose.  */
  if (dlopen ("tst-nodelete2mod.so", RTLD_NOW) != NULL)
    {
      printf ("Unique symbols test failed\n");
      result = 1;
    }

  if (pthread)
    dlclose (pthread);

  if (result == 0)
    printf ("SUCCESS\n");

  return result;
}

#include <support/test-driver.c>
