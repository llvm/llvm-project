#include <dlfcn.h>
#include <stdio.h>


static int
do_test (void)
{
  void *h = dlmopen (LM_ID_NEWLM, "$ORIGIN/tst-dlmopen1mod.so", RTLD_LAZY);
  if (h == NULL)
    {
      printf ("cannot get handle for %s: %s\n",
	      "tst-dlmopen1mod.so", dlerror ());
      return 1;
    }

  /* Do not unload.  */

  return 0;
}

#include <support/test-driver.c>
