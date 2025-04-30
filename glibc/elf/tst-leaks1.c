#include <stdio.h>
#include <dlfcn.h>
#include <mcheck.h>
#include <stdlib.h>

static int
do_test (void)
{
  void *h;
  int ret = 0;
  /* Carry out *one* failing call to dlopen before starting mtrace to
     force any one-time inintialization that may happen to the
     executable link map e.g. expansion and caching of $ORIGIN.  */
  h = dlopen ("$ORIGIN/tst-leaks1.o", RTLD_LAZY);
  if (h != NULL)
    {
      puts ("dlopen unexpectedly succeeded");
      ret = 1;
      dlclose (h);
    }

  /* Start tracing and run each test 5 times to see if there are any
     leaks in the failing dlopen.  */
  mtrace ();

  for (int i = 0; i < 10; i++)
    {
      h = dlopen (i < 5
		  ? "./tst-leaks1.c"
		  : "$ORIGIN/tst-leaks1.o", RTLD_LAZY);
      if (h != NULL)
	{
	  puts ("dlopen unexpectedly succeeded");
	  ret = 1;
	  dlclose (h);
	}
    }

  return ret;
}

#include <support/test-driver.c>
