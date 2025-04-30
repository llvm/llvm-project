#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <gnu/lib-names.h>
#include <ldsodefs.h>


static int
do_test (void)
{
  int result = 0;

  for (int i = 1; i <= 10; ++i)
    {
      void *h[DL_NNS - 1];
      char used[DL_NNS];

      printf ("round %d\n", i);

      memset (used, '\0', sizeof (used));
      used[LM_ID_BASE] = 1;

      for (int j = 0; j < DL_NNS - 1; ++j)
	{
	  h[j] = dlmopen (LM_ID_NEWLM, "$ORIGIN/tst-dlmopen1mod.so",
			  RTLD_LAZY);
	  if (h[j] == NULL)
	    {
	      printf ("round %d, namespace %d: load failed: %s\n",
		      i, j, dlerror ());
	      return 1;
	    }
	  Lmid_t ns;
	  if (dlinfo (h[j], RTLD_DI_LMID, &ns) != 0)
	    {
	      printf ("round %d, namespace %d: dlinfo failed: %s\n",
		      i, j, dlerror ());
	      return 1;
	    }
	  if (ns < 0 || ns >= DL_NNS)
	    {
	      printf ("round %d, namespace %d: invalid namespace %ld",
		      i, j, (long int) ns);
	      result = 1;
	    }
	  else if (used[ns] != 0)
	    {
	      printf ("\
round %d, namespace %d: duplicate allocate of namespace %ld",
		      i, j, (long int) ns);
	      result = 1;
	    }
	  else
	    used[ns] = 1;
	}

      for (int j = 0; j < DL_NNS - 1; ++j)
	if (dlclose (h[j]) != 0)
	  {
	    printf ("round %d, namespace %d: close failed: %s\n",
		    i, j, dlerror ());
	    return 1;
	  }
    }

  return result;
}

#include <support/test-driver.c>
