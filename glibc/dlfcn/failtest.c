#include <dlfcn.h>
#include <stdio.h>


/* Number of rounds we perform the test.  */
#define TEST_ROUNDS	10


static const char unknown[] = "a-file-with-this-name-does-not-exist";
static const char exists[] = "failtestmod.so";


int
main (void)
{
  int i;

  setvbuf (stdout, NULL, _IONBF, 0);

  for (i = 0; i < TEST_ROUNDS; ++i)
    {
      void *dsc;

      printf ("Round %d: Try loading \"%s\"\n", i, unknown);

      dsc = dlopen (unknown, RTLD_NOW);
      if (dsc != NULL)
	{
	  printf ("We found a file of name \"%s\": this should not happen\n",
		  unknown);
	  return 1;
	}

      printf ("Round %d: loading \"%s\" failed\n", i, unknown);

      /* Don't use `dlerror', just load an existing file.  */
      dsc = dlopen (exists, RTLD_NOW);
      if (dsc == NULL)
	{
	  printf ("Could not load \"%s\": %s\n", exists, dlerror ());
	  return 1;
	}

      printf ("Round %d: Loaded \"%s\"\n", i, exists);

      dlclose (dsc);

      printf ("Round %d: Unloaded \"%s\"\n", i, exists);
    }

  return 0;
}


extern void foo (void);

void
foo (void)
{
}
