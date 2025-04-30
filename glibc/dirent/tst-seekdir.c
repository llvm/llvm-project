#include <stdio.h>
#include <dirent.h>
#include <stdlib.h>

static int
do_test (void)
{
  DIR * dirp;
  long int save3 = 0;
  long int cur;
  int i = 0;
  int result = 0;
  struct dirent *dp;
  long int save0;
  long int rewind;

  dirp = opendir (".");
  if (dirp == NULL)
    {
      printf ("opendir failed: %m\n");
      return 1;
    }

  save0 = telldir (dirp);
  if (save0 == -1)
    {
      printf ("telldir failed: %m\n");
      result = 1;
    }

  for (dp = readdir (dirp); dp != NULL; dp = readdir (dirp))
    {
      /* save position 3 (after fourth entry) */
      if (i++ == 3)
	save3 = telldir (dirp);

      printf ("%s\n", dp->d_name);

      /* stop at 400 (just to make sure dirp->__offset and dirp->__size are
	 scrambled */
      if (i == 400)
	break;
    }

  printf ("going back past 4-th entry...\n");

  /* go back to saved entry */
  seekdir (dirp, save3);

  /* Check whether telldir equals to save3 now.  */
  cur = telldir (dirp);
  if (cur != save3)
    {
      printf ("seekdir (d, %ld); telldir (d) == %ld\n", save3, cur);
      result = 1;
    }

  /* print remaining files (3-last) */
  for (dp = readdir (dirp); dp != NULL; dp = readdir (dirp))
    printf ("%s\n", dp->d_name);

  /* Check rewinddir */
  rewinddir (dirp);
  rewind = telldir (dirp);
  if (rewind == -1)
    {
      printf ("telldir failed: %m\n");
      result = 1;
    }
  else if (save0 != rewind)
    {
      printf ("rewinddir didn't reset directory stream\n");
      result = 1;
    }

  closedir (dirp);
  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
