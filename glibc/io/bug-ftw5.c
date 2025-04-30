#include <errno.h>
#include <ftw.h>
#include <stdio.h>

static int
fn (const char *file, const struct stat *sb, int flag, struct FTW *s)
{
  puts (file);
  return FTW_STOP;
}

static int
do_test (void)
{
  if (nftw ("/", fn, 0, FTW_CHDIR | FTW_ACTIONRETVAL) < 0)
    {
      printf ("nftw / FTW_CHDIR: %m\n");
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
