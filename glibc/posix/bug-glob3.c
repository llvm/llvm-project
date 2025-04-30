#include <glob.h>
#include <stdio.h>
#include <string.h>

static int
do_test (void)
{
  int result = 0;
  glob_t g;
  g.gl_pathc = 0;

  int r = glob ("", 0, NULL, &g);
  if (r != GLOB_NOMATCH)
    {
      puts ("glob (\"\", 0, NULL, &g) did not fail");
      result = 1;
    }
  else if (g.gl_pathc != 0)
    {
      puts ("gl_pathc after glob (\"\", 0, NULL, &g) not zero");
      result = 1;
    }

  r = glob ("", GLOB_NOCHECK, NULL, &g);
  if (r != 0)
    {
      puts ("glob (\"\", GLOB_NOCHECK, NULL, &g) did fail");
      result = 1;
    }
  else if (g.gl_pathc != 1)
    {
      puts ("gl_pathc after glob (\"\", GLOB_NOCHECK, NULL, &g) not 1");
      result = 1;
    }
  else if (strcmp (g.gl_pathv[0], "") != 0)
    {
      puts ("gl_pathv[0] after glob (\"\", GLOB_NOCHECK, NULL, &g) not \"\"");
      result = 1;
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
