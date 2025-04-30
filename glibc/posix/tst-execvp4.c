#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

#ifndef EXECVP
# define EXECVP(file, argv)  execvp (file, argv)
#endif

static int
do_test (void)
{
  char buf[40] = "/usr/bin/does-not-exist";
  size_t stemlen = strlen (buf);
  struct stat64 st;
  int cnt = 0;
  while (stat64 (buf, &st) != -1 || errno != ENOENT
	 || stat64 (buf + 4, &st) != -1 || errno != ENOENT)
    {
      if (cnt++ == 100)
	{
	  puts ("cannot find a unique file name");
	  return 0;
	}

      strcpy (buf + stemlen, ".XXXXXX");
      mktemp (buf);
    }

  unsetenv ("PATH");
  char *argv[] = { buf + 9, NULL };
  EXECVP (argv[0], argv);
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
