#include <errno.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


static int
do_test (void)
{
  char tmpl[] = "/tmp/tst-fdopendir2-XXXXXX";
  int fd = mkstemp (tmpl);
  if (fd == -1)
    {
      puts ("cannot open temp file");
      return 1;
    }

  errno = 0;
  DIR *d = fdopendir (fd);

  int e = errno;

  close (fd);
  unlink (tmpl);

  if (d != NULL)
    {
      puts ("fdopendir with normal file descriptor did not fail");
      return 1;
    }
  if (e != ENOTDIR)
    {
      printf ("fdopendir set errno to %d, not %d as expected\n", e, ENOTDIR);
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
