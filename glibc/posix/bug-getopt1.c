/* BZ 11039 */
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

static int
one_test (const char *fmt, int argc, char *argv[], int expected[argc - 1])
{
  optind = 1;

  int res = 0;
  for (int i = 0; i < argc - 1; ++i)
    {
      rewind (stderr);
      if (ftruncate (fileno (stderr), 0) != 0)
	{
	  puts ("cannot truncate file");
	  return 1;
	}

      int c = getopt (argc, argv, fmt);
      if (c != expected[i])
	{
	  printf ("format '%s' test %d failed: expected '%c', got '%c'\n",
		  fmt, i, expected[i], c);
	  res = 1;
	}
      if (ftell (stderr) != 0)
	{
	  printf ("format '%s' test %d failed: printed to stderr\n",
		  fmt, i);
	  res = 1;
	}
    }

  return res;
}


static int
do_test (void)
{
  char fname[] = "/tmp/bug-getopt1.XXXXXX";
  int fd = mkstemp (fname);
  if (fd == -1)
    {
      printf ("mkstemp failed: %m\n");
      return 1;
    }
  close (fd);

  if (freopen (fname, "w+", stderr) == NULL)
    {
      puts ("cannot redirect stderr");
      return 1;
    }

  remove (fname);

  int ret = one_test ("+:a:b", 2,
		      (char *[2]) { (char *) "bug-getopt1", (char *) "-a" },
		      (int [1]) { ':' });

  ret |= one_test ("+:a:b", 3,
		   (char *[3]) { (char *) "bug-getopt1", (char *) "-b",
				 (char *) "-a" },
		   (int [2]) { 'b', ':' });

  if (ret == 0)
    puts ("all OK");

  return ret;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
