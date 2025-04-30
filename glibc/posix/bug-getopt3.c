/* BZ 11040 */
#include <getopt.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

static const struct option opts[] =
  {
    { "alpha",	no_argument,       NULL, 'a' },
    { "beta",	required_argument, NULL, 'b' },
    { NULL,	0,                 NULL, 0 }
  };

static int
one_test (const char *fmt, int argc, char *argv[], int n, int expected[n],
	  int out[n])
{
  optind = 1;

  int res = 0;
  for (int i = 0; i < n; ++i)
    {
      rewind (stderr);
      if (ftruncate (fileno (stderr), 0) != 0)
	{
	  puts ("cannot truncate file");
	  return 1;
	}

      int c = getopt_long (argc, argv, fmt, opts, NULL);
      if (c != expected[i])
	{
	  printf ("format '%s' test %d failed: expected '%c', got '%c'\n",
		  fmt, i, expected[i], c);
	  res = 1;
	}
      if ((ftell (stderr) != 0) != out[i])
	{
	  printf ("format '%s' test %d failed: %sprinted to stderr\n",
		  fmt, i, out[i] ? "not " : "");
	  res = 1;
	}
    }

  return res;
}


static int
do_test (void)
{
  char fname[] = "/tmp/bug-getopt3.XXXXXX";
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

  int ret = one_test ("ab:W;", 2,
		      (char *[2]) { (char *) "bug-getopt3", (char *) "-a;" },
		      2, (int [2]) { 'a', '?' }, (int [2]) { 0, 1 });

  ret |= one_test ("ab:W;", 2,
		   (char *[2]) { (char *) "bug-getopt3", (char *) "-a:" }, 2,
		   (int [2]) { 'a', '?' }, (int [2]) { 0, 1 });

  if (ret == 0)
    puts ("all OK");

  return ret;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
