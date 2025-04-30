/* BZ 11041 */
#include <getopt.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

static const struct option opts[] =
  {
    { "alpha",    optional_argument, NULL, 'a' },
    { NULL,       0,                 NULL, 0 }
  };

static int
one_test (const char *fmt, int argc, char *argv[], int n, int expected[n])
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
	  printf ("%s: format '%s' test %d failed: expected '%c', got '%c'\n",
		  argv[0], fmt, i, expected[i], c);
	  res = 1;
	}
      else if (optarg != NULL)
	{
	  printf ("%s: format '%s' test %d failed: optarg is \"%s\", not NULL\n",
		  argv[0], fmt, i, optarg);
	  res = 1;
	}
      if (ftell (stderr) != 0)
	{
	  printf ("%s: format '%s' test %d failed: printed to stderr\n",
		  argv[0], fmt, i);
	  res = 1;
	}
    }

  return res;
}


static int
do_test (void)
{
  char fname[] = "/tmp/bug-getopt4.XXXXXX";
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

  int ret = one_test ("W;", 2,
		      (char *[2]) { (char *) "bug-getopt4a", (char *) "--a" },
		      1, (int [1]) { 'a' });

  ret |= one_test ("W;", 3,
		   (char *[3]) { (char *) "bug-getopt4b", (char *) "-W",
				 (char *) "a" },
		   1, (int [1]) { 'a' });

  if (ret == 0)
    puts ("all OK");

  return ret;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
