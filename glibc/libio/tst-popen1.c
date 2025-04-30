#include <fcntl.h>
#include <stdio.h>

static int
do_test (void)
{
  int res = 0;

  FILE *fp = popen ("echo hello", "r");
  if (fp == NULL)
    {
      puts ("first popen failed");
      res = 1;
    }
  else
    {
      int fd = fileno (fp);
      if (fcntl (fd, F_GETFD) == FD_CLOEXEC)
	{
	  puts ("first popen(\"r\") set FD_CLOEXEC");
	  res = 1;
	}

      pclose (fp);
    }

  fp = popen ("echo hello", "re");
  if (fp == NULL)
    {
      puts ("second popen failed");
      res = 1;
    }
  else
    {
      int fd = fileno (fp);
      if (fcntl (fd, F_GETFD) != FD_CLOEXEC)
	{
	  puts ("second popen(\"r\") did not set FD_CLOEXEC");
	  res = 1;
	}

      pclose (fp);
    }

  return res;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
