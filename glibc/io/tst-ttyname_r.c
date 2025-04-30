#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

static void do_prepare (void);
#define PREPARE(argc, argv) do_prepare ()
static int do_test (void);
#define TEST_FUNCTION do_test ()
#include <test-skeleton.c>

static int temp_fd;

static void
do_prepare (void)
{
  char *temp_file;
  temp_fd = create_temp_file ("tst-ttyname_r.", &temp_file);
  if (temp_fd == -1)
    error (1, errno, "cannot create temporary file");
}

static int
do_test (void)
{
  int ret = 0;
  char buf[sysconf (_SC_TTY_NAME_MAX) + 1];
  int res = ttyname_r (-1, buf, sizeof (buf));
  if (res != EBADF)
    {
      printf ("1st ttyname_r returned with res %d\n", res);
      ret++;
    }
  res = ttyname_r (temp_fd, buf, sizeof (buf));
  if (res != ENOTTY)
    {
      printf ("2nd ttyname_r returned with res %d\n", res);
      ret++;
    }
  return ret;
}
