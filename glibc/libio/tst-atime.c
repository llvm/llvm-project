#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/statvfs.h>


static int do_test (void);
#define TEST_FUNCTION do_test ()
#include <test-skeleton.c>


static int
do_test (void)
{
  char *buf;
  int fd;
  FILE *fp;
  int ch;
  struct stat st1;
  struct stat st2;

  buf = (char *) malloc (strlen (test_dir) + sizeof "/tst-atime.XXXXXX");
  if (buf == NULL)
    {
      printf ("cannot allocate memory: %m\n");
      return 1;
    }
  stpcpy (stpcpy (buf, test_dir), "/tst-atime.XXXXXX");

  fd = mkstemp (buf);
  if (fd == -1)
    {
      printf ("cannot open temporary file: %m\n");
      return 1;
    }

#ifdef ST_NOATIME
  /* Make sure the filesystem doesn't have the noatime option set.  If
     statvfs is not available just continue.  */
  struct statvfs sv;
  int e = fstatvfs (fd, &sv);
  if (e != ENOSYS)
    {
      if (e != 0)
	{
	  printf ("cannot statvfs '%s': %m\n", buf);
	  return 1;
	}

      if ((sv.f_flag & ST_NOATIME) != 0)
	{
	  puts ("Bah!  The filesystem is mounted with noatime");
	  return 0;
	}
    }
#endif

  /* Make sure it gets removed.  */
  add_temp_file (buf);

  if (write (fd, "some string\n", 12) != 12)
    {
      printf ("cannot write temporary file: %m\n");
      return 1;
    }

  if (lseek (fd, 0, SEEK_SET) == (off_t) -1)
    {
      printf ("cannot reposition temporary file: %m\n");
      return 1;
    }

  fp = fdopen (fd, "r");
  if (fp == NULL)
    {
      printf ("cannot create stream: %m\n");
      return 1;
    }

  if (fstat (fd, &st1) == -1)
    {
      printf ("first stat failed: %m\n");
      return 1;
    }

  sleep (2);

  ch = fgetc (fp);
  if (ch != 's')
    {
      printf ("did not read correct character: got '%c', expected 's'\n", ch);
      return 1;
    }

  if (fstat (fd, &st2) == -1)
    {
      printf ("second stat failed: %m\n");
      return 1;
    }

  if (st1.st_atime > st2.st_atime)
    {
      puts ("second atime smaller");
      return 1;
    }
  else if (st1.st_atime == st2.st_atime)
    {
      puts ("atime has not changed");
      return 1;
    }

  fclose (fp);

  return 0;
}
