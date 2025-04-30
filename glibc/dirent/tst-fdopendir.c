#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <dirent.h>
#include <stdbool.h>
#include <string.h>
#include <sys/stat.h>

#ifndef O_NOATIME
# define O_NOATIME	0
#endif

static int
do_test (void)
{
  char fname[] = "/tmp/jXXXXXX";
  int fd = mkstemp (fname);
  if (fd == -1)
    {
      puts ("mkstemp failed");
      return 1;
    }

  write (fd, "hello", 5);
  close (fd);

  struct stat64 st;
  if (stat64 (fname, &st) == -1)
    {
      puts ("first stat failed");
      return 0;
    }

  /* Make sure there is enough time between the creation and the access.  */
  sleep (2);

  fd = open (fname, O_RDONLY | O_NOATIME);
  if (fd == -1)
    {
      puts ("first open failed");
      return 1;
    }

  char buf[5];
  read(fd, buf, sizeof (buf));
  close(fd);

  struct stat64 st2;
  if (stat64 (fname, &st2) == -1)
    {
      puts ("second stat failed");
      return 0;
    }

  bool no_noatime = false;
#ifdef _STATBUF_ST_NSEC
  if (st.st_atim.tv_sec != st2.st_atim.tv_sec
      || st.st_atim.tv_nsec != st2.st_atim.tv_nsec)
#else
  if (st.st_atime != st2.st_atime)
#endif
    {
      puts ("file atime changed");
      no_noatime = true;
    }

  unlink(fname);

  strcpy(fname, "/tmp/dXXXXXX");
  char *d = mkdtemp (fname);
  if (d == NULL)
    {
      puts ("mkdtemp failed");
      return 1;
    }

  if (stat64 (d, &st) == -1)
    {
      puts ("third stat failed");
      return 0;
    }
  sleep (2);

  fd = open64 (d, O_RDONLY|O_NDELAY|O_DIRECTORY|O_NOATIME);
  if (fd == -1)
    {
      puts ("second open failed");
      return 1;
    }
  DIR *dir = fdopendir (fd);
  if (dir == NULL)
    {
      puts ("fdopendir failed");
      return 1;
    }

  struct dirent *de;
  while ((de = readdir (dir)) != NULL)
    ;

  closedir (dir);

  if (stat64 (d, &st2) == -1)
    {
      puts ("fourth stat failed");
      return 0;
    }
#ifdef _STATBUF_ST_NSEC
  if (!no_noatime
      && (st.st_atim.tv_sec != st2.st_atim.tv_sec
	 || st.st_atim.tv_nsec != st2.st_atim.tv_nsec))
#else
  if (!no_noatime && st.st_atime != st2.st_atime)
#endif
    {
      puts ("directory atime changed");
      return 1;
    }

  rmdir(fname);

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
