#include <errno.h>
#include <ftw.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int cb_called;

static int
cb (const char *fname, const struct stat *st, int flag)
{
  printf ("%s %d\n", fname, flag);
  cb_called = 1;
  return 0;
}

int
main (void)
{
  char tmp[] = "/tmp/ftwXXXXXX";
  char tmp2[] = "/tmp/ftwXXXXXX/ftwXXXXXX";
  char *dname;
  char *dname2;
  int r;
  int e;

  if (getuid () == 0)
    {
      puts ("this test needs to be run by ordinary user");
      exit (0);
    }

  dname = mkdtemp (tmp);
  if (dname == NULL)
    {
      printf ("mkdtemp: %m\n");
      exit (1);
    }

  memcpy (tmp2, tmp, strlen (tmp));
  dname2 = mkdtemp (tmp2);
  if (dname2 == NULL)
    {
      printf ("mkdtemp: %m\n");
      rmdir (dname);
      exit (1);
    }

  if (chmod (dname, S_IWUSR|S_IWGRP|S_IWOTH) != 0)
    {
      printf ("chmod: %m\n");
      rmdir (dname);
      exit (1);
    }

  r = ftw (dname2, cb, 10);
  e = errno;
  printf ("r = %d", r);
  if (r != 0)
    printf (", errno = %d", errno);
  puts ("");

  chmod (dname, S_IRWXU|S_IRWXG|S_IRWXO);
  rmdir (dname2);
  rmdir (dname);

  return (r != -1 && e == EACCES) || cb_called;
}
