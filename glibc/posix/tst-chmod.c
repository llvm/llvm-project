/* Test for chmod functions.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 2000.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <dirent.h>
#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <mcheck.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>


#define OUT_OF_MEMORY \
  do {									      \
    puts ("cannot allocate memory");					      \
    result = 1;								      \
    goto fail;								      \
  } while (0)

static int
do_test (int argc, char *argv[])
{
  const char *builddir;
  struct stat64 st1;
  struct stat64 st2;
  char *buf;
  char *testdir;
  char *testfile = NULL;
  char *startdir;
  size_t buflen;
  int fd;
  int result = 0;
  DIR *dir;

  mtrace ();

  if (argc <= 1)
    error (EXIT_FAILURE, 0, "no parameters");

  /* This is where we will create the test files.  */
  builddir = argv[1];
  buflen = strlen (builddir) + 50;

  startdir = getcwd (NULL, 0);
  if (startdir == NULL)
    {
      printf ("cannot get current directory: %m\n");
      exit (EXIT_FAILURE);
    }

  /* A buffer large enough for everything we need.  */
  buf = (char *) alloca (buflen);

  /* Create the directory name.  */
  snprintf (buf, buflen, "%s/chmoddirXXXXXX", builddir);

  if (mkdtemp (buf) == NULL)
    {
      printf ("cannot create test directory: %m\n");
      exit (EXIT_FAILURE);
    }

  if (chmod ("", 0600) == 0)
    {
      puts ("chmod(\"\", 0600 didn't fail");
      result = 1;
    }
  else if (errno != ENOENT)
    {
      puts ("chmod(\"\",0600) does not set errno to ENOENT");
      result = 1;
    }

  /* Create a duplicate.  */
  testdir = strdup (buf);
  if (testdir == NULL)
    OUT_OF_MEMORY;

  if (stat64 (testdir, &st1) != 0)
    {
      printf ("cannot stat test directory: %m\n");
      exit (1);
    }
  if (!S_ISDIR (st1.st_mode))
    {
      printf ("file not created as directory: %m\n");
      exit (1);
    }

  /* We have to wait for a second to make sure the ctime changes.  */
  sleep (1);

  /* Remove all access rights from the directory.  */
  if (chmod (testdir, 0) != 0)
    {
      printf ("cannot change mode of test directory: %m\n");
      result = 1;
      goto fail;
    }

  if (stat64 (testdir, &st2) != 0)
    {
      printf ("cannot stat test directory: %m\n");
      result = 1;
      goto fail;
    }

  /* Compare result.  */
  if ((st2.st_mode & ALLPERMS) != 0)
    {
      printf ("chmod(...,0) on directory left bits nonzero: %o\n",
	      st2.st_mode & ALLPERMS);
      result = 1;
    }
  if (st1.st_ctime >= st2.st_ctime)
    {
      puts ("chmod(...,0) did not set ctime correctly");
      result = 1;
    }

  /* Name of a file in the directory.  */
  snprintf (buf, buflen, "%s/file", testdir);
  testfile = strdup (buf);
  if (testfile == NULL)
    OUT_OF_MEMORY;

  fd = creat (testfile, 0);
  if (fd != -1)
    {
      if (getuid () != 0)
	{
	  puts ("managed to create test file in protected directory");
	  result = 1;
	}
      close (fd);
    }
  else if (errno != EACCES)
    {
      puts ("creat didn't generate correct errno value");
      result = 1;
    }

  /* With this mode it still shouldn't be possible to create a file.  */
  if (chmod (testdir, 0600) != 0)
    {
      printf ("cannot change mode of test directory to 0600: %m\n");
      result = 1;
      goto fail;
    }

  fd = creat (testfile, 0);
  if (fd != -1)
    {
      if (getuid () != 0)
	{
	  puts ("managed to create test file in no-x protected directory");
	  result = 1;
	}
      close (fd);
    }
  else if (errno != EACCES)
    {
      puts ("creat didn't generate correct errno value");
      result = 1;
    }

  /* Change the directory mode back to allow creating a file.  This
     time with fchmod.  */
  dir = opendir (testdir);
  if (dir != NULL)
    {
      if (fchmod (dirfd (dir), 0700) != 0)
	{
	  printf ("cannot change mode of test directory to 0700: %m\n");
	  result = 1;
	  closedir (dir);
	  goto fail;
	}

      closedir (dir);
    }
  else
    {
      printf ("cannot open directory: %m\n");
      result = 1;

      if (chmod (testdir, 0700) != 0)
	{
	  printf ("cannot change mode of test directory to 0700: %m\n");
	  goto fail;
	}
    }

  fd = creat (testfile, 0);
  if (fd == -1)
    {
      puts ("still didn't manage to create test file in protected directory");
      result = 1;
      goto fail;
    }
  if (fstat64 (fd, &st1) != 0)
    {
      printf ("cannot stat new file: %m\n");
      result = 1;
    }
  else if ((st1.st_mode & ALLPERMS) != 0)
    {
      puts ("file not created with access mode 0");
      result = 1;
    }
  close (fd);

  snprintf (buf, buflen, "%s/..", testdir);
  chdir (buf);
  /* We are now in the directory above the one we create the test
     directory in.  */

  sleep (1);
  snprintf (buf, buflen, "./%s/../%s/file",
	    basename (testdir), basename (testdir));
  if (chmod (buf, 0600) != 0)
    {
      printf ("cannot change mode of file to 0600: %m\n");
      result = 1;
      goto fail;
    }
  snprintf (buf, buflen, "./%s//file", basename (testdir));
  if (stat64 (buf, &st2) != 0)
    {
      printf ("cannot stat new file: %m\n");
      result = 1;
    }
  else if ((st2.st_mode & ALLPERMS) != 0600)
    {
      puts ("file mode not changed to 0600");
      result = 1;
    }
  else if (st1.st_ctime >= st2.st_ctime)
    {
      puts ("chmod(\".../file\",0600) did not set ctime correctly");
      result = 1;
    }

  if (chmod (buf, 0777 | S_ISUID | S_ISGID) != 0)
    {
      printf ("cannot change mode of file to %o: %m\n",
	      0777 | S_ISUID | S_ISGID);
      result = 1;
    }
  if (stat64 (buf, &st2) != 0)
    {
      printf ("cannot stat test file: %m\n");
      result = 1;
    }
  else if ((st2.st_mode & ALLPERMS) != (0777 | S_ISUID | S_ISGID))
    {
      puts ("file mode not changed to 0777 | S_ISUID | S_ISGID");
      result = 1;
    }

  if (chmod (basename (testdir), 0777 | S_ISUID | S_ISGID | S_ISVTX) != 0)
    {
      printf ("cannot change mode of test directory to %o: %m\n",
	      0777 | S_ISUID | S_ISGID | S_ISVTX);
      result = 1;
    }
  if (stat64 (basename (testdir), &st2) != 0)
    {
      printf ("cannot stat test directory: %m\n");
      result = 1;
    }
  else if ((st2.st_mode & ALLPERMS) != (0777 | S_ISUID | S_ISGID | S_ISVTX))
    {
      puts ("directory mode not changed to 0777 | S_ISUID | S_ISGID | S_ISGID");
      result = 1;
    }

  snprintf (buf, buflen, "./%s/no-such-file", basename (testdir));
  if (chmod (buf, 0600) != -1)
    {
      puts ("chmod(\".../no-such-file\",0600) did not fail");
      result = 1;
    }
  else if (errno != ENOENT)
    {
      puts ("chmod(\".../no-such-file\",0600) does not set errno to ENOENT");
      result = 1;
    }

  snprintf (buf, buflen, "%s/", basename (testdir));
  if (chmod (basename (testdir), 0677) != 0)
    {
      printf ("cannot change mode of test directory to 0677: %m\n");
      result = 1;
    }
  else
    {
      snprintf (buf, buflen, "./%s/file", basename (testdir));
      if (chmod (buf, 0600) == 0)
	{
	  if (getuid () != 0)
	    {
	      puts ("chmod(\".../file\") with no-exec directory succeeded");
	      result = 1;
	    }
	}
      else if (errno != EACCES)
	{
	  puts ("chmod(\".../file\") with no-exec directory didn't set EACCES");
	  result = 1;
	}
    }

  if (chmod (basename (testdir), 0777) != 0)
    {
      printf ("cannot change mode of test directory to 0777: %m\n");
      result = 1;
      goto fail;
    }

  snprintf (buf, buflen, "%s/file/cannot-be", basename (testdir));
  if (chmod (buf, 0600) == 0)
    {
      puts ("chmod(\".../file/cannot-be\",0600) did not fail");
      result = 1;
    }
  else if (errno != ENOTDIR)
    {
      puts ("chmod(\".../file/cannot-be\",0600) does not set errno to ENOTDIR");
      result = 1;
    }

 fail:
  chdir (startdir);

  /* Remove all the files.  */
  chmod (testdir, 0700);
  if (testfile != NULL)
    {
      chmod (testfile, 0700);
      unlink (testfile);
    }
  rmdir (testdir);

  /* Free the resources.  */
  free (testfile);
  free (testdir);
  free (startdir);

  return result;
}

#include "../test-skeleton.c"
