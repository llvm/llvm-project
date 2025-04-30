/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2000.

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
#include <fcntl.h>
#include <mcheck.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <libc-diag.h>

/* We expect four arguments:
   - source directory name
   - object directory
   - common object directory
   - the program name with path
*/
int
main (int argc, char *argv[])
{
  const char *srcdir;
  const char *objdir;
  const char *common_objdir;
  const char *progpath;
  struct stat64 st1;
  struct stat64 st2;
  struct stat64 st3;
  DIR *dir1;
  DIR *dir2;
  int result = 0;
  struct dirent64 *d;
  union
    {
      struct dirent64 d;
      char room [offsetof (struct dirent64, d_name[0]) + NAME_MAX + 1];
    }
    direntbuf;
  char *objdir_copy1;
  char *objdir_copy2;
  char *buf;
  int fd;

  mtrace ();

  if (argc < 5)
    {
      puts ("not enough parameters");
      exit (1);
    }

  /* Make parameters available with nicer names.  */
  srcdir = argv[1];
  objdir = argv[2];
  common_objdir = argv[3];
  progpath = argv[4];

  /* First test the current source dir.  We cannot really compare the
     result of `getpwd' with the srcdir string but we have other means.  */
  if (stat64 (".", &st1) < 0)
    {
      printf ("cannot stat starting directory: %m\n");
      exit (1);
    }

  if (chdir (srcdir) < 0)
    {
      printf ("cannot change to source directory: %m\n");
      exit (1);
    }
  if (stat64 (".", &st2) < 0)
    {
      printf ("cannot stat source directory: %m\n");
      exit (1);
    }

  /* The two last stat64 calls better were for the same directory.  */
  if (st1.st_dev != st2.st_dev || st1.st_ino != st2.st_ino)
    {
      printf ("stat of source directory failed: (%lld,%lld) vs (%lld,%lld)\n",
	      (long long int) st1.st_dev, (long long int) st1.st_ino,
	      (long long int) st2.st_dev, (long long int) st2.st_ino);
      exit (1);
    }

  /* Change to the object directory.  */
  if (chdir (objdir) < 0)
    {
      printf ("cannot change to object directory: %m\n");
      exit (1);
    }
  if (stat64 (".", &st1) < 0)
    {
      printf ("cannot stat object directory: %m\n");
      exit (1);
    }
  /* Is this the same we get as with the full path?  */
  if (stat64 (objdir, &st2) < 0)
    {
      printf ("cannot stat object directory with full path: %m\n");
      exit (1);
    }
  if (st1.st_dev != st2.st_dev || st1.st_ino != st2.st_ino)
    {
      printf ("stat of object directory failed: (%lld,%lld) vs (%lld,%lld)\n",
	      (long long int) st1.st_dev, (long long int) st1.st_ino,
	      (long long int) st2.st_dev, (long long int) st2.st_ino);
      exit (1);
    }

  objdir_copy1 = getcwd (NULL, 0);
  if (objdir_copy1 == NULL)
    {
      printf ("cannot get current directory name for object directory: %m\n");
      result = 1;
    }

  /* First test: this directory must include our program.  */
  if (stat64 (progpath, &st2) < 0)
    {
      printf ("cannot stat program: %m\n");
      exit (1);
    }

  dir1 = opendir (".");
  if (dir1 == NULL)
    {
      printf ("cannot open object directory: %m\n");
      exit (1);
    }

  while ((d = readdir64 (dir1)) != NULL)
    {
      if (d->d_type != DT_UNKNOWN && d->d_type != DT_REG)
	continue;

      if (d->d_ino == st2.st_ino)
	{
	  /* Might be it.  Test the device.  We could use the st_dev
	     element from st1 but what the heck, do more testing.  */
	  if (stat64 (d->d_name, &st3) < 0)
	    {
	      printf ("cannot stat entry from readdir: %m\n");
	      result = 1;
	      d = NULL;
	      break;
	    }

	  if (st3.st_dev == st2.st_dev)
	    break;
	}
    }

  if (d == NULL)
    {
      puts ("haven't found program in object directory");
      result = 1;
    }

  /* We leave dir1 open.  */

  /* Stat using file descriptor.  */
  if (fstat64 (dirfd (dir1), &st2) < 0)
    {
      printf ("cannot fstat object directory: %m\n");
      result = 1;
    }
  if (st1.st_dev != st2.st_dev || st1.st_ino != st2.st_ino)
    {
      printf ("fstat of object directory failed: (%lld,%lld) vs (%lld,%lld)\n",
	      (long long int) st1.st_dev, (long long int) st1.st_ino,
	      (long long int) st2.st_dev, (long long int) st2.st_ino);
      exit (1);
    }

  if (chdir ("..") < 0)
    {
      printf ("cannot go to common object directory with \"..\": %m\n");
      exit (1);
    }

  if (stat64 (".", &st1) < 0)
    {
      printf ("cannot stat common object directory: %m\n");
      exit (1);
    }
  /* Is this the same we get as with the full path?  */
  if (stat64 (common_objdir, &st2) < 0)
    {
      printf ("cannot stat common object directory with full path: %m\n");
      exit (1);
    }
  if (st1.st_dev != st2.st_dev || st1.st_ino != st2.st_ino)
    {
      printf ("stat of object directory failed: (%lld,%lld) vs (%lld,%lld)\n",
	      (long long int) st1.st_dev, (long long int) st1.st_ino,
	      (long long int) st2.st_dev, (long long int) st2.st_ino);
      exit (1);
    }

  /* Stat using file descriptor.  */
  if (fstat64 (dirfd (dir1), &st2) < 0)
    {
      printf ("cannot fstat object directory: %m\n");
      result = 1;
    }

  dir2 = opendir (common_objdir);
  if (dir2 == NULL)
    {
      printf ("cannot open common object directory: %m\n");
      exit (1);
    }

  while ((d = readdir64 (dir2)) != NULL)
    {
      if (d->d_type != DT_UNKNOWN && d->d_type != DT_DIR)
	continue;

      if (d->d_ino == st2.st_ino)
	{
	  /* Might be it.  Test the device.  We could use the st_dev
	     element from st1 but what the heck, do more testing.  */
	  if (stat64 (d->d_name, &st3) < 0)
	    {
	      printf ("cannot stat entry from readdir: %m\n");
	      result = 1;
	      d = NULL;
	      break;
	    }

	  if (st3.st_dev == st2.st_dev)
	    break;
	}
    }

  /* This better should be the object directory again.  */
  if (fchdir (dirfd (dir1)) < 0)
    {
      printf ("cannot fchdir to object directory: %m\n");
      exit (1);
    }

  objdir_copy2 = getcwd (NULL, 0);
  if (objdir_copy2 == NULL)
    {
      printf ("cannot get current directory name for object directory: %m\n");
      result = 1;
    }
  if (strcmp (objdir_copy1, objdir_copy2) != 0)
    {
      puts ("getcwd returned a different string the second time");
      result = 1;
    }

  /* This better should be the common object directory again.  */
  if (fchdir (dirfd (dir2)) < 0)
    {
      printf ("cannot fchdir to common object directory: %m\n");
      exit (1);
    }

  if (stat64 (".", &st2) < 0)
    {
      printf ("cannot stat common object directory: %m\n");
      exit (1);
    }
  if (st1.st_dev != st2.st_dev || st1.st_ino != st2.st_ino)
    {
      printf ("stat of object directory failed: (%lld,%lld) vs (%lld,%lld)\n",
	      (long long int) st1.st_dev, (long long int) st1.st_ino,
	      (long long int) st2.st_dev, (long long int) st2.st_ino);
      exit (1);
    }

  buf = (char *) malloc (strlen (objdir_copy1) + 1 + sizeof "tst-dir.XXXXXX");
  if (buf == NULL)
    {
      printf ("cannot allocate buffer: %m");
      exit (1);
    }

  stpcpy (stpcpy (stpcpy (buf, objdir_copy1), "/"), "tst-dir.XXXXXX");
  if (mkdtemp (buf) == NULL)
    {
      printf ("cannot create test directory in object directory: %m\n");
      exit (1);
    }
  if (stat64 (buf, &st1) < 0)
    {
      printf ("cannot stat new directory \"%s\": %m\n", buf);
      exit (1);
    }
  if (chmod (buf, 0700) < 0)
    {
      printf ("cannot change mode of new directory: %m\n");
      exit (1);
    }

  /* The test below covers the deprecated readdir64_r function.  */
  DIAG_PUSH_NEEDS_COMMENT;
  DIAG_IGNORE_NEEDS_COMMENT (4.9, "-Wdeprecated-declarations");

  /* Try to find the new directory.  */
  rewinddir (dir1);
  while (readdir64_r (dir1, &direntbuf.d, &d) == 0 && d != NULL)
    {
      if (d->d_type != DT_UNKNOWN && d->d_type != DT_DIR)
	continue;

      if (d->d_ino == st1.st_ino)
	{
	  /* Might be it.  Test the device.  We could use the st_dev
	     element from st1 but what the heck, do more testing.  */
	  size_t len = strlen (objdir) + 1 + _D_EXACT_NAMLEN (d) + 1;
	  char tmpbuf[len];

	  stpcpy (stpcpy (stpcpy (tmpbuf, objdir), "/"), d->d_name);

	  if (stat64 (tmpbuf, &st3) < 0)
	    {
	      printf ("cannot stat entry from readdir: %m\n");
	      result = 1;
	      d = NULL;
	      break;
	    }

	  if (st3.st_dev == st2.st_dev
	      && strcmp (d->d_name, buf + strlen (buf) - 14) == 0)
	    break;
	}
    }

  DIAG_POP_NEEDS_COMMENT;

  if (d == NULL)
    {
      printf ("haven't found new directory \"%s\"\n", buf);
      exit (1);
    }

  if (closedir (dir2) < 0)
    {
      printf ("closing dir2 failed: %m\n");
      result = 1;
    }

  if (chdir (buf) < 0)
    {
      printf ("cannot change to new directory: %m\n");
      exit (1);
    }

  dir2 = opendir (buf);
  if (dir2 == NULL)
    {
      printf ("cannot open new directory: %m\n");
      exit (1);
    }

  if (fstat64 (dirfd (dir2), &st2) < 0)
    {
      printf ("cannot fstat new directory \"%s\": %m\n", buf);
      exit (1);
    }
  if (st1.st_dev != st2.st_dev || st1.st_ino != st2.st_ino)
    {
      printf ("stat of new directory failed: (%lld,%lld) vs (%lld,%lld)\n",
	      (long long int) st1.st_dev, (long long int) st1.st_ino,
	      (long long int) st2.st_dev, (long long int) st2.st_ino);
      exit (1);
    }

  if (mkdir ("another-dir", 0777) < 0)
    {
      printf ("cannot create \"another-dir\": %m\n");
      exit (1);
    }
  fd = open ("and-a-file", O_RDWR | O_CREAT | O_EXCL, 0666);
  if (fd == -1)
    {
      printf ("cannot create \"and-a-file\": %m\n");
      exit (1);
    }
  close (fd);

  /* Some tests about error reporting.  */
  errno = 0;
  if (chdir ("and-a-file") >= 0)
    {
      printf ("chdir to \"and-a-file\" succeeded\n");
      exit (1);
    }
  if (errno != ENOTDIR)
    {
      printf ("chdir to \"and-a-file\" didn't set correct error\n");
      result = 1;
    }

  errno = 0;
  if (chdir ("and-a-file/..") >= 0)
    {
      printf ("chdir to \"and-a-file/..\" succeeded\n");
      exit (1);
    }
  if (errno != ENOTDIR)
    {
      printf ("chdir to \"and-a-file/..\" didn't set correct error\n");
      result = 1;
    }

  errno = 0;
  if (chdir ("another-dir/../and-a-file") >= 0)
    {
      printf ("chdir to \"another-dir/../and-a-file\" succeeded\n");
      exit (1);
    }
  if (errno != ENOTDIR)
    {
      printf ("chdir to \"another-dir/../and-a-file\" didn't set correct error\n");
      result = 1;
    }

  /* The test below covers the deprecated readdir64_r function.  */
  DIAG_PUSH_NEEDS_COMMENT;
  DIAG_IGNORE_NEEDS_COMMENT (4.9, "-Wdeprecated-declarations");

  /* We now should have a directory and a file in the new directory.  */
  rewinddir (dir2);
  while (readdir64_r (dir2, &direntbuf.d, &d) == 0 && d != NULL)
    {
      if (strcmp (d->d_name, ".") == 0
	  || strcmp (d->d_name, "..") == 0
	  || strcmp (d->d_name, "another-dir") == 0)
	{
	  if (d->d_type != DT_UNKNOWN && d->d_type != DT_DIR)
	    {
	      printf ("d_type for \"%s\" is wrong\n", d->d_name);
	      result = 1;
	    }
	  if (stat64 (d->d_name, &st3) < 0)
	    {
	      printf ("cannot stat \"%s\" is wrong\n", d->d_name);
	      result = 1;
	    }
	  else if (! S_ISDIR (st3.st_mode))
	    {
	      printf ("\"%s\" is no directory\n", d->d_name);
	      result = 1;
	    }
	}
      else if (strcmp (d->d_name, "and-a-file") == 0)
	{
	  if (d->d_type != DT_UNKNOWN && d->d_type != DT_REG)
	    {
	      printf ("d_type for \"%s\" is wrong\n", d->d_name);
	      result = 1;
	    }
	  if (stat64 (d->d_name, &st3) < 0)
	    {
	      printf ("cannot stat \"%s\" is wrong\n", d->d_name);
	      result = 1;
	    }
	  else if (! S_ISREG (st3.st_mode))
	    {
	      printf ("\"%s\" is no regular file\n", d->d_name);
	      result = 1;
	    }
	}
      else
	{
	  printf ("unexpected directory entry \"%s\"\n", d->d_name);
	  result = 1;
	}
    }

  DIAG_POP_NEEDS_COMMENT;

  if (stat64 ("does-not-exist", &st1) >= 0)
    {
      puts ("stat for unexisting file did not fail");
      result = 1;
    }

  /* Free all resources.  */

  if (closedir (dir1) < 0)
    {
      printf ("closing dir1 failed: %m\n");
      result = 1;
    }
  if (closedir (dir2) < 0)
    {
      printf ("second closing dir2 failed: %m\n");
      result = 1;
    }

  if (rmdir ("another-dir") < 0)
    {
      printf ("cannot remove \"another-dir\": %m\n");
      result = 1;
    }

  if (unlink ("and-a-file") < 0)
    {
      printf ("cannot remove \"and-a-file\": %m\n");
      result = 1;
    }

  /* One more test before we leave: mkdir() is supposed to fail with
     EEXIST if the named file is a symlink.  */
  if (symlink ("a-symlink", "a-symlink") != 0)
    {
      printf ("cannot create symlink \"a-symlink\": %m\n");
      result = 1;
    }
  else
    {
      if (mkdir ("a-symlink", 0666) == 0)
	{
	  puts ("can make directory \"a-symlink\"");
	  result = 1;
	}
      else if (errno != EEXIST)
	{
	  puts ("mkdir(\"a-symlink\") does not fail with EEXIST\n");
	  result = 1;
	}
      if (unlink ("a-symlink") < 0)
	{
	  printf ("cannot unlink \"a-symlink\": %m\n");
	  result = 1;
	}
    }

  if (chdir (srcdir) < 0)
    {
      printf ("cannot change back to source directory: %m\n");
      exit (1);
    }

  if (rmdir (buf) < 0)
    {
      printf ("cannot remove \"%s\": %m\n", buf);
      result = 1;
    }
  free (objdir_copy1);
  free (objdir_copy2);

  if (result == 0)
    puts ("all OK");

  return result;
}
