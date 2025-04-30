#include <dirent.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>


static void prepare (void);
#define PREPARE(argc, argv) prepare ()

static int do_test (void);
#define TEST_FUNCTION do_test ()

#include "../test-skeleton.c"

static int dir_fd;

static void
prepare (void)
{
  size_t test_dir_len = strlen (test_dir);
  static const char dir_name[] = "/tst-openat.XXXXXX";

  size_t dirbuflen = test_dir_len + sizeof (dir_name);
  char *dirbuf = malloc (dirbuflen);
  if (dirbuf == NULL)
    {
      puts ("out of memory");
      exit (1);
    }

  snprintf (dirbuf, dirbuflen, "%s%s", test_dir, dir_name);
  if (mkdtemp (dirbuf) == NULL)
    {
      puts ("cannot create temporary directory");
      exit (1);
    }

  add_temp_file (dirbuf);

  dir_fd = open (dirbuf, O_RDONLY | O_DIRECTORY);
  if (dir_fd == -1)
    {
      puts ("cannot open directory");
      exit (1);
    }
}


static int
do_test (void)
{
  /* fdopendir takes over the descriptor, make a copy.  */
  int dupfd = dup (dir_fd);
  if (dupfd == -1)
    {
      puts ("dup failed");
      return 1;
    }
  if (lseek (dupfd, 0, SEEK_SET) != 0)
    {
      puts ("1st lseek failed");
      return 1;
    }

  /* The directory should be empty safe the . and .. files.  */
  DIR *dir = fdopendir (dupfd);
  if (dir == NULL)
    {
      puts ("fdopendir failed");
      return 1;
    }
  struct dirent64 *d;
  while ((d = readdir64 (dir)) != NULL)
    if (strcmp (d->d_name, ".") != 0 && strcmp (d->d_name, "..") != 0)
      {
	printf ("temp directory contains file \"%s\"\n", d->d_name);
	return 1;
      }
  closedir (dir);

  /* Try to create a file.  */
  int fd = openat (dir_fd, "some-file", O_CREAT|O_RDWR|O_EXCL, 0666);
  if (fd == -1)
    {
      if (errno == ENOSYS)
	{
	  puts ("*at functions not supported");
	  return 0;
	}

      puts ("file creation failed");
      return 1;
    }
  write (fd, "hello", 5);

  /* Before closing the file, try using this file descriptor to open
     another file.  This must fail.  */
  int fd2 = openat (fd, "should-not-work", O_CREAT|O_RDWR, 0666);
  if (fd2 != -1)
    {
      puts ("openat using descriptor for normal file worked");
      return 1;
    }
  if (errno != ENOTDIR)
    {
      puts ("error for openat using descriptor for normal file not ENOTDIR ");
      return 1;
    }

  close (fd);
  puts ("file created");

  /* fdopendir takes over the descriptor, make a copy.  */
  dupfd = dup (dir_fd);
  if (dupfd == -1)
    {
      puts ("dup failed");
      return 1;
    }
  if (lseek (dupfd, 0, SEEK_SET) != 0)
    {
      puts ("2nd lseek failed");
      return 1;
    }

  /* The directory should be empty safe the . and .. files.  */
  dir = fdopendir (dupfd);
  if (dir == NULL)
    {
      puts ("fdopendir failed");
      return 1;
    }
  bool seen_file = false;
  while ((d = readdir64 (dir)) != NULL)
    if (strcmp (d->d_name, ".") != 0 && strcmp (d->d_name, "..") != 0)
      {
	if (strcmp (d->d_name, "some-file") != 0)
	  {
	    printf ("temp directory contains file \"%s\"\n", d->d_name);
	    return 1;
	  }

	seen_file = true;
      }
  closedir (dir);

  if (!seen_file)
    {
      puts ("file not created in correct directory");
      return 1;
    }

  int cwdfd = open (".", O_RDONLY | O_DIRECTORY);
  if (cwdfd == -1)
    {
      puts ("cannot get descriptor for cwd");
      return 1;
    }

  if (fchdir (dir_fd) != 0)
    {
      puts ("1st fchdir failed");
      return 1;
    }

  if (unlink ("some-file") != 0)
    {
      puts ("unlink failed");
      return 1;
    }

  if (fchdir (cwdfd) != 0)
    {
      puts ("2nd fchdir failed");
      return 1;
    }

  close (dir_fd);
  close (cwdfd);

  /* With the file descriptor closed the next call must fail.  */
  fd = openat (dir_fd, "some-file", O_CREAT|O_RDWR|O_EXCL, 0666);
  if (fd != -1)
    {
      puts ("openat using closed descriptor succeeded");
      return 1;
    }
  if (errno != EBADF)
    {
      puts ("openat using closed descriptor did not set EBADF");
      return 1;
    }

  fd = openat (-1, "some-file", O_CREAT|O_RDWR|O_EXCL, 0666);
  if (fd != -1)
    {
      puts ("openat using -1 descriptor succeeded");
      return 1;
    }
  if (errno != EBADF)
    {
      puts ("openat using -1 descriptor did not set EBADF");
      return 1;
    }

  return 0;
}
