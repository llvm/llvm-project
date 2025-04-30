/* Test for faccessat function.  */

#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>


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
  static const char dir_name[] = "/tst-faccessat.XXXXXX";

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

  /* The directory should be empty save the . and .. files.  */
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
  puts ("file created");

  /* Before closing the file, try using this file descriptor to open
     another file.  This must fail.  */
  if (faccessat (fd, "should-not-work", F_OK, AT_EACCESS) != -1)
    {
      puts ("faccessat using descriptor for normal file worked");
      return 1;
    }
  if (errno != ENOTDIR)
    {
      puts ("\
error for faccessat using descriptor for normal file not ENOTDIR ");
      return 1;
    }

  close (fd);

  int result = 0;

  if (faccessat (dir_fd, "some-file", F_OK, AT_EACCESS))
    {
      printf ("faccessat F_OK: %m\n");
      result = 1;
    }
  if (faccessat (dir_fd, "some-file", W_OK, AT_EACCESS))
    {
      printf ("faccessat W_OK: %m\n");
      result = 1;
    }

  errno = 0;
  if (faccessat (dir_fd, "some-file", X_OK, AT_EACCESS) == 0
      || errno != EACCES)
    {
      printf ("faccessat X_OK on nonexecutable: %m\n");
      result = 1;
    }

  if (fchmodat (dir_fd, "some-file", 0400, 0) != 0)
    {
      printf ("fchownat failed: %m\n");
      return 1;
    }

  if (faccessat (dir_fd, "some-file", R_OK, AT_EACCESS))
    {
      printf ("faccessat R_OK: %m\n");
      result = 1;
    }

  errno = 0;
  if (faccessat (dir_fd, "some-file", W_OK, AT_EACCESS) == 0
      ? (geteuid () != 0) : (errno != EACCES))
    {
      printf ("faccessat W_OK on unwritable file: %m\n");
      result = 1;
    }

  /* Create a file descriptor which is closed again right away.  */
  int dir_fd2 = dup (dir_fd);
  if (dir_fd2 == -1)
    {
      puts ("dup failed");
      return 1;
    }
  close (dir_fd2);

  /* With the file descriptor closed the next call must fail.  */
  if (faccessat (dir_fd2, "some-file", F_OK, AT_EACCESS) != -1)
    {
      puts ("faccessat using closed descriptor succeeded");
      return 1;
    }
  if (errno != EBADF)
    {
      puts ("faccessat using closed descriptor did not set EBADF");
      return 1;
    }

  /* Same with a non-existing file.  */
  if (faccessat (dir_fd2, "non-existing-file", F_OK, AT_EACCESS) != -1)
    {
      puts ("2nd faccessat using closed descriptor succeeded");
      return 1;
    }
  if (errno != EBADF)
    {
      puts ("2nd faccessat using closed descriptor did not set EBADF");
      return 1;
    }

  if (unlinkat (dir_fd, "some-file", 0) != 0)
    {
      puts ("unlinkat failed");
      result = 1;
    }

  close (dir_fd);

  fd = faccessat (-1, "some-file", F_OK, AT_EACCESS);
  if (fd != -1)
    {
      puts ("faccessat using -1 descriptor succeeded");
      return 1;
    }
  if (errno != EBADF)
    {
      puts ("faccessat using -1 descriptor did not set EBADF");
      return 1;
    }

  return result;
}
