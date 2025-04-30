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
  static const char dir_name[] = "/tst-renameat.XXXXXX";

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
  puts ("file created");

  struct stat64 st1;
  if (fstat64 (fd, &st1) != 0)
    {
      puts ("fstat64 failed");
      return 1;
    }

  /* Using a descriptor for a normal file must fail.  */
  if (renameat (fd, "some-file", dir_fd, "another-file") == 0)
    {
      puts ("renameat with normal file descriptor succeeded");
      return 1;
    }
  if (errno != ENOTDIR)
    {
      puts ("error for renameat with normal file descriptor not ENOTDIR");
      return 1;
    }

  if (renameat (dir_fd, "some-file", fd, "another-file") == 0)
    {
      puts ("2nd renameat with normal file descriptor succeeded");
      return 1;
    }
  if (errno != ENOTDIR)
    {
      puts ("error for 2nd renameat with normal file descriptor not ENOTDIR");
      return 1;
    }

  close (fd);

  if (renameat (dir_fd, "some-file", dir_fd, "another-file") != 0)
    {
      puts ("renameat failed");
      return 1;
    }

  struct stat64 st2;
  if (fstatat64 (dir_fd, "some-file", &st2, 0) == 0)
    {
      puts ("fstatat64 succeeded");
      return 1;
    }
  if (errno != ENOENT)
    {
      puts ("fstatat64 did not fail with ENOENT");
      return 1;
    }

  if (fstatat64 (dir_fd, "another-file", &st2, 0) != 0)
    {
      puts ("2nd fstatat64 failed");
      return 1;
    }

  if (st1.st_dev != st2.st_dev
      || st1.st_ino != st2.st_ino
      || st1.st_size != st2.st_size)
    {
      puts ("stat results do not match");
      return 1;
    }

  /* Create a file descriptor which is closed again right away.  */
  int dir_fd2 = dup (dir_fd);
  if (dir_fd2 == -1)
    {
      puts ("dup failed");
      return 1;
    }
  close (dir_fd2);

  if (renameat (dir_fd2, "another-file", dir_fd, "some-file") == 0)
    {
      puts ("renameat with closed file descriptor succeeded");
      return 1;
    }
  if (errno != EBADF)
    {
      puts ("error for renameat with closed file descriptor not EBADF");
      return 1;
    }

  if (renameat (dir_fd, "another-file", dir_fd2, "some-file") == 0)
    {
      puts ("2nd renameat with closed file descriptor succeeded");
      return 1;
    }
  if (errno != EBADF)
    {
      puts ("error for 2nd renameat with closed file descriptor not EBADF");
      return 1;
    }

  if (unlinkat (dir_fd, "another-file", 0) != 0)
    {
      puts ("unlinkat failed");
      return 1;
    }

  if (renameat (-1, "another-file", dir_fd, "some-file") == 0)
    {
      puts ("renameat with invalid file descriptor succeeded");
      return 1;
    }
  if (errno != EBADF)
    {
      puts ("error for renameat with invalid file descriptor not EBADF");
      return 1;
    }

  if (renameat (dir_fd, "another-file", -1, "some-file") == 0)
    {
      puts ("2nd renameat with invalid file descriptor succeeded");
      return 1;
    }
  if (errno != EBADF)
    {
      puts ("error for 2nd renameat with invalid file descriptor not EBADF");
      return 1;
    }

  close (dir_fd);

  return 0;
}
