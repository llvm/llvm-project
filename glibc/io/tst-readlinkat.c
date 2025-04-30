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
  static const char dir_name[] = "/tst-symlinkat.XXXXXX";

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

  static const char symlinkcontent[] = "some-file";
  if (symlinkat (symlinkcontent, dir_fd, "another-file") != 0)
    {
      puts ("symlinkat failed");
      return 1;
    }

  struct stat64 st2;
  if (fstatat64 (dir_fd, "another-file", &st2, AT_SYMLINK_NOFOLLOW) != 0)
    {
      puts ("fstatat64 failed");
      return 1;
    }
  if (!S_ISLNK (st2.st_mode))
    {
      puts ("2nd fstatat64 does not show file is a symlink");
      return 1;
    }

  if (fstatat64 (dir_fd, symlinkcontent, &st2, AT_SYMLINK_NOFOLLOW) == 0)
    {
      puts ("2nd fstatat64 succeeded");
      return 1;
    }

  char buf[100];
  int n = readlinkat (dir_fd, "another-file", buf, sizeof (buf));
  if (n == -1)
    {
      puts ("readlinkat failed");
      return 1;
    }
  if (n != sizeof (symlinkcontent) - 1)
    {
      printf ("readlinkat returned %d, expected %zu\n",
	      n, sizeof (symlinkcontent) - 1);
      return 1;
    }
  if (strncmp (buf, symlinkcontent, n) != 0)
    {
      puts ("readlinkat retrieved wrong link content");
      return 1;
    }

  if (unlinkat (dir_fd, "another-file", 0) != 0)
    {
      puts ("unlinkat failed");
      return 1;
    }

  close (dir_fd);

  return 0;
}
