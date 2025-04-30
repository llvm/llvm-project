#include <dirent.h>
#include <fcntl.h>
#include <stdbool.h>
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
  static const char dir_name[] = "/tst-mkfifoat.XXXXXX";

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

  /* Create a new directory.  */
  int e = mkfifoat (dir_fd, "some-fifo", 0777);
  if (e == -1)
    {
      if (errno == ENOSYS)
	{
	  puts ("*at functions not supported");
	  return 0;
	}

      puts ("fifo creation failed");
      return 1;
    }

  struct stat64 st1;
  if (fstatat64 (dir_fd, "some-fifo", &st1, 0) != 0)
    {
      puts ("fstat64 failed");
      return 1;
    }
  if (!S_ISFIFO (st1.st_mode))
    {
      puts ("mkfifoat did not create FIFO");
      return 1;
    }

  dupfd = dup (dir_fd);
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

  dir = fdopendir (dupfd);
  if (dir == NULL)
    {
      puts ("2nd fdopendir failed");
      return 1;
    }
  bool has_some_fifo = false;
  while ((d = readdir64 (dir)) != NULL)
    if (strcmp (d->d_name, "some-fifo") == 0)
      {
	has_some_fifo = true;
	if (d->d_type != DT_UNKNOWN && d->d_type != DT_FIFO)
	  {
	    puts ("d_type for some-fifo wrong");
	    return 1;
	  }
      }
    else if (strcmp (d->d_name, ".") != 0 && strcmp (d->d_name, "..") != 0)
      {
	printf ("temp directory contains file \"%s\"\n", d->d_name);
	return 1;
      }
  closedir (dir);

  if (!has_some_fifo)
    {
      puts ("some-fifo not in directory list");
      return 1;
    }

  if (unlinkat (dir_fd, "some-fifo", 0) != 0)
    {
      puts ("unlinkat failed");
      return 1;
    }

  close (dir_fd);

  return 0;
}
