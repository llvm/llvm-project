/* Test case for bug with mmap stdio read past end of file.  */

#include <stdio.h>
#include <error.h>
#include <errno.h>

static void do_prepare (void);
#define PREPARE(argc, argv) do_prepare ()
static int do_test (void);
#define TEST_FUNCTION do_test ()
#include <test-skeleton.c>

static char *temp_file;

static const char text1[] = "hello\n";

static void
do_prepare (void)
{
  int temp_fd = create_temp_file ("tst-mmap-offend.", &temp_file);
  if (temp_fd == -1)
    error (1, errno, "cannot create temporary file");
  else
    {
      ssize_t cc = write (temp_fd, text1, sizeof text1 - 1);
      if (cc != sizeof text1 - 1)
	error (1, errno, "cannot write to temporary file");
    }
  close (temp_fd);
}

static int
do_test (void)
{
  unsigned char buffer[8192];
  int result = 0;
  FILE *f = fopen (temp_file, "rm");
  size_t cc;

  if (f == NULL)
    {
      perror (temp_file);
      return 1;
    }

  cc = fread (buffer, 1, sizeof (buffer), f);
  printf ("fread %zu: \"%.*s\"\n", cc, (int) cc, buffer);
  if (cc != sizeof text1 - 1)
    {
      perror ("fread");
      result = 1;
    }

  if (fseek (f, 2048, SEEK_SET) != 0)
    {
      perror ("fseek off end");
      result = 1;
    }

  if (fread (buffer, 1, sizeof (buffer), f) != 0
      || ferror (f) || !feof (f))
    {
      printf ("after fread error %d eof %d\n",
	      ferror (f), feof (f));
      result = 1;
    }

  printf ("ftell %ld\n", ftell (f));

  if (fseek (f, 0, SEEK_SET) != 0)
    {
      perror ("fseek rewind");
      result = 1;
    }

  cc = fread (buffer, 1, sizeof (buffer), f);
  printf ("fread after rewind %zu: \"%.*s\"\n", cc, (int) cc, buffer);
  if (cc != sizeof text1 - 1)
    {
      perror ("fread after rewind");
      result = 1;
    }

  fclose (f);
  return result;
}
