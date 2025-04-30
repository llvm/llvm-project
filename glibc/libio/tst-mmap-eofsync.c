/* Test program for synchronization of stdio state with file after EOF.  */

#include <stdio.h>
#include <error.h>
#include <errno.h>

static void do_prepare (void);
#define PREPARE(argc, argv) do_prepare ()
static int do_test (void);
#define TEST_FUNCTION do_test ()
#include <test-skeleton.c>

static char *temp_file;
static int temp_fd;

static char text1[] = "Line the first\n";
static char text2[] = "Line the second\n";

static void
do_prepare (void)
{
  temp_fd = create_temp_file ("tst-mmap-eofsync.", &temp_file);
  if (temp_fd == -1)
    error (1, errno, "cannot create temporary file");
  else
    {
      ssize_t cc = write (temp_fd, text1, sizeof text1 - 1);
      if (cc != sizeof text1 - 1)
	error (1, errno, "cannot write to temporary file");
    }
}

static int
do_test (void)
{
  FILE *f;
  char buf[128];
  int result = 0;
  int c;

  f = fopen (temp_file, "rm");
  if (f == NULL)
    {
      perror (temp_file);
      return 1;
    }

  if (fgets (buf, sizeof buf, f) == NULL)
    {
      perror ("fgets");
      return 1;
    }

  if (strcmp (buf, text1))
    {
      printf ("read \"%s\", expected \"%s\"\n", buf, text1);
      result = 1;
    }

  printf ("feof = %d, ferror = %d immediately after fgets\n",
	  feof (f), ferror (f));

  c = fgetc (f);
  if (c == EOF)
    printf ("fgetc -> EOF (feof = %d, ferror = %d)\n",
	    feof (f), ferror (f));
  else
    {
      printf ("fgetc returned %o (feof = %d, ferror = %d)\n",
	      c, feof (f), ferror (f));
      result = 1;
    }

  c = write (temp_fd, text2, sizeof text2 - 1);
  if (c == sizeof text2 - 1)
    printf ("wrote more to file\n");
  else
    {
      printf ("wrote %d != %zd (%m)\n", c, sizeof text2 - 1);
      result = 1;
    }

  if (fgets (buf, sizeof buf, f) == NULL)
    {
      printf ("second fgets fails: feof = %d, ferror = %d (%m)\n",
	      feof (f), ferror (f));
      clearerr (f);
      if (fgets (buf, sizeof buf, f) == NULL)
	{
	  printf ("retry fgets fails: feof = %d, ferror = %d (%m)\n",
		  feof (f), ferror (f));
	  result = 1;
	}
    }
  if (result == 0 && strcmp (buf, text2))
    {
      printf ("second time read \"%s\", expected \"%s\"\n", buf, text2);
      result = 1;
    }

  fclose (f);

  return result;
}
