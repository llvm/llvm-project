/* Test program for synchronization of stdio state with file after EOF.  */

#include <stdio.h>
#include <error.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>

static void do_prepare (void);
#define PREPARE(argc, argv) do_prepare ()
static int do_test (void);
#define TEST_FUNCTION do_test ()
#include <test-skeleton.c>

static char *temp_file;
static int temp_fd;

static char *pages;

static void
do_prepare (void)
{
  pages = xmalloc (getpagesize () * 2);
  memset (pages, 'a', getpagesize ());
  memset (pages + getpagesize (), 'b', getpagesize ());

  temp_fd = create_temp_file ("tst-mmap2-eofsync.", &temp_file);
  if (temp_fd == -1)
    error (1, errno, "cannot create temporary file");
  else
    {
      ssize_t cc = write (temp_fd, pages, getpagesize ());
      if (cc != getpagesize ())
	error (1, errno, "cannot write to temporary file");
    }
}

static int
do_test (void)
{
  const size_t pagesize = getpagesize ();
  FILE *f;
  char buf[pagesize];
  int result = 0;
  int c;

  f = fopen (temp_file, "rm");
  if (f == NULL)
    {
      perror (temp_file);
      return 1;
    }

  if (fread (buf, pagesize, 1, f) != 1)
    {
      perror ("fread");
      return 1;
    }

  if (memcmp (buf, pages, pagesize))
    {
      puts ("data mismatch in page 1");
      result = 1;
    }

  printf ("feof = %d, ferror = %d immediately after fread\n",
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

  c = write (temp_fd, pages + pagesize, pagesize);
  if (c == (ssize_t) pagesize)
    printf ("wrote more to file\n");
  else
    {
      printf ("wrote %d != %zd (%m)\n", c, pagesize);
      result = 1;
    }

  if (fread (buf, pagesize, 1, f) != 1)
    {
      printf ("second fread fails: feof = %d, ferror = %d (%m)\n",
	      feof (f), ferror (f));
      clearerr (f);
      if (fread (buf, pagesize, 1, f) != 1)
	{
	  printf ("retry fread fails: feof = %d, ferror = %d (%m)\n",
		  feof (f), ferror (f));
	  result = 1;
	}
    }
  if (result == 0 && memcmp (buf, pages + pagesize, pagesize))
    {
      puts ("data mismatch in page 2");
      result = 1;
    }

  fseek (f, pagesize - 1, SEEK_SET);
  c = fgetc (f);
  if (c != 'a')
    {
      printf ("fgetc at end of page 1 read '%c' (%m)\n", c);
      result = 1;
    }

  if (ftruncate (temp_fd, pagesize) < 0)
    {
      printf ("ftruncate failed: %m\n");
      result = 1;
    }

  fflush (f);

  c = fgetc (f);
  if (c == EOF)
    printf ("after truncate fgetc -> EOF (feof = %d, ferror = %d)\n",
	    feof (f), ferror (f));
  else
    {
      printf ("after truncate fgetc returned '%c' (feof = %d, ferror = %d)\n",
	      c, feof (f), ferror (f));
      result = 1;
    }

  fclose (f);

  return result;
}
