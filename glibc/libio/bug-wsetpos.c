/* Test program for fsetpos on a wide character stream.  */

#include <assert.h>
#include <stdio.h>
#include <wchar.h>

static void do_prepare (void);
#define PREPARE(argc, argv) do_prepare ()
static int do_test (void);
#define TEST_FUNCTION do_test ()
#include <test-skeleton.c>

static const char pattern[] = "12345";
static char *temp_file;

static void
do_prepare (void)
{
  int fd = create_temp_file ("bug-wsetpos.", &temp_file);
  if (fd == -1)
    {
      printf ("cannot create temporary file: %m\n");
      exit (1);
    }
  write (fd, pattern, sizeof (pattern));
  close (fd);
}

static int
do_test (void)
{
  FILE *fp = fopen (temp_file, "r");
  fpos_t pos;
  wchar_t c;

  if (fp == NULL)
    {
      printf ("fdopen: %m\n");
      return 1;
    }

  c = fgetwc (fp); assert (c == L'1');
  c = fgetwc (fp); assert (c == L'2');

  if (fgetpos (fp, &pos) == EOF)
    {
      printf ("fgetpos: %m\n");
      return 1;
    }

  rewind (fp);
  if (ferror (fp))
    {
      printf ("rewind: %m\n");
      return 1;
    }

  c = fgetwc (fp); assert (c == L'1');

  if (fsetpos (fp, &pos) == EOF)
    {
      printf ("fsetpos: %m\n");
      return 1;
    }

  c = fgetwc (fp);
  if (c != L'3')
    {
      puts ("fsetpos failed");
      return 1;
    }

  puts ("Test succeeded.");
  return 0;
}
