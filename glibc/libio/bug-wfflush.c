/* Test program for bug in wide streams.  */

#include <stdio.h>
#include <wchar.h>

static void do_prepare (void);
#define PREPARE(argc, argv) do_prepare ()
static int do_test (void);
#define TEST_FUNCTION do_test ()
#include <test-skeleton.c>

static char *temp_file;

static void
do_prepare (void)
{
  int fd = create_temp_file ("bug-ungetc.", &temp_file);
  if (fd == -1)
    {
      printf ("cannot create temporary file: %m\n");
      exit (1);
    }
  write (fd, "1!", 2);
  close (fd);
}

static int
do_test (void)
{
  FILE *f = fopen (temp_file, "r+");

  if (f == NULL)
    {
      printf ("fopen: %m\n");
      return 1;
    }

#define L_(s) L##s
  //#define fwscanf fscanf
  //#define fwprintf fprintf

  int i;
  if (fwscanf (f, L_("%d!"), &i) != 1)
    {
      printf ("fwscanf failed\n");
      return 1;
    }

  rewind (f);
  if (ferror (f))
    {
      printf ("rewind: %m\n");
      return 1;
    }

  if (fputws (L_("1!"), f) == EOF)
    {
      printf ("fputws: %m\n");
      return 1;
    }

  if (fflush (f) != 0)
    {
      printf ("fflush: %m\n");
      return 1;
    }

  if (fclose (f) != 0)
    {
      printf ("fclose: %m\n");
      return 1;
    }

  puts ("Test succeeded.");
  return 0;
}
