/* Test program for ungetc/ftell interaction bug.  */

#include <stdio.h>

static void do_prepare (void);
#define PREPARE(argc, argv) do_prepare ()
static int do_test (void);
#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

static const char pattern[] = "12345";
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
  write (fd, pattern, sizeof (pattern));
  close (fd);
}

static int
do_one_test (int mode)
{
  FILE *f;

  f = fopen (temp_file, "r");
  if (f == NULL)
    {
      printf ("could not open temporary file: %m\n");
      return 1;
    }

  if (mode == 1 && ftell (f) != 0)
    {
      printf ("first ftell returned wrong position %ld\n", ftell (f));
      return 1;
    }

  if (fgetc (f) != '1' || fgetc (f) != '2')
    {
      puts ("fgetc failed");
      return 1;
    }

  if (mode == 2 && ftell (f) != 2)
    {
      printf ("second ftell returned wrong position %ld\n", ftell (f));
      return 1;
    }

  if (ungetc ('6', f) != '6')
    {
      puts ("ungetc failed");
      return 1;
    }

  if (ftell (f) != 1)
    {
      printf ("third ftell returned wrong position %ld\n", ftell (f));
      return 1;
    }

  if (fgetc (f) != '6')
    {
      puts ("fgetc failed");
      return 1;
    }

  if (ftell (f) != 2)
    {
      printf ("fourth ftell returned wrong position %ld\n", ftell (f));
      return 1;
    }

  fclose (f);

  return 0;
}

static int
do_test (void)
{
  int mode;
  for (mode = 0; mode <= 2; mode++)
    if (do_one_test (mode))
      return 1;
  return 0;
}
