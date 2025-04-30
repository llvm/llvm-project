/* Test program for ungetc/ftell interaction bug.  */

#include <stdio.h>

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
do_test (void)
{
  int i;
  FILE *f;
  char buf[10];
  long offset, diff;
  int result = 0;

  f = fopen (temp_file, "rw");

  rewind (f);
  for (i = 0; i < 3; i++)
    printf ("%c\n", getc (f));
  offset = ftell (f);
  printf ("offset = %ld\n", offset);
  if (ungetc ('4', f) != '4')
    {
      printf ("ungetc failed\n");
      abort ();
    }
  printf ("offset after ungetc = %ld\n", ftell (f));

  i = fread ((void *) buf, 4, (size_t) 1, f);
  printf ("read %d (%c), offset = %ld\n", i, buf[0], ftell (f));
  diff = ftell (f) - offset;
  if (diff != 3)
    {
      printf ("ftell did not update properly.  got %ld, expected 3\n", diff);
      result = 1;
    }
  fclose (f);

  return result;
}
