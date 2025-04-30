#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int fd;
static char *fname;


static void prepare (void);
#define PREPARE(argc, argv) prepare ()


#define TEST_FUNCTION do_test ()
static int do_test (void);
#include "../test-skeleton.c"


static void
prepare (void)
{
  fd = create_temp_file ("wrewind.", &fname);
  if (fd == -1)
    exit (3);
}


static int
do_test (void)
{
  char buf[100];
  FILE *fp;
  int result = 0;

  fp = fdopen (fd, "w");
  if (fp == NULL)
    {
      puts ("cannot create file");
      exit (1);
    }

  if (fputs ("one\n", fp) == EOF || fputs ("two\n", fp) == EOF)
    {
      puts ("cannot create filec content");
      exit (1);
    }

  fclose (fp);

  fp = fopen (fname, "a+");
  if (fp == NULL)
    {
      puts ("cannot fopen a+");
      exit (1);
    }

  if (fgets (buf, sizeof (buf), fp) == NULL)
    {
      puts ("cannot read after fopen a+");
      exit (1);
    }

  if (strcmp (buf, "one\n") != 0)
    {
      puts ("read after fopen a+ produced wrong result");
      result = 1;
    }

  fclose (fp);

  fd = open (fname, O_RDWR);
  if (fd == -1)
    {
      puts ("open failed");
      exit (1);
    }

  fp = fdopen (fd, "a+");
  if (fp == NULL)
    {
      puts ("fopen after open failed");
      exit (1);
    }

  if (fgets (buf, sizeof (buf), fp) == NULL)
    {
      puts ("cannot read after fdopen a+");
      exit (1);
    }

  if (strcmp (buf, "one\n") != 0)
    {
      puts ("read after fdopen a+ produced wrong result");
      result = 1;
    }

  fclose (fp);

  return result;
}
