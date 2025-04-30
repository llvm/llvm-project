#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>


static void do_prepare (void);
#define PREPARE(argc, argv) do_prepare ()
static int do_test (void);
#define TEST_FUNCTION do_test ()
#include <test-skeleton.c>


int fd;


static void
do_prepare (void)
{
  fd = create_temp_file ("tst-eof.", NULL);
  if (fd == -1)
    {
      printf ("cannot create temporary file: %m\n");
      exit (1);
    }
}


static int
do_test (void)
{
  char buf[40];
  FILE *fp;

  if (write (fd, "some string\n", 12) != 12)
    {
      printf ("cannot write temporary file: %m\n");
      return 1;
    }

  if (lseek (fd, 0, SEEK_SET) == (off_t) -1)
    {
      printf ("cannot reposition temporary file: %m\n");
      return 1;
    }

  fp = fdopen (fd, "r");
  if (fp == NULL)
    {
      printf ("cannot create stream: %m\n");
      return 1;
    }

  if (feof (fp))
    {
      puts ("EOF set after fdopen");
      return 1;
    }

  if (fread (buf, 1, 20, fp) != 12)
    {
      puts ("didn't read the correct number of bytes");
      return 1;
    }

  if (! feof (fp))
    {
      puts ("EOF not set after fread");
      return 1;
    }

  fclose (fp);

  return 0;
}
