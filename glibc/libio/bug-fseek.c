#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


static char *fname;


static void do_prepare (void);
#define PREPARE(argc, argv) do_prepare ()
static int do_test (void);
#define TEST_FUNCTION do_test ()
#include <test-skeleton.c>


static void
do_prepare (void)
{
  static const char pattern[] = "12345678901234567890";
  int fd = create_temp_file ("bug-fseek.", &fname);
  if (fd == -1)
    {
      printf ("cannot create temporary file: %m\n");
      exit (1);
    }

  if (write (fd, pattern, sizeof (pattern)) != sizeof (pattern))
    {
      perror ("short write");
      abort ();
    }
  close (fd);
}



static int
do_test (void)
{
  FILE *f;
  int result = 0;
  char buf[10];


  if ((f = fopen (fname, "r")) == (FILE *) NULL)
    {
      perror ("fopen(\"r\")");
    }

  fread (buf, 3, 1, f);
  errno = 0;
  if (fseek (f, -10, SEEK_CUR) == 0)
    {
      printf ("fseek() for r to before start of file worked!\n");
      result = 1;
    }
  else if (errno != EINVAL)
    {
      printf ("\
fseek() for r to before start of file did not set errno to EINVAL.  \
Got %d instead\n",
	 errno);
      result = 1;
    }

  fclose (f);


  if ((f = fopen (fname, "r+")) == (FILE *) NULL)
    {
      perror ("fopen(\"r+\")");
    }

  fread (buf, 3, 1, f);
  errno = 0;
  if (fseek (f, -10, SEEK_CUR) == 0)
    {
      printf ("fseek() for r+ to before start of file worked!\n");
      result = 1;
    }
  else if (errno != EINVAL)
    {
      printf ("\
fseek() for r+ to before start of file did not set errno to EINVAL.  \
Got %d instead\n",
	 errno);
      result = 1;
    }

  fclose (f);


  if ((f = fopen (fname, "r+")) == (FILE *) NULL)
    {
      perror ("fopen(\"r+\")");
    }

  fread (buf, 3, 1, f);
  if (ftell (f) != 3)
    {
      puts ("ftell failed");
      return 1;
    }
  errno = 0;
  if (fseek (f, -10, SEEK_CUR) == 0)
    {
      printf ("fseek() for r+ to before start of file worked!\n");
      result = 1;
    }
  else if (errno != EINVAL)
    {
      printf ("\
fseek() for r+ to before start of file did not set errno to EINVAL.  \
Got %d instead\n",
	 errno);
      result = 1;
    }

  fclose (f);

  return result;
}
