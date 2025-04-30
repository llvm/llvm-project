#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>


static void prepare (int argc, char *argv[]);
static int do_test (void);
#define PREPARE(argc, argv) prepare (argc, argv)
#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"


static char *copy;

static void
prepare (int argc, char *argv[])
{
  char *buf;
  int off;
  asprintf (&buf, "cp %s %n%s-copy", argv[0], &off, argv[0]);
  if (buf == NULL)
    {
      puts ("asprintf  failed");
      exit (1);
    }
  if (system (buf) != 0)
    {
      puts ("system  failed");
      exit (1);
    }

  /* Make it not executable.  */
  copy = buf + off;
  if (chmod (copy, 0666) != 0)
    {
      puts ("chmod  failed");
      exit (1);
    }

  add_temp_file (copy);
}


static int
do_test (void)
{
  char *argv[] = { copy, NULL };

  errno = 0;
  execv (copy, argv);

  if (errno != EACCES)
    {
      printf ("errno = %d (%m), expected EACCES\n", errno);
      return 1;
    }

  return 0;
}
