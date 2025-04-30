/* Test for bug in fflush synchronization behavior.  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static char *fname;

static void prepare (void);
#define PREPARE(argc, argv) prepare ()


#define TEST_FUNCTION do_test ()
static int do_test (void);
#include "../test-skeleton.c"


static void
prepare (void)
{
  int fd = create_temp_file ("bug-mmap-fflush.", &fname);
  if (fd == -1)
    exit (3);
  /* We don't need the descriptor.  */
  close (fd);
}


static int
do_test (void)
{
  FILE *f;
  off_t o;
  char buffer[1024];

  snprintf (buffer, sizeof (buffer), "echo 'From foo@bar.com' > %s", fname);
  system (buffer);
  f = fopen (fname, "r");
  fseek (f, 0, SEEK_END);
  o = ftello (f);
  fseek (f, 0, SEEK_SET);
  fflush (f);
  snprintf (buffer, sizeof (buffer), "echo 'From bar@baz.edu' >> %s", fname);
  system (buffer);
  fseek (f, o, SEEK_SET);
  if (fgets (buffer, 1024, f) == NULL)
    exit (1);
  if (strncmp (buffer, "From ", 5) != 0)
    exit (1);
  fclose (f);
  exit (0);
}
