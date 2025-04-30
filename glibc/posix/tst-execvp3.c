#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>


static void do_prepare (void);
#define PREPARE(argc, argv) do_prepare ()
static int do_test (void);
#define TEST_FUNCTION do_test ()

#include "../test-skeleton.c"

#ifndef EXECVP
# define EXECVP(file, argv)  execvp (file, argv)
#endif

static char *fname;

static void
do_prepare (void)
{
  int fd = create_temp_file ("testscript", &fname);
  dprintf (fd, "echo foo\n");
  fchmod (fd, 0700);
  close (fd);
}


static int
do_test (void)
{
  if  (setenv ("PATH", test_dir, 1) != 0)
    {
      puts ("setenv failed");
      return 1;
    }

  char *argv[] = { fname, NULL };
  EXECVP (basename (fname), argv);

  /* If we come here, the execvp call failed.  */
  return 1;
}
