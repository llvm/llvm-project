/* Check execvpe script argument handling.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/param.h>

static char *fname1;
static char *fname2;
static char *logname;

static void do_prepare (void);
#define PREPARE(argc, argv) do_prepare ()
static int do_test (void);
#define TEST_FUNCTION do_test ()

#include "../test-skeleton.c"

static void
do_prepare (void)
{
  int logfd = create_temp_file ("logfile", &logname);
  close (logfd);

  int fd1 = create_temp_file ("testscript", &fname1);
  dprintf (fd1, "echo foo $1 $2 $3 > %s\n", logname);
  fchmod (fd1, 0700);
  close (fd1);

  int fd2 = create_temp_file ("testscript", &fname2);
  dprintf (fd2, "echo foo > %s\n", logname);
  fchmod (fd2, 0700);
  close (fd2);
}

static int
run_script (const char *fname, char *args[])
{
  /* We want to test the `execvpe' function.  To do this we restart the
     program with an additional parameter.  */
  int status;
  pid_t pid = fork ();
  if (pid == 0)
    {
      execvpe (fname, args, NULL);

      puts ("Cannot exec");
      exit (EXIT_FAILURE);
    }
  else if (pid == (pid_t) -1)
    {
      puts ("Cannot fork");
      return 1;
    }

  /* Wait for the child.  */
  if (waitpid (pid, &status, 0) != pid)
    {
      puts ("Wrong child");
      return 1;
    }

  if (WTERMSIG (status) != 0)
    {
      puts ("Child terminated incorrectly");
      return 1;
    }

  return 0;
}

static int
check_output (const char *expected)
{
  /* Check log output.  */
  FILE *arq = fopen (logname, "r");
  if (arq == NULL)
    {
      puts ("Error opening output file");
      return 1;
    }

  char line[128];
  if (fgets (line, sizeof (line), arq) == NULL)
    {
      puts ("Error reading output file");
      return 1;
    }
  fclose (arq);

  if (strcmp (line, expected) != 0)
    {
      puts ("Output file different than expected");
      return 1;
    }

  return 0;
}

static int
do_test (void)
{
  if  (setenv ("PATH", test_dir, 1) != 0)
    {
      puts ("setenv failed");
      return 1;
    }

  /* First check resulting script run with some arguments results in correct
     output file.  */
  char *args1[] = { fname1, (char*) "1", (char *) "2", (char *) "3", NULL };
  if (run_script (fname1,args1))
    return 1;
  if (check_output ("foo 1 2 3\n"))
    return 1;

  /* Same as before but with an expected empty argument list.  */
  char *args2[] = { fname2, NULL };
  if (run_script (fname2, args2))
    return 1;
  if (check_output ("foo\n"))
    return 1;

  /* Same as before but with an empty argument list.  */
  char *args3[] = { NULL };
  if (run_script (fname2, args3))
    return 1;
  if (check_output ("foo\n"))
    return 1;

  return 0;
}
