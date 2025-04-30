/* General tests for execpve.
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

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <wait.h>


/* Nonzero if the program gets called via `exec'.  */
static int restart;


#define CMDLINE_OPTIONS \
  { "restart", no_argument, &restart, 1 },

/* Prototype for our test function.  */
extern void do_prepare (int argc, char *argv[]);
extern int do_test (int argc, char *argv[]);

#include "../test-skeleton.c"

#define EXECVPE_KEY    "EXECVPE_ENV"
#define EXECVPE_VALUE  "execvpe_test"


static int
handle_restart (void)
{
  /* First check if only one variable is passed on execvpe.  */
  int env_count = 0;
  for (char **e = environ; *e != NULL; ++e)
    if (++env_count == INT_MAX)
      {
	printf ("Environment variable number overflow");
	exit (EXIT_FAILURE);
      }
  if (env_count != 1)
    {
      printf ("Wrong number of environment variables");
      exit (EXIT_FAILURE);
    }

  /* Check if the combinarion os "EXECVPE_ENV=execvpe_test"  */
  const char *env = getenv (EXECVPE_KEY);
  if (env == NULL)
    {
      printf ("Test environment variable not found");
      exit (EXIT_FAILURE);
    }

  if (strncmp (env, EXECVPE_VALUE, sizeof (EXECVPE_VALUE)))
    {
      printf ("Test environment variable with wrong value");
      exit (EXIT_FAILURE);
    }

  return 0;
}


int
do_test (int argc, char *argv[])
{
  pid_t pid;
  int status;

  /* We must have
     - one or four parameters left if called initially
       + path for ld.so		optional
       + "--library-path"	optional
       + the library path	optional
       + the application name

    if --enable-hardcoded-path-in-tests is used, just
      + the application name
  */

  if (restart)
    {
      if (argc != 1)
	{
	  printf ("Wrong number of arguments (%d) in restart\n", argc);
	  exit (EXIT_FAILURE);
	}

      return handle_restart ();
    }

  if (argc != 2 && argc != 5)
    {
      printf ("Wrong number of arguments (%d)\n", argc);
      exit (EXIT_FAILURE);
    }

  /* We want to test the `execvpe' function.  To do this we restart the
     program with an additional parameter.  */
  pid = fork ();
  if (pid == 0)
    {
      /* This is the child.  Construct the command line.  */

      /* We cast here to char* because the test itself does not modify neither
	 the argument nor the environment list.  */
      char *envs[] = { (char*)(EXECVPE_KEY "=" EXECVPE_VALUE), NULL };
      if (argc == 5)
	{
	  char *args[] = { argv[1], argv[2], argv[3], argv[4],
			   (char *) "--direct", (char *) "--restart", NULL };
	  execvpe (args[0], args, envs);
	}
      else
	{
	  char *args[] = { argv[0],
			   (char *) "--direct", (char *) "--restart", NULL };
	  execvpe (args[0], args, envs);
	}

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
  status = WEXITSTATUS (status);

  return status;
}
