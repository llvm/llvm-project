/* Minimal /bin/sh for in-container use.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#define _FILE_OFFSET_BITS 64

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sys/types.h>
#include <dirent.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/fcntl.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <stdarg.h>
#include <sys/sysmacros.h>
#include <ctype.h>
#include <utime.h>
#include <errno.h>
#include <error.h>

#include <support/support.h>

/* Design considerations

 General rule: optimize for developer time, not run time.

 Specifically:

 * Don't worry about slow algorithms
 * Don't worry about free'ing memory
 * Don't implement anything the testsuite doesn't need.
 * Line and argument counts are limited, see below.

*/

#define MAX_ARG_COUNT 100
#define MAX_LINE_LENGTH 1000

/* Debugging is enabled via --debug, which must be the first argument.  */
static int debug_mode = 0;
#define dprintf if (debug_mode) fprintf

/* Emulate the "/bin/true" command.  Arguments are ignored.  */
static int
true_func (char **argv)
{
  return 0;
}

/* Emulate the "/bin/echo" command.  Options are ignored, arguments
   are printed to stdout.  */
static int
echo_func (char **argv)
{
  int i;

  for (i = 0; argv[i]; i++)
    {
      if (i > 0)
	putchar (' ');
      fputs (argv[i], stdout);
    }
  putchar ('\n');

  return 0;
}

/* Emulate the "/bin/cp" command.  Options are ignored.  Only copies
   one source file to one destination file.  Directory destinations
   are not supported.  */
static int
copy_func (char **argv)
{
  char *sname = argv[0];
  char *dname = argv[1];
  int sfd = -1, dfd = -1;
  struct stat st;
  int ret = 1;

  sfd = open (sname, O_RDONLY);
  if (sfd < 0)
    {
      fprintf (stderr, "cp: unable to open %s for reading: %s\n",
	       sname, strerror (errno));
      return 1;
    }

  if (fstat (sfd, &st) < 0)
    {
      fprintf (stderr, "cp: unable to fstat %s: %s\n",
	       sname, strerror (errno));
      goto out;
    }

  dfd = open (dname, O_WRONLY | O_TRUNC | O_CREAT, 0600);
  if (dfd < 0)
    {
      fprintf (stderr, "cp: unable to open %s for writing: %s\n",
	       dname, strerror (errno));
      goto out;
    }

  if (support_copy_file_range (sfd, 0, dfd, 0, st.st_size, 0) != st.st_size)
    {
      fprintf (stderr, "cp: cannot copy file %s to %s: %s\n",
	       sname, dname, strerror (errno));
      goto out;
    }

  ret = 0;
  chmod (dname, st.st_mode & 0777);

out:
  if (sfd >= 0)
    close (sfd);
  if (dfd >= 0)
    close (dfd);

  return ret;

}

/* Emulate the 'exit' builtin.  The exit value is optional.  */
static int
exit_func (char **argv)
{
  int exit_val = 0;

  if (argv[0] != 0)
    exit_val = atoi (argv[0]) & 0xff;
  exit (exit_val);
  return 0;
}

/* Emulate the "/bin/kill" command.  Options are ignored.  */
static int
kill_func (char **argv)
{
  int signum = SIGTERM;
  int i;

  for (i = 0; argv[i]; i++)
    {
      pid_t pid;
      if (strcmp (argv[i], "$$") == 0)
	pid = getpid ();
      else
	pid = atoi (argv[i]);
      kill (pid, signum);
    }
  return 0;
}

/* This is a list of all the built-in commands we understand.  */
static struct {
  const char *name;
  int (*func) (char **argv);
} builtin_funcs[] = {
  { "true", true_func },
  { "echo", echo_func },
  { "cp", copy_func },
  { "exit", exit_func },
  { "kill", kill_func },
  { NULL, NULL }
};

/* Run one tokenized command.  argv[0] is the command.  argv is
   NULL-terminated.  */
static void
run_command_array (char **argv)
{
  int i, j;
  pid_t pid;
  int status;
  int (*builtin_func) (char **args);

  if (argv[0] == NULL)
    return;

  builtin_func = NULL;

  int new_stdin = 0;
  int new_stdout = 1;
  int new_stderr = 2;

  dprintf (stderr, "run_command_array starting\n");
  for (i = 0; argv[i]; i++)
    dprintf (stderr, "   argv [%d] `%s'\n", i, argv[i]);

  for (j = i = 0; argv[i]; i++)
    {
      if (strcmp (argv[i], "<") == 0 && argv[i + 1])
	{
	  new_stdin = open (argv[i + 1], O_WRONLY|O_CREAT|O_TRUNC, 0777);
	  ++i;
	  continue;
	}
      if (strcmp (argv[i], ">") == 0 && argv[i + 1])
	{
	  new_stdout = open (argv[i + 1], O_WRONLY|O_CREAT|O_TRUNC, 0777);
	  ++i;
	  continue;
	}
      if (strcmp (argv[i], ">>") == 0 && argv[i + 1])
	{
	  new_stdout = open (argv[i + 1], O_WRONLY|O_CREAT|O_APPEND, 0777);
	  ++i;
	  continue;
	}
      if (strcmp (argv[i], "2>") == 0 && argv[i + 1])
	{
	  new_stderr = open (argv[i + 1], O_WRONLY|O_CREAT|O_TRUNC, 0777);
	  ++i;
	  continue;
	}
      argv[j++] = argv[i];
    }
  argv[j] = NULL;


  for (i = 0; builtin_funcs[i].name != NULL; i++)
    if (strcmp (argv[0], builtin_funcs[i].name) == 0)
       builtin_func = builtin_funcs[i].func;

  dprintf (stderr, "builtin %p argv0 `%s'\n", builtin_func, argv[0]);

  pid = fork ();
  if (pid < 0)
    {
      fprintf (stderr, "sh: fork failed\n");
      exit (1);
    }

  if (pid == 0)
    {
      if (new_stdin != 0)
	{
	  dup2 (new_stdin, 0);
	  close (new_stdin);
	}
      if (new_stdout != 1)
	{
	  dup2 (new_stdout, 1);
	  close (new_stdout);
	}
      if (new_stderr != 2)
	{
	  dup2 (new_stderr, 2);
	  close (new_stderr);
	}

      if (builtin_func != NULL)
	exit (builtin_func (argv + 1));

      execvp (argv[0], argv);

      fprintf (stderr, "sh: execing %s failed: %s",
	       argv[0], strerror (errno));
      exit (127);
    }

  waitpid (pid, &status, 0);

  dprintf (stderr, "exiting run_command_array\n");

  if (WIFEXITED (status))
    {
      int rv = WEXITSTATUS (status);
      if (rv)
	exit (rv);
    }
  else if (WIFSIGNALED (status))
    {
      int sig = WTERMSIG (status);
      raise (sig);
    }
  else
    exit (1);
}

/* Run one command-as-a-string, by tokenizing it.  Limited to
   MAX_ARG_COUNT arguments.  Simple substitution is done of $1 to $9
   (as whole separate tokens) from iargs[].  Quoted strings work if
   the quotes wrap whole tokens; i.e. "foo bar" but not foo" bar".  */
static void
run_command_string (const char *cmdline, const char **iargs)
{
  char *args[MAX_ARG_COUNT+1];
  int ap = 0;
  const char *start, *end;
  int nargs;

  for (nargs = 0; iargs[nargs] != NULL; ++nargs)
    ;

  dprintf (stderr, "run_command_string starting: '%s'\n", cmdline);

  while (ap < MAX_ARG_COUNT)
    {
      /* If the argument is quoted, this is the quote character, else NUL.  */
      int in_quote = 0;

      /* Skip whitespace up to the next token.  */
      while (*cmdline && isspace (*cmdline))
	cmdline ++;
      if (*cmdline == 0)
	break;

      start = cmdline;
      /* Check for quoted argument.  */
      in_quote = (*cmdline == '\'' || *cmdline == '"') ? *cmdline : 0;

      /* Skip to end of token; either by whitespace or matching quote.  */
      dprintf (stderr, "in_quote %d\n", in_quote);
      while (*cmdline
	     && (!isspace (*cmdline) || in_quote))
	{
	  if (*cmdline == in_quote
	      && cmdline != start)
	    in_quote = 0;
	  dprintf (stderr, "[%c]%d ", *cmdline, in_quote);
	  cmdline ++;
	}
      dprintf (stderr, "\n");

      /* Allocate space for this token and store it in args[].  */
      end = cmdline;
      dprintf (stderr, "start<%s> end<%s>\n", start, end);
      args[ap] = (char *) xmalloc (end - start + 1);
      memcpy (args[ap], start, end - start);
      args[ap][end - start] = 0;

      /* Strip off quotes, if found.  */
      dprintf (stderr, "args[%d] = <%s>\n", ap, args[ap]);
      if (args[ap][0] == '\''
	  && args[ap][strlen (args[ap])-1] == '\'')
	{
	  args[ap][strlen (args[ap])-1] = 0;
	  args[ap] ++;
	}

      else if (args[ap][0] == '"'
	  && args[ap][strlen (args[ap])-1] == '"')
	{
	  args[ap][strlen (args[ap])-1] = 0;
	  args[ap] ++;
	}

      /* Replace positional parameters like $4.  */
      else if (args[ap][0] == '$'
	       && isdigit (args[ap][1])
	       && args[ap][2] == 0)
	{
	  int a = args[ap][1] - '1';
	  if (0 <= a && a < nargs)
	    args[ap] = strdup (iargs[a]);
	}

      ap ++;

      if (*cmdline == 0)
	break;
    }

  /* Lastly, NULL terminate the array and run it.  */
  args[ap] = NULL;
  run_command_array (args);
}

/* Run a script by reading lines and passing them to the above
   function.  */
static void
run_script (const char *filename, const char **args)
{
  char line[MAX_LINE_LENGTH + 1];
  dprintf (stderr, "run_script starting: '%s'\n", filename);
  FILE *f = fopen (filename, "r");
  if (f == NULL)
    {
      fprintf (stderr, "sh: %s: %s\n", filename, strerror (errno));
      exit (1);
    }
  while (fgets (line, sizeof (line), f) != NULL)
    {
      if (line[0] == '#')
	{
	  dprintf (stderr, "comment: %s\n", line);
	  continue;
	}
      run_command_string (line, args);
    }
  fclose (f);
}

int
main (int argc, const char **argv)
{
  int i;

  if (strcmp (argv[1], "--debug") == 0)
    {
      debug_mode = 1;
      --argc;
      ++argv;
    }

  dprintf (stderr, "container-sh starting:\n");
  for (i = 0; i < argc; i++)
    dprintf (stderr, "  argv[%d] is `%s'\n", i, argv[i]);

  if (strcmp (argv[1], "-c") == 0)
    run_command_string (argv[2], argv+3);
  else
    run_script (argv[1], argv+2);

  dprintf (stderr, "normal exit 0\n");
  return 0;
}
