/* Test for vfork functions.
   Copyright (C) 2007-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2007.

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
#include <mcheck.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/wait.h>

static int do_test (void);
static void do_prepare (void);
char *tmpdirname;

#define TEST_FUNCTION do_test ()
#define PREPARE(argc, argv) do_prepare ()
#include "../test-skeleton.c"

static void
run_script (const char *script, char *const argv[])
{
  for (size_t i = 0; i < 5; i++)
    {
      pid_t pid = vfork ();
      if (pid < 0)
	FAIL_EXIT1 ("vfork failed: %m");
      else if (pid == 0)
	{
	  execvp (script, argv);
	  _exit (errno);
	}

      int status;
      if (TEMP_FAILURE_RETRY (waitpid (pid, &status, 0)) != pid)
	FAIL_EXIT1 ("waitpid failed");
      else if (status != 0)
	{
	  if (WIFEXITED (status))
	    FAIL_EXIT1 ("%s failed with status %d\n", script,
			WEXITSTATUS (status));
	  else
	    FAIL_EXIT1 ("%s killed by signal %d\n", script,
			WTERMSIG (status));
	}
    }
}

static int
do_test (void)
{
  mtrace ();

  const char *path = getenv ("PATH");
  if (path == NULL)
    path = "/bin";
  char pathbuf[strlen (tmpdirname) + 1 + strlen (path) + 1];
  strcpy (stpcpy (stpcpy (pathbuf, tmpdirname), ":"), path);
  if (setenv ("PATH", pathbuf, 1) < 0)
    {
      puts ("setenv failed");
      return 1;
    }

  /* Although manual states first argument should be the script name itself,
     current execv{p,e} implementation allows it.  */
  char *argv00[] = { NULL };
  run_script ("script0.sh", argv00);

  char *argv01[] = { (char*) "script0.sh", NULL };
  run_script ("script0.sh", argv01);

  char *argv1[] = { (char *) "script1.sh", (char *) "1", NULL };
  run_script ("script1.sh", argv1);

  char *argv2[] = { (char *) "script2.sh", (char *) "2", NULL };
  run_script ("script2.sh", argv2);

  /* Same as before but with execlp.  */
  for (size_t i = 0; i < 5; i++)
    {
      pid_t pid = vfork ();
      if (pid < 0)
	{
	  printf ("vfork failed: %m\n");
	  return 1;
	}
      else if (pid == 0)
	{
	  execlp ("script2.sh", "script2.sh", "3", NULL);
	  _exit (errno);
	}
      int status;
      if (TEMP_FAILURE_RETRY (waitpid (pid, &status, 0)) != pid)
	{
	  puts ("waitpid failed");
	  return 1;
	}
      else if (status != 0)
	{
	  printf ("script2.sh failed with status %d\n", status);
	  return 1;
	}
    }

  unsetenv ("PATH");
  char *argv4[] = { (char *) "echo", (char *) "script 4", NULL };
  run_script ("echo", argv4);

  return 0;
}

static void
create_script (const char *script, const char *contents, size_t size)
{
  int fd = open (script, O_WRONLY | O_CREAT, 0700);
  if (fd < 0
      || TEMP_FAILURE_RETRY (write (fd, contents, size)) != size
      || fchmod (fd, S_IRUSR | S_IXUSR) < 0)
    FAIL_EXIT1 ("could not write %s\n", script);
  close (fd);
}

static void
do_prepare (void)
{
  size_t len = strlen (test_dir) + sizeof ("/tst-vfork3.XXXXXX");
  tmpdirname = malloc (len);
  if (tmpdirname == NULL)
    FAIL_EXIT1 ("out of memory");
  strcpy (stpcpy (tmpdirname, test_dir), "/tst-vfork3.XXXXXX");

  tmpdirname = mkdtemp (tmpdirname);
  if (tmpdirname == NULL)
    FAIL_EXIT1 ("could not create temporary directory");

  char script0[len + sizeof "/script0.sh"];
  char script1[len + sizeof "/script1.sh"];
  char script2[len + sizeof "/script2.sh"];

  strcpy (stpcpy (script0, tmpdirname), "/script0.sh");
  strcpy (stpcpy (script1, tmpdirname), "/script1.sh");
  strcpy (stpcpy (script2, tmpdirname), "/script2.sh");

  add_temp_file (tmpdirname);
  add_temp_file (script0);
  add_temp_file (script1);
  add_temp_file (script2);

  const char content0[] = "#!/bin/sh\necho empty\n";
  create_script (script0, content0, sizeof content0);

  const char content1[] = "#!/bin/sh\necho script $1\n";
  create_script (script1, content1, sizeof content1);

  const char content2[] = "echo script $1\n";
  create_script (script2, content2, sizeof content2);
}
