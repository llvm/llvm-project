/* Basic tests for pldd program.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <stdbool.h>

#include <array_length.h>
#include <gnu/lib-names.h>

#include <support/subprocess.h>
#include <support/capture_subprocess.h>
#include <support/check.h>
#include <support/support.h>
#include <support/xptrace.h>
#include <support/xunistd.h>
#include <sys/mman.h>
#include <errno.h>
#include <signal.h>

static void
target_process (void *arg)
{
  pause ();
}

static void
pldd_process (void *arg)
{
  pid_t *target_pid_ptr = (pid_t *) arg;

  /* Create a copy of current test to check with pldd.  As the
     target_process is a child of this pldd_process, pldd is also able
     to attach to target_process if YAMA is configured to 1 =
     "restricted ptrace".  */
  struct support_subprocess target = support_subprocess (target_process, NULL);

  /* Store the pid of target-process as do_test needs it in order to
     e.g. terminate it at end of the test.  */
  *target_pid_ptr = target.pid;

  /* Three digits per byte plus null terminator.  */
  char pid[3 * sizeof (uint32_t) + 1];
  snprintf (pid, array_length (pid), "%d", target.pid);

  char *prog = xasprintf ("%s/pldd", support_bindir_prefix);

  /* Run pldd and use the pid of target_process as argument.  */
  execve (prog, (char *const []) { (char *) prog, pid, NULL },
	  (char *const []) { NULL });

  FAIL_EXIT1 ("Returned from execve: errno=%d=%m\n", errno);
}

/* The test runs in a container because pldd does not support tracing
   a binary started by the loader iself (as with testrun.sh).  */

static bool
in_str_list (const char *libname, const char *const strlist[])
{
  for (const char *const *str = strlist; *str != NULL; str++)
    if (strcmp (libname, *str) == 0)
      return true;
  return false;
}

static int
do_test (void)
{
  /* Check if our subprocess can be debugged with ptrace.  */
  {
    int ptrace_scope = support_ptrace_scope ();
    if (ptrace_scope >= 2)
      FAIL_UNSUPPORTED ("/proc/sys/kernel/yama/ptrace_scope >= 2");
  }

  pid_t *target_pid_ptr = (pid_t *) xmmap (NULL, sizeof (pid_t),
					   PROT_READ | PROT_WRITE,
					   MAP_SHARED | MAP_ANONYMOUS, -1);

  /* Run 'pldd' on test subprocess which will be created in pldd_process.
     The pid of the subprocess will be written to target_pid_ptr.  */
  struct support_capture_subprocess pldd;
  pldd = support_capture_subprocess (pldd_process, target_pid_ptr);
  support_capture_subprocess_check (&pldd, "pldd", 0, sc_allow_stdout);

  /* Check 'pldd' output.  The test is expected to be linked against only
     loader and libc.  */
  {
    pid_t pid;
    char buffer[512];
#define STRINPUT(size) "%" # size "s"

    FILE *out = fmemopen (pldd.out.buffer, pldd.out.length, "r");
    TEST_VERIFY (out != NULL);

    /* First line is in the form of <pid>: <full path of executable>  */
    TEST_COMPARE (fscanf (out, "%u: " STRINPUT (512), &pid, buffer), 2);

    TEST_COMPARE (pid, *target_pid_ptr);
    TEST_COMPARE (strcmp (basename (buffer), "tst-pldd"), 0);

    /* It expects only one loader and libc loaded by the program.  */
    bool interpreter_found = false, libc_found = false;
    while (fgets (buffer, array_length (buffer), out) != NULL)
      {
	/* Ignore vDSO.  */
	if (buffer[0] != '/')
	  continue;

	/* Remove newline so baseline (buffer) can compare against the
	   LD_SO and LIBC_SO macros unmodified.  */
	if (buffer[strlen(buffer)-1] == '\n')
	  buffer[strlen(buffer)-1] = '\0';

	const char *libname = basename (buffer);

	/* It checks for default names in case of build configure with
	   --enable-hardcoded-path-in-tests (BZ #24506).  */
	if (in_str_list (libname,
			 (const char *const []) { "ld.so", LD_SO, NULL }))
	  {
	    TEST_COMPARE (interpreter_found, false);
	    interpreter_found = true;
	    continue;
	  }

	if (in_str_list (libname,
			 (const char *const []) { "libc.so", LIBC_SO, NULL }))
	  {
	    TEST_COMPARE (libc_found, false);
	    libc_found = true;
	    continue;
	  }
      }
    TEST_COMPARE (interpreter_found, true);
    TEST_COMPARE (libc_found, true);

    fclose (out);
  }

  support_capture_subprocess_free (&pldd);
  if (kill (*target_pid_ptr, SIGKILL) != 0)
    FAIL_EXIT1 ("Unable to kill target_process: errno=%d=%m\n", errno);
  xmunmap (target_pid_ptr, sizeof (pid_t));

  return 0;
}

#include <support/test-driver.c>
