/* Main worker function for the test driver.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
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

#include <support/test-driver.h>
#include <support/check.h>
#include <support/temp_file-internal.h>
#include <support/support.h>

#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <malloc.h>
#include <signal.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <xstdio.h>

static const struct option default_options[] =
{
  TEST_DEFAULT_OPTIONS
  { NULL, 0, NULL, 0 }
};

/* Show people how to run the program.  */
static void
usage (const struct option *options)
{
  size_t i;

  printf ("Usage: %s [options]\n"
          "\n"
          "Environment Variables:\n"
          "  TIMEOUTFACTOR          An integer used to scale the timeout\n"
          "  TMPDIR                 Where to place temporary files\n"
          "  TEST_COREDUMPS         Do not disable coredumps if set\n"
          "\n",
          program_invocation_short_name);
  printf ("Options:\n");
  for (i = 0; options[i].name; ++i)
    {
      int indent;

      indent = printf ("  --%s", options[i].name);
      if (options[i].has_arg == required_argument)
        indent += printf (" <arg>");
      printf ("%*s", 25 - indent, "");
      switch (options[i].val)
        {
        case 'v':
          printf ("Increase the output verbosity");
          break;
        case OPT_DIRECT:
          printf ("Run the test directly (instead of forking & monitoring)");
          break;
        case OPT_TESTDIR:
          printf ("Override the TMPDIR env var");
          break;
        }
      printf ("\n");
    }
}

/* The PID of the test process.  */
static pid_t test_pid;

/* The cleanup handler passed to test_main.  */
static void (*cleanup_function) (void);

static void
print_timestamp (const char *what, struct timespec tv)
{
  struct tm tm;
  /* Casts of tv.tv_nsec below are necessary because the type of
     tv_nsec is not literally long int on all supported platforms.  */
  if (gmtime_r (&tv.tv_sec, &tm) == NULL)
    printf ("%s: %lld.%09ld\n",
            what, (long long int) tv.tv_sec, (long int) tv.tv_nsec);
  else
    printf ("%s: %04d-%02d-%02dT%02d:%02d:%02d.%09ld\n",
            what, 1900 + tm.tm_year, tm.tm_mon + 1, tm.tm_mday,
            tm.tm_hour, tm.tm_min, tm.tm_sec, (long int) tv.tv_nsec);
}

/* Timeout handler.  We kill the child and exit with an error.  */
static void
__attribute__ ((noreturn))
signal_handler (int sig)
{
  int killed;
  int status;

  /* Do this first to avoid further interference from the
     subprocess.  */
  struct timespec now;
  clock_gettime (CLOCK_REALTIME, &now);
  struct stat64 st;
  bool st_available = fstat64 (STDOUT_FILENO, &st) == 0 && st.st_mtime != 0;

  assert (test_pid > 1);
  /* Kill the whole process group.  */
  kill (-test_pid, SIGKILL);
  /* In case setpgid failed in the child, kill it individually too.  */
  kill (test_pid, SIGKILL);

  /* Wait for it to terminate.  */
  int i;
  for (i = 0; i < 5; ++i)
    {
      killed = waitpid (test_pid, &status, WNOHANG|WUNTRACED);
      if (killed != 0)
        break;

      /* Delay, give the system time to process the kill.  If the
         nanosleep() call return prematurely, all the better.  We
         won't restart it since this probably means the child process
         finally died.  */
      struct timespec ts;
      ts.tv_sec = 0;
      ts.tv_nsec = 100000000;
      nanosleep (&ts, NULL);
    }
  if (killed != 0 && killed != test_pid)
    {
      printf ("Failed to kill test process: %m\n");
      exit (1);
    }

  if (cleanup_function != NULL)
    cleanup_function ();

  if (sig == SIGINT)
    {
      signal (sig, SIG_DFL);
      raise (sig);
    }

  if (killed == 0 || (WIFSIGNALED (status) && WTERMSIG (status) == SIGKILL))
    puts ("Timed out: killed the child process");
  else if (WIFSTOPPED (status))
    printf ("Timed out: the child process was %s\n",
            strsignal (WSTOPSIG (status)));
  else if (WIFSIGNALED (status))
    printf ("Timed out: the child process got signal %s\n",
            strsignal (WTERMSIG (status)));
  else
    printf ("Timed out: killed the child process but it exited %d\n",
            WEXITSTATUS (status));

  print_timestamp ("Termination time", now);
  if (st_available)
    print_timestamp ("Last write to standard output", st.st_mtim);

  /* Exit with an error.  */
  exit (1);
}

/* This must be volatile as it will be modified by the debugger.  */
static volatile int wait_for_debugger = 0;

/* Run test_function or test_function_argv.  */
static int
run_test_function (int argc, char **argv, const struct test_config *config)
{
  const char *wfd = getenv("WAIT_FOR_DEBUGGER");
  if (wfd != NULL)
    wait_for_debugger = atoi (wfd);
  if (wait_for_debugger)
    {
      pid_t mypid;
      FILE *gdb_script;
      char *gdb_script_name;
      int inside_container = 0;

      mypid = getpid();
      if (mypid < 3)
	{
	  const char *outside_pid = getenv("PID_OUTSIDE_CONTAINER");
	  if (outside_pid)
	    {
	      mypid = atoi (outside_pid);
	      inside_container = 1;
	    }
	}

      gdb_script_name = (char *) xmalloc (strlen (argv[0]) + strlen (".gdb") + 1);
      sprintf (gdb_script_name, "%s.gdb", argv[0]);
      gdb_script = xfopen (gdb_script_name, "w");

      fprintf (stderr, "Waiting for debugger, test process is pid %d\n", mypid);
      fprintf (stderr, "gdb -x %s\n", gdb_script_name);
      if (inside_container)
	fprintf (gdb_script, "set sysroot %s/testroot.root\n", support_objdir_root);
      fprintf (gdb_script, "file\n");
      fprintf (gdb_script, "file %s\n", argv[0]);
      fprintf (gdb_script, "symbol-file %s\n", argv[0]);
      fprintf (gdb_script, "exec-file %s\n", argv[0]);
      fprintf (gdb_script, "attach %ld\n", (long int) mypid);
      fprintf (gdb_script, "set wait_for_debugger = 0\n");
      fclose (gdb_script);
      free (gdb_script_name);
    }

  /* Wait for the debugger to set wait_for_debugger to zero.  */
  while (wait_for_debugger)
    usleep (1000);

  if (config->test_function != NULL)
    return config->test_function ();
  else if (config->test_function_argv != NULL)
    return config->test_function_argv (argc, argv);
  else
    {
      printf ("error: no test function defined\n");
      exit (1);
    }
}

static bool test_main_called;

const char *test_dir = NULL;
unsigned int test_verbose = 0;

/* If test failure reporting has been linked in, it may contribute
   additional test failures.  */
static int
adjust_exit_status (int status)
{
  if (support_report_failure != NULL)
    return support_report_failure (status);
  return status;
}

int
support_test_main (int argc, char **argv, const struct test_config *config)
{
  if (test_main_called)
    {
      printf ("error: test_main called for a second time\n");
      exit (1);
    }
  test_main_called = true;
  const struct option *options;
  if (config->options != NULL)
    options = config->options;
  else
    options = default_options;

  cleanup_function = config->cleanup_function;

  int direct = 0;       /* Directly call the test function?  */
  int status;
  int opt;
  unsigned int timeoutfactor = 1;
  pid_t termpid;

  /* If we're debugging the test, we need to disable timeouts and use
     the initial pid (esp if we're running inside a container).  */
  if (getenv("WAIT_FOR_DEBUGGER") != NULL)
    direct = 1;

  if (!config->no_mallopt)
    {
      /* Make uses of freed and uninitialized memory known.  Do not
         pull in a definition for mallopt if it has not been defined
         already.  */
      extern __typeof__ (mallopt) mallopt __attribute__ ((weak));
      if (mallopt != NULL)
        mallopt (M_PERTURB, 42);
    }

  while ((opt = getopt_long (argc, argv, config->optstring, options, NULL))
	 != -1)
    switch (opt)
      {
      case '?':
        usage (options);
        exit (1);
      case 'v':
        ++test_verbose;
        break;
      case OPT_DIRECT:
        direct = 1;
        break;
      case OPT_TESTDIR:
        test_dir = optarg;
        break;
      default:
        if (config->cmdline_function != NULL)
          config->cmdline_function (opt);
      }

  /* If set, read the test TIMEOUTFACTOR value from the environment.
     This value is used to scale the default test timeout values. */
  char *envstr_timeoutfactor = getenv ("TIMEOUTFACTOR");
  if (envstr_timeoutfactor != NULL)
    {
      char *envstr_conv = envstr_timeoutfactor;
      unsigned long int env_fact;

      env_fact = strtoul (envstr_timeoutfactor, &envstr_conv, 0);
      if (*envstr_conv == '\0' && envstr_conv != envstr_timeoutfactor)
        timeoutfactor = MAX (env_fact, 1);
    }

  /* Set TMPDIR to specified test directory.  */
  if (test_dir != NULL)
    {
      setenv ("TMPDIR", test_dir, 1);

      if (chdir (test_dir) < 0)
        {
          printf ("chdir: %m\n");
          exit (1);
        }
    }
  else
    {
      test_dir = getenv ("TMPDIR");
      if (test_dir == NULL || test_dir[0] == '\0')
        test_dir = "/tmp";
    }
  if (support_set_test_dir != NULL)
    support_set_test_dir (test_dir);

  int timeout = config->timeout;
  if (timeout == 0)
    timeout =  DEFAULT_TIMEOUT;

  /* Make sure we see all message, even those on stdout.  */
  if (!config->no_setvbuf)
    setvbuf (stdout, NULL, _IONBF, 0);

  /* Make sure temporary files are deleted.  */
  if (support_delete_temp_files != NULL)
      atexit (support_delete_temp_files);

  /* Correct for the possible parameters.  */
  argv[optind - 1] = argv[0];
  argv += optind - 1;
  argc -= optind - 1;

  /* Call the initializing function, if one is available.  */
  if (config->prepare_function != NULL)
    config->prepare_function (argc, argv);

  const char *envstr_direct = getenv ("TEST_DIRECT");
  if (envstr_direct != NULL)
    {
      FILE *f = fopen (envstr_direct, "w");
      if (f == NULL)
        {
          printf ("cannot open TEST_DIRECT output file '%s': %m\n",
                  envstr_direct);
          exit (1);
        }

      fprintf (f, "timeout=%u\ntimeoutfactor=%u\n",
               config->timeout, timeoutfactor);
      if (config->expected_status != 0)
        fprintf (f, "exit=%u\n", config->expected_status);
      if (config->expected_signal != 0)
        fprintf (f, "signal=%s\n", strsignal (config->expected_signal));

      if (support_print_temp_files != NULL)
        support_print_temp_files (f);

      fclose (f);
      direct = 1;
    }

  bool disable_coredumps;
  {
    const char *coredumps = getenv ("TEST_COREDUMPS");
    disable_coredumps = coredumps == NULL || coredumps[0] == '\0';
  }

  /* If we are not expected to fork run the function immediately.  */
  if (direct)
    return adjust_exit_status (run_test_function (argc, argv, config));

  /* Set up the test environment:
     - prevent core dumps
     - set up the timer
     - fork and execute the function.  */

  test_pid = fork ();
  if (test_pid == 0)
    {
      /* This is the child.  */
      if (disable_coredumps)
        {
          /* Try to avoid dumping core.  This is necessary because we
             run the test from the source tree, and the coredumps
             would end up there (and not in the build tree).  */
          struct rlimit core_limit;
          core_limit.rlim_cur = 0;
          core_limit.rlim_max = 0;
          setrlimit (RLIMIT_CORE, &core_limit);
        }

      /* We put the test process in its own pgrp so that if it bogusly
         generates any job control signals, they won't hit the whole build.  */
      if (setpgid (0, 0) != 0)
        printf ("Failed to set the process group ID: %m\n");

      /* Execute the test function and exit with the return value.   */
      exit (run_test_function (argc, argv, config));
    }
  else if (test_pid < 0)
    {
      printf ("Cannot fork test program: %m\n");
      exit (1);
    }

  /* Set timeout.  */
  signal (SIGALRM, signal_handler);
  alarm (timeout * timeoutfactor);

  /* Make sure we clean up if the wrapper gets interrupted.  */
  signal (SIGINT, signal_handler);

  /* Wait for the regular termination.  */
  termpid = TEMP_FAILURE_RETRY (waitpid (test_pid, &status, 0));
  if (termpid == -1)
    {
      printf ("Waiting for test program failed: %m\n");
      exit (1);
    }
  if (termpid != test_pid)
    {
      printf ("Oops, wrong test program terminated: expected %ld, got %ld\n",
              (long int) test_pid, (long int) termpid);
      exit (1);
    }

  /* Process terminated normaly without timeout etc.  */
  if (WIFEXITED (status))
    {
      if (config->expected_status == 0)
        {
          if (config->expected_signal == 0)
            /* Exit with the return value of the test.  */
            return adjust_exit_status (WEXITSTATUS (status));
          else
            {
              printf ("Expected signal '%s' from child, got none\n",
                      strsignal (config->expected_signal));
              exit (1);
            }
        }
      else
        {
          /* Non-zero exit status is expected */
          if (WEXITSTATUS (status) != config->expected_status)
            {
              printf ("Expected status %d, got %d\n",
                      config->expected_status, WEXITSTATUS (status));
              exit (1);
            }
        }
      return adjust_exit_status (0);
    }
  /* Process was killed by timer or other signal.  */
  else
    {
      if (config->expected_signal == 0)
        {
          printf ("Didn't expect signal from child: got `%s'\n",
                  strsignal (WTERMSIG (status)));
          exit (1);
        }
      else if (WTERMSIG (status) != config->expected_signal)
        {
          printf ("Incorrect signal from child: got `%s', need `%s'\n",
                  strsignal (WTERMSIG (status)),
                  strsignal (config->expected_signal));
          exit (1);
        }

      return adjust_exit_status (0);
    }
}
