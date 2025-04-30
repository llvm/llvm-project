/* Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

/* fprintf is a cancellation point, but getopt is not supposed to be a
   cancellation point, even when it prints error messages.  */

/* Note: getopt.h must be included first in this file, so we get the
   GNU getopt rather than the POSIX one.  */
#include <getopt.h>

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include <fcntl.h>
#include <pthread.h>
#include <unistd.h>

#include <support/support.h>
#include <support/temp_file.h>
#include <support/xthread.h>

static bool
check_stderr (bool expect_errmsg, FILE *stderr_trapped)
{
  static char *lineptr = 0;
  static size_t linesz = 0;

  bool got_errmsg = false;
  rewind (stderr_trapped);
  while (getline (&lineptr, &linesz, stderr_trapped) > 0)
    {
      got_errmsg = true;
      fputs (lineptr, stdout);
    }
  rewind (stderr_trapped);
  ftruncate (fileno (stderr_trapped), 0);
  return got_errmsg == expect_errmsg;
}

struct test_short
{
  const char *label;
  const char *opts;
  const char *const argv[8];
  int argc;
  bool expect_errmsg;
};

struct test_long
{
  const char *label;
  const char *opts;
  const struct option longopts[4];
  const char *const argv[8];
  int argc;
  bool expect_errmsg;
};

#define DEFINE_TEST_DRIVER(test_type, getopt_call)			\
  struct test_type##_tdata						\
  {									\
    pthread_mutex_t *sync;						\
    const struct test_type *tcase;					\
    bool ok;								\
  };									\
									\
  static void *								\
  test_type##_threadproc (void *data)					\
  {									\
    struct test_type##_tdata *tdata = data;				\
    const struct test_type *tc = tdata->tcase;				\
									\
    xpthread_mutex_lock (tdata->sync);					\
    xpthread_mutex_unlock (tdata->sync);				\
									\
    /* At this point, this thread has a cancellation pending.		\
       We should still be able to get all the way through a getopt	\
       loop without being cancelled.					\
       Setting optind to 0 forces getopt to reinitialize itself.  */	\
    optind = 0;								\
    opterr = 1;								\
    optopt = 0;								\
    while (getopt_call != -1)						\
      ;									\
    tdata->ok = true;							\
									\
    pthread_testcancel();						\
    return 0;								\
  }									\
									\
  static bool								\
  do_##test_type (const struct test_type *tcase, FILE *stderr_trapped)	\
  {									\
    pthread_mutex_t sync;						\
    struct test_type##_tdata tdata;					\
									\
    printf("begin: %s\n", tcase->label);				\
									\
    xpthread_mutex_init (&sync, 0);					\
    xpthread_mutex_lock (&sync);					\
									\
    tdata.sync = &sync;							\
    tdata.tcase = tcase;						\
    tdata.ok = false;							\
									\
    pthread_t thr = xpthread_create (0, test_type##_threadproc,		\
				     (void *)&tdata);			\
    xpthread_cancel (thr);						\
    xpthread_mutex_unlock (&sync);					\
    void *rv = xpthread_join (thr);					\
									\
    xpthread_mutex_destroy (&sync);					\
									\
    bool ok = true;							\
    if (!check_stderr (tcase->expect_errmsg, stderr_trapped))		\
      {									\
	ok = false;							\
	printf("FAIL: %s: stderr not as expected\n", tcase->label);	\
      }									\
    if (!tdata.ok)							\
      {									\
	ok = false;							\
	printf("FAIL: %s: did not complete loop\n", tcase->label);	\
      }									\
    if (rv != PTHREAD_CANCELED)						\
      {									\
	ok = false;							\
	printf("FAIL: %s: thread was not cancelled\n", tcase->label);	\
      }									\
    if (ok)								\
      printf ("pass: %s\n", tcase->label);				\
    return ok;								\
  }

DEFINE_TEST_DRIVER (test_short,
		    getopt (tc->argc, (char *const *)tc->argv, tc->opts))
DEFINE_TEST_DRIVER (test_long,
		    getopt_long (tc->argc, (char *const *)tc->argv,
				 tc->opts, tc->longopts, 0))

/* Caution: all option strings must begin with a '+' or '-' so that
   getopt does not attempt to permute the argument vector (which is in
   read-only memory).  */
const struct test_short tests_short[] = {
  { "no errors",
    "+ab:c", { "program", "-ac", "-b", "x", 0 }, 4, false },
  { "invalid option",
    "+ab:c", { "program", "-d", 0 },		 2, true },
  { "missing argument",
    "+ab:c", { "program", "-b", 0 },		 2, true },
  { 0 }
};

const struct test_long tests_long[] = {
  { "no errors (long)",
    "+ab:c", { { "alpha",   no_argument,       0, 'a' },
	       { "bravo",   required_argument, 0, 'b' },
	       { "charlie", no_argument,       0, 'c' },
	       { 0 } },
    { "program", "-a", "--charlie", "--bravo=x", 0 }, 4, false },

  { "invalid option (long)",
    "+ab:c", { { "alpha",   no_argument,       0, 'a' },
	       { "bravo",   required_argument, 0, 'b' },
	       { "charlie", no_argument,       0, 'c' },
	       { 0 } },
    { "program", "-a", "--charlie", "--dingo", 0 }, 4, true },

  { "unwanted argument",
    "+ab:c", { { "alpha",   no_argument,       0, 'a' },
	       { "bravo",   required_argument, 0, 'b' },
	       { "charlie", no_argument,       0, 'c' },
	       { 0 } },
    { "program", "-a", "--charlie=dingo", "--bravo=x", 0 }, 4, true },

  { "missing argument",
    "+ab:c", { { "alpha",   no_argument,       0, 'a' },
	       { "bravo",   required_argument, 0, 'b' },
	       { "charlie", no_argument,       0, 'c' },
	       { 0 } },
    { "program", "-a", "--charlie", "--bravo", 0 }, 4, true },

  { "ambiguous options",
    "+uvw", { { "veni", no_argument, 0, 'u' },
	      { "vedi", no_argument, 0, 'v' },
	      { "veci", no_argument, 0, 'w' } },
    { "program", "--ve", 0 }, 2, true },

  { "no errors (long W)",
    "+ab:cW;", { { "alpha",   no_argument,	 0, 'a' },
		 { "bravo",   required_argument, 0, 'b' },
		 { "charlie", no_argument,	 0, 'c' },
		 { 0 } },
    { "program", "-a", "-W", "charlie", "-W", "bravo=x", 0 }, 6, false },

  { "missing argument (W itself)",
    "+ab:cW;", { { "alpha",   no_argument,	 0, 'a' },
		 { "bravo",   required_argument, 0, 'b' },
		 { "charlie", no_argument,	 0, 'c' },
		 { 0 } },
    { "program", "-a", "-W", "charlie", "-W", 0 }, 5, true },

  { "missing argument (W longopt)",
    "+ab:cW;", { { "alpha",   no_argument,	 0, 'a' },
		 { "bravo",   required_argument, 0, 'b' },
		 { "charlie", no_argument,	 0, 'c' },
		 { 0 } },
    { "program", "-a", "-W", "charlie", "-W", "bravo", 0 }, 6, true },

  { "unwanted argument (W longopt)",
    "+ab:cW;", { { "alpha",   no_argument,	 0, 'a' },
		 { "bravo",   required_argument, 0, 'b' },
		 { "charlie", no_argument,	 0, 'c' },
		 { 0 } },
    { "program", "-a", "-W", "charlie=dingo", "-W", "bravo=x", 0 }, 6, true },

  { "ambiguous options (W)",
    "+uvwW;", { { "veni", no_argument, 0, 'u' },
		{ "vedi", no_argument, 0, 'v' },
		{ "veci", no_argument, 0, 'w' } },
    { "program", "-W", "ve", 0 }, 3, true },

  { 0 }
};

static int
do_test (void)
{
  int stderr_trap = create_temp_file ("stderr", 0);
  if (stderr_trap < 0)
    {
      perror ("create_temp_file");
      return 1;
    }
  FILE *stderr_trapped = fdopen(stderr_trap, "r+");
  if (!stderr_trapped)
    {
      perror ("fdopen");
      return 1;
    }
  int old_stderr = dup (fileno (stderr));
  if (old_stderr < 0)
    {
      perror ("dup");
      return 1;
    }
  if (dup2 (stderr_trap, 2) < 0)
    {
      perror ("dup2");
      return 1;
    }
  rewind (stderr);

  bool success = true;

  for (const struct test_short *tcase = tests_short; tcase->label; tcase++)
    success = do_test_short (tcase, stderr_trapped) && success;

  for (const struct test_long *tcase = tests_long; tcase->label; tcase++)
    success = do_test_long (tcase, stderr_trapped) && success;

  dup2 (old_stderr, 2);
  close (old_stderr);
  fclose (stderr_trapped);

  return success ? 0 : 1;
}

#include <support/test-driver.c>
