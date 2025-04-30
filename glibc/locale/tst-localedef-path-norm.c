/* Test for localedef path name handling and normalization.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

/* The test runs localedef with various named paths to test for expected
   behaviours dealing with codeset name normalization.  That is to say that use
   of UTF-8, and it's variations, are normalized to utf8.  Likewise that values
   after the @ are not normalized and left as-is.  The test needs to run
   localedef with known input values and then check that the generated path
   matches the expected value after normalization.  */

/* Note: In some cases adding -v (verbose) to localedef changes the exit
   status to a non-zero value because some warnings are only enabled in verbose
   mode.  This should probably be changed so warnings are either present or not
   present, regardless of verbosity.  POSIX requires that any warnings cause the
   exit status to be non-zero.  */

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <support/capture_subprocess.h>
#include <support/check.h>
#include <support/support.h>
#include <support/xunistd.h>

/* Full path to localedef.  */
char *prog;

/* Execute localedef in a subprocess.  */
static void
execv_wrapper (void *args)
{
  char **argv = args;

  execv (prog, argv);
  FAIL_EXIT1 ("execv: %m");
}

struct test_closure
{
  /* Arguments for running localedef.  */
  const char *const argv[16];
  /* Expected directory name for compiled locale.  */
  const char *exp;
  /* Expected path to compiled locale.  */
  const char *complocaledir;
};

/* Run localedef with DATA.ARGV arguments (NULL terminated), and expect path to
   the compiled locale is "DATA.COMPLOCALEDIR/DATA.EXP".  */
static void
run_test (struct test_closure data)
{
  const char * const *args = data.argv;
  const char *exp = data.exp;
  const char *complocaledir = data.complocaledir;
  struct stat64 fs;

  /* Expected output path.  */
  const char *path = xasprintf ("%s/%s", complocaledir, exp);

  /* Run test.  */
  struct support_capture_subprocess result;
  result = support_capture_subprocess (execv_wrapper, (void *)args);
  support_capture_subprocess_check (&result, "execv", 0, sc_allow_none);
  support_capture_subprocess_free (&result);

  /* Verify path is present and is a directory.  */
  xstat (path, &fs);
  TEST_VERIFY_EXIT (S_ISDIR (fs.st_mode));
  printf ("info: Directory '%s' exists.\n", path);
}

static int
do_test (void)
{
  /* We are running as root inside the container.  */
  prog = xasprintf ("%s/localedef", support_bindir_prefix);

  /* We need an arbitrary absolute path for localedef output.
     Writing to a non-default absolute path disables any kind
     of path normalization since we expect the user wants the path
     exactly as they specified it.  */
#define ABSDIR "/output"
  xmkdirp (ABSDIR, 0777);

  /* It takes ~10 seconds to serially execute 9 localedef test.  We
     could run the compilations in parallel if we want to reduce test
     time.  We don't want to split this out into distinct tests because
     it would require multiple chroots.  Batching the same localedef
     tests saves disk space during testing.  */

  /* Test 1: Expected normalization.
     Run localedef and expect output in $(complocaledir)/en_US1.utf8,
     with normalization changing UTF-8 to utf8.  */
  run_test ((struct test_closure)
	    {
	      .argv = { prog,
			"--no-archive",
			"-i", "en_US",
			"-f", "UTF-8",
			"en_US1.UTF-8", NULL },
	      .exp = "en_US1.utf8",
	      .complocaledir = support_complocaledir_prefix
	    });

  /* Test 2: No normalization past '@'.
     Run localedef and expect output in $(complocaledir)/en_US2.utf8@tEsT,
     with normalization changing UTF-8@tEsT to utf8@tEsT (everything after
     @ is untouched).  */
  run_test ((struct test_closure)
	    {
	      .argv = { prog,
			"--no-archive",
			"-i", "en_US",
			"-f", "UTF-8",
			"en_US2.UTF-8@tEsT", NULL },
	      .exp = "en_US2.utf8@tEsT",
	      .complocaledir = support_complocaledir_prefix
	    });

  /* Test 3: No normalization past '@' despite period.
     Run localedef and expect output in $(complocaledir)/en_US3@tEsT.UTF-8,
     with normalization changing nothing (everything after @ is untouched)
     despite there being a period near the end.  */
  run_test ((struct test_closure)
	    {
	      .argv = { prog,
			"--no-archive",
			"-i", "en_US",
			"-f", "UTF-8",
			"en_US3@tEsT.UTF-8", NULL },
	      .exp = "en_US3@tEsT.UTF-8",
	      .complocaledir = support_complocaledir_prefix
	    });

  /* Test 4: Normalize numeric codeset by adding 'iso' prefix.
     Run localedef and expect output in $(complocaledir)/en_US4.88591,
     with normalization changing 88591 to iso88591.  */
  run_test ((struct test_closure)
	    {
	      .argv = { prog,
			"--no-archive",
			"-i", "en_US",
			"-f", "UTF-8",
			"en_US4.88591", NULL },
	      .exp = "en_US4.iso88591",
	      .complocaledir = support_complocaledir_prefix
	    });

  /* Test 5: Don't add 'iso' prefix if first char is alpha.
     Run localedef and expect output in $(complocaledir)/en_US5.a88591,
     with normalization changing nothing.  */
  run_test ((struct test_closure)
	    {
	      .argv = { prog,
			"--no-archive",
			"-i", "en_US",
			"-f", "UTF-8",
			"en_US5.a88591", NULL },
	      .exp = "en_US5.a88591",
	      .complocaledir = support_complocaledir_prefix
	    });

  /* Test 6: Don't add 'iso' prefix if last char is alpha.
     Run localedef and expect output in $(complocaledir)/en_US6.88591a,
     with normalization changing nothing.  */
  run_test ((struct test_closure)
	    {
	      .argv = { prog,
			"--no-archive",
			"-i", "en_US",
			"-f", "UTF-8",
			"en_US6.88591a", NULL },
	      .exp = "en_US6.88591a",
	      .complocaledir = support_complocaledir_prefix
	    });

  /* Test 7: Don't normalize anything with an absolute path.
     Run localedef and expect output in ABSDIR/en_US7.UTF-8,
     with normalization changing nothing.  */
  run_test ((struct test_closure)
	    {
	      .argv = { prog,
			"--no-archive",
			"-i", "en_US",
			"-f", "UTF-8",
			ABSDIR "/en_US7.UTF-8", NULL },
	      .exp = "en_US7.UTF-8",
	      .complocaledir = ABSDIR
	    });

  /* Test 8: Don't normalize anything with an absolute path.
     Run localedef and expect output in ABSDIR/en_US8.UTF-8@tEsT,
     with normalization changing nothing.  */
  run_test ((struct test_closure)
	    {
	      .argv = { prog,
			"--no-archive",
			"-i", "en_US",
			"-f", "UTF-8",
			ABSDIR "/en_US8.UTF-8@tEsT", NULL },
	      .exp = "en_US8.UTF-8@tEsT",
	      .complocaledir = ABSDIR
	    });

  /* Test 9: Don't normalize anything with an absolute path.
     Run localedef and expect output in ABSDIR/en_US9@tEsT.UTF-8,
     with normalization changing nothing.  */
  run_test ((struct test_closure)
	    {
	      .argv = { prog,
			"--no-archive",
			"-i", "en_US",
			"-f", "UTF-8",
			ABSDIR "/en_US9@tEsT.UTF-8", NULL },
	      .exp = "en_US9@tEsT.UTF-8",
	      .complocaledir = ABSDIR
	    });

  return 0;
}

#define TIMEOUT 30
#include <support/test-driver.c>
