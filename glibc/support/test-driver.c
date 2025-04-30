/* Main function for test programs.
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

/* This file should be included from test cases.  It will define a
   main function which provides the test wrapper.

   It assumes that the test case defines a function

     int do_test (void);

   and arranges for that function being called under the test wrapper.
   The do_test function should return 0 to indicate a passing test, 1
   to indicate a failing test, or 77 to indicate an unsupported test.
   Other result values could be used to indicate a failing test, but
   the result of the expression is passed to exit and exit only
   returns the lower 8 bits of its input.  A non-zero return with some
   values could cause a test to incorrectly be considered passing when
   it really failed.  For this reason, the function should always
   return 0 (EXIT_SUCCESS), 1 (EXIT_FAILURE), or 77
   (EXIT_UNSUPPORTED).

   The test function may print out diagnostic or warning messages as well
   as messages about failures.  These messages should be printed to stdout
   and not stderr so that the output is properly ordered with respect to
   the rest of the glibc testsuite run output.

   Several preprocessors macros can be defined before including this
   file.

   The name of the do_test function can be changed with the
   TEST_FUNCTION macro.  It must expand to the desired function name.

   If the test case needs access to command line parameters, it must
   define the TEST_FUNCTION_ARGV macro with the name of the test
   function.  It must have the following type:

     int TEST_FUNCTION_ARGV (int argc, char **argv);

   This overrides the do_test default function and is incompatible
   with the TEST_FUNCTION macro.

   If PREPARE is defined, it must expand to the name of a function of
   the type

     void PREPARE (int argc, char **);

   This function will be called early, after parsing the command line,
   but before running the test, in the parent process which acts as
   the test supervisor.

   If CLEANUP_HANDLER is defined, it must expand to the name of a
   function of the type

     void CLEANUP_HANDLER (void);

   This function will be called from the timeout (SIGALRM) signal
   handler.

   If EXPECTED_SIGNAL is defined, it must expanded to a constant which
   denotes the expected signal number.

   If EXPECTED_STATUS is defined, it must expand to the expected exit
   status.

   If TIMEOUT is defined, it must be positive constant.  It overrides
   the default test timeout and is measured in seconds.

   If TEST_NO_MALLOPT is defined, the test wrapper will not call
   mallopt.

   Custom command line handling can be implemented by defining the
   CMDLINE_OPTION macro (after including the <getopt.h> header; this
   requires _GNU_SOURCE to be defined).  This macro must expand to a
   to a comma-separated list of braced initializers for struct option
   from <getopt.h>, with a trailing comma.  CMDLINE_PROCESS can be
   defined as the name of a function which is called to process these
   options.  The function is passed the option character/number and
   has this type:

     void CMDLINE_PROCESS (int);

   If the program also to process custom default short command line
   argument (similar to getopt) it must define CMDLINE_OPTSTRING
   with the expected options (for instance "vb").
*/

#include <support/test-driver.h>

#include <string.h>

int
main (int argc, char **argv)
{
  struct test_config test_config;
  memset (&test_config, 0, sizeof (test_config));

#ifdef PREPARE
  test_config.prepare_function = (PREPARE);
#endif

#if defined (TEST_FUNCTION) && defined (TEST_FUNCTON_ARGV)
# error TEST_FUNCTION and TEST_FUNCTION_ARGV cannot be defined at the same time
#endif
#if defined (TEST_FUNCTION)
  test_config.test_function = TEST_FUNCTION;
#elif defined (TEST_FUNCTION_ARGV)
  test_config.test_function_argv = TEST_FUNCTION_ARGV;
#else
  test_config.test_function = do_test;
#endif

#ifdef CLEANUP_HANDLER
  test_config.cleanup_function = CLEANUP_HANDLER;
#endif

#ifdef EXPECTED_SIGNAL
  test_config.expected_signal = (EXPECTED_SIGNAL);
#endif

#ifdef EXPECTED_STATUS
  test_config.expected_status = (EXPECTED_STATUS);
#endif

#ifdef TEST_NO_MALLOPT
  test_config.no_mallopt = 1;
#endif

#ifdef TEST_NO_SETVBUF
  test_config.no_setvbuf = 1;
#endif

#ifdef TIMEOUT
  test_config.timeout = TIMEOUT;
#endif

#ifdef CMDLINE_OPTIONS
  struct option options[] =
    {
      CMDLINE_OPTIONS
      TEST_DEFAULT_OPTIONS
    };
  test_config.options = &options;
#endif
#ifdef CMDLINE_PROCESS
  test_config.cmdline_function = CMDLINE_PROCESS;
#endif
#ifdef CMDLINE_OPTSTRING
  test_config.optstring = "+" CMDLINE_OPTSTRING;
#else
  test_config.optstring = "+";
#endif

  return support_test_main (argc, argv, &test_config);
}
