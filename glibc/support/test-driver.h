/* Interfaces for the test driver.
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

#ifndef SUPPORT_TEST_DRIVER_H
#define SUPPORT_TEST_DRIVER_H

#include <sys/cdefs.h>

__BEGIN_DECLS

struct test_config
{
  void (*prepare_function) (int argc, char **argv);
  int (*test_function) (void);
  int (*test_function_argv) (int argc, char **argv);
  void (*cleanup_function) (void);
  void (*cmdline_function) (int);
  const void *options;   /* Custom options if not NULL.  */
  int timeout;           /* Test timeout in seconds.  */
  int expected_status;   /* Expected exit status.  */
  int expected_signal;   /* If non-zero, expect termination by signal.  */
  char no_mallopt;       /* Boolean flag to disable mallopt.  */
  char no_setvbuf;       /* Boolean flag to disable setvbuf.  */
  const char *optstring; /* Short command line options.  */
};

enum
  {
    /* Test exit status which indicates that the feature is
       unsupported. */
    EXIT_UNSUPPORTED = 77,

    /* Default timeout is twenty seconds.  Tests should normally
       complete faster than this, but if they don't, that's abnormal
       (a bug) anyways.  */
    DEFAULT_TIMEOUT = 20,

    /* Used for command line argument parsing.  */
    OPT_DIRECT = 1000,
    OPT_TESTDIR,
  };

/* Options provided by the test driver.  */
#define TEST_DEFAULT_OPTIONS                            \
  { "verbose", no_argument, NULL, 'v' },                \
  { "direct", no_argument, NULL, OPT_DIRECT },          \
  { "test-dir", required_argument, NULL, OPT_TESTDIR }, \

/* The directory the test should use for temporary files.  */
extern const char *test_dir;

/* The number of --verbose arguments specified during program
   invocation.  This variable can be used to control the verbosity of
   tests.  */
extern unsigned int test_verbose;

/* Output that is only emitted if at least one --verbose argument was
   specified. */
#define verbose_printf(...)                      \
  do {                                           \
    if (test_verbose > 0)                        \
      printf (__VA_ARGS__);                      \
  } while (0);

int support_test_main (int argc, char **argv, const struct test_config *);

__END_DECLS

#endif /* SUPPORT_TEST_DRIVER_H */
