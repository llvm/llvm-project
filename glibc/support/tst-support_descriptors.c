/* Tests for monitoring file descriptor usage.
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

#include <fcntl.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <support/capture_subprocess.h>
#include <support/check.h>
#include <support/descriptors.h>
#include <support/support.h>
#include <support/xunistd.h>

/* This is the next free descriptor that the subprocess will pick.  */
static int free_descriptor;

static void
subprocess_no_change (void *closure)
{
  struct support_descriptors *descrs = support_descriptors_list ();
  int fd = xopen ("/dev/null", O_WRONLY, 0);
  TEST_COMPARE (fd, free_descriptor);
  xclose (fd);
  support_descriptors_free (descrs);
}

static void
subprocess_closed_descriptor (void *closure)
{
  int fd = xopen ("/dev/null", O_WRONLY, 0);
  TEST_COMPARE (fd, free_descriptor);
  struct support_descriptors *descrs = support_descriptors_list ();
  xclose (fd);
  support_descriptors_check (descrs); /* Will report failure.  */
  puts ("EOT");
  support_descriptors_free (descrs);
}

static void
subprocess_opened_descriptor (void *closure)
{
  struct support_descriptors *descrs = support_descriptors_list ();
  int fd = xopen ("/dev/null", O_WRONLY, 0);
  TEST_COMPARE (fd, free_descriptor);
  support_descriptors_check (descrs); /* Will report failure.  */
  puts ("EOT");
  support_descriptors_free (descrs);
}

static void
subprocess_changed_descriptor (void *closure)
{
  int fd = xopen ("/dev/null", O_WRONLY, 0);
  TEST_COMPARE (fd, free_descriptor);
  struct support_descriptors *descrs = support_descriptors_list ();
  xclose (fd);
  TEST_COMPARE (xopen ("/dev", O_DIRECTORY | O_RDONLY, 0), fd);
  support_descriptors_check (descrs); /* Will report failure.  */
  puts ("EOT");
  support_descriptors_free (descrs);
}

static void
report_subprocess_output (const char *name,
                          struct support_capture_subprocess *proc)
{
  printf ("info: BEGIN %s output\n"
          "%s"
          "info: END %s output\n",
          name, proc->out.buffer, name);
}

/* Use an explicit flag to preserve failure status across
   support_record_failure_reset calls.  */
static bool good = true;

static void
test_run (void)
{
  struct support_capture_subprocess proc = support_capture_subprocess
    (&subprocess_no_change, NULL);
  support_capture_subprocess_check (&proc, "subprocess_no_change",
                                    0, sc_allow_none);
  support_capture_subprocess_free (&proc);

  char *expected = xasprintf ("\nDifferences:\n"
                              "error: descriptor %d was closed\n"
                              "EOT\n",
                              free_descriptor);
  good = good && !support_record_failure_is_failed ();
  proc = support_capture_subprocess (&subprocess_closed_descriptor, NULL);
  good = good && support_record_failure_is_failed ();
  support_record_failure_reset (); /* Discard the reported error.  */
  report_subprocess_output ("subprocess_closed_descriptor", &proc);
  TEST_VERIFY (strstr (proc.out.buffer, expected) != NULL);
  support_capture_subprocess_check (&proc, "subprocess_closed_descriptor",
                                    0, sc_allow_stdout);
  support_capture_subprocess_free (&proc);
  free (expected);

  expected = xasprintf ("\nDifferences:\n"
                        "error: descriptor %d was opened (\"/dev/null\")\n"
                        "EOT\n",
                        free_descriptor);
  good = good && !support_record_failure_is_failed ();
  proc = support_capture_subprocess (&subprocess_opened_descriptor, NULL);
  good = good && support_record_failure_is_failed ();
  support_record_failure_reset (); /* Discard the reported error.  */
  report_subprocess_output ("subprocess_opened_descriptor", &proc);
  TEST_VERIFY (strstr (proc.out.buffer, expected) != NULL);
  support_capture_subprocess_check (&proc, "subprocess_opened_descriptor",
                                    0, sc_allow_stdout);
  support_capture_subprocess_free (&proc);
  free (expected);

  expected = xasprintf ("\nDifferences:\n"
                        "error: descriptor %d changed from \"/dev/null\""
                        " to \"/dev\"\n"
                        "error: descriptor %d changed ino ",
                        free_descriptor, free_descriptor);
  good = good && !support_record_failure_is_failed ();
  proc = support_capture_subprocess (&subprocess_changed_descriptor, NULL);
  good = good && support_record_failure_is_failed ();
  support_record_failure_reset (); /* Discard the reported error.  */
  report_subprocess_output ("subprocess_changed_descriptor", &proc);
  TEST_VERIFY (strstr (proc.out.buffer, expected) != NULL);
  support_capture_subprocess_check (&proc, "subprocess_changed_descriptor",
                                    0, sc_allow_stdout);
  support_capture_subprocess_free (&proc);
  free (expected);
}

static int
do_test (void)
{
  puts ("info: initial descriptor set");
  {
    struct support_descriptors *descrs = support_descriptors_list ();
    support_descriptors_dump (descrs, "info:  ", stdout);
    support_descriptors_free (descrs);
  }

  free_descriptor = xopen ("/dev/null", O_WRONLY, 0);
  puts ("info: descriptor set with additional free descriptor");
  {
    struct support_descriptors *descrs = support_descriptors_list ();
    support_descriptors_dump (descrs, "info:  ", stdout);
    support_descriptors_free (descrs);
  }
  TEST_VERIFY (free_descriptor >= 3);
  xclose (free_descriptor);

  /* Initial test run without a sentinel descriptor.  The presence of
     such a descriptor exercises different conditions in the list
     comparison in support_descriptors_check.  */
  test_run ();

  /* Allocate a sentinel descriptor at the end of the descriptor list,
     after free_descriptor.  */
  int sentinel_fd;
  {
    int fd = xopen ("/dev/full", O_WRONLY, 0);
    TEST_COMPARE (fd, free_descriptor);
    sentinel_fd = dup (fd);
    TEST_VERIFY_EXIT (sentinel_fd > fd);
    xclose (fd);
  }
  puts ("info: descriptor set with sentinel descriptor");
  {
    struct support_descriptors *descrs = support_descriptors_list ();
    support_descriptors_dump (descrs, "info:  ", stdout);
    support_descriptors_free (descrs);
  }

  /* Second test run with sentinel descriptor.  */
  test_run ();

  xclose (sentinel_fd);

  return !good;
}

#include <support/test-driver.c>
