/* Check _dl_exception_create_format.
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

#include <ldsodefs.h>
#include <array_length.h>

#include <support/check.h>
#include <support/xunistd.h>
#include <support/capture_subprocess.h>

#define TEST(es, objn, fmt, ...)					\
  ({									\
     struct dl_exception exception;					\
     _dl_exception_create_format (&exception, objn, fmt, __VA_ARGS__);	\
     TEST_COMPARE_STRING (exception.objname, objn == NULL ? "" : objn);	\
     TEST_COMPARE_STRING (exception.errstring, es);			\
     _dl_exception_free (&exception);					\
   })

static void
do_test_invalid_conversion (void *closure)
{
  TEST ("(null)", NULL, "%p", NULL);
}

/* Exit status after abnormal termination.  */
static int invalid_status;

static void
init_invalid_status (void)
{
  pid_t pid = xfork ();
  if (pid == 0)
    _exit (127);
  xwaitpid (pid, &invalid_status, 0);
  if (WIFEXITED (invalid_status))
    invalid_status = WEXITSTATUS (invalid_status);
}

static int
do_test (void)
{
  init_invalid_status ();

  TEST ("test",      NULL,   "%s",      "test");
  TEST ("test-test", NULL,   "%s-test", "test");
  TEST ("test",      "test", "%s",      "test");
  TEST ("test-test", "test", "%s-test", "test");

  TEST ("test%",      NULL,   "%s%%",      "test");
  TEST ("test%-test", NULL,   "%s%%-test", "test");
  TEST ("test%",      "test", "%s%%",      "test");
  TEST ("test%-test", "test", "%s%%-test", "test");

  TEST ("0000007b",      NULL,   "%x",      123);
  TEST ("0000007b-test", NULL,   "%x-test", 123);
  TEST ("0000007b",      "test", "%x",      123);
  TEST ("0000007b-test", "test", "%x-test", 123);

#define TEST_LONG(es, objn, fmt, ...)				\
  ({								\
     if (sizeof (int) == sizeof (long int))			\
       TEST (es, objn, fmt, __VA_ARGS__);			\
     else							\
       TEST ("ffffffff" es, objn, fmt, __VA_ARGS__);		\
   })

  TEST_LONG ("fffffffd",      NULL,   "%lx",      (long int)~2ul);
  TEST_LONG ("fffffffd-test", NULL,   "%lx-test", (long int)~2ul);
  TEST_LONG ("fffffffd",      "test", "%lx",      (long int)~2ul);
  TEST_LONG ("fffffffd-test", "test", "%lx-test", (long int)~2ul);

  TEST_LONG ("fffffffe",      NULL,   "%zx",      (size_t)~1ul);
  TEST_LONG ("fffffffe-test", NULL,   "%zx-test", (size_t)~1ul);
  TEST_LONG ("fffffffe",      "test", "%zx",      (size_t)~1ul);
  TEST_LONG ("fffffffe-test", "test", "%zx-test", (size_t)~1ul);

  struct support_capture_subprocess result;
  result = support_capture_subprocess (do_test_invalid_conversion, NULL);
  support_capture_subprocess_check (&result, "dl-exception",
				    invalid_status, sc_allow_stderr);
  TEST_COMPARE_STRING (result.err.buffer,
		       "Fatal error: invalid format in exception string\n");

  return 0;
}

#include <support/test-driver.c>
