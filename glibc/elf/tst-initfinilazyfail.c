/* Test that lazy binding failures in constructors and destructors are fatal.
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

#include <dlfcn.h>
#include <string.h>
#include <support/capture_subprocess.h>
#include <support/check.h>
#include <support/xdlfcn.h>

static void
test_constructor (void *closure)
{
  void *handle = dlopen ("tst-initlazyfailmod.so", RTLD_LAZY);
  if (handle == NULL)
    FAIL_EXIT (2, "dlopen did not terminate the process: %s", dlerror ());
  else
    FAIL_EXIT (2, "dlopen did not terminate the process (%p)", handle);
}

static void
test_destructor (void *closure)
{
  void *handle = xdlopen ("tst-finilazyfailmod.so", RTLD_LAZY);
  int ret = dlclose (handle);
  const char *message = dlerror ();
  if (message != NULL)
    FAIL_EXIT (2, "dlclose did not terminate the process: %d, %s",
               ret, message);
  else
    FAIL_EXIT (2, "dlopen did not terminate the process: %d", ret);
}

static int
do_test (void)
{
  {
    struct support_capture_subprocess proc
      = support_capture_subprocess (test_constructor, NULL);
    support_capture_subprocess_check (&proc, "constructor", 127,
                                      sc_allow_stderr);
    printf ("info: constructor failure output: [[%s]]\n", proc.err.buffer);
    TEST_VERIFY (strstr (proc.err.buffer,
                         "tst-initfinilazyfail: symbol lookup error: ")
                 != NULL);
    TEST_VERIFY (strstr (proc.err.buffer,
                         "tst-initlazyfailmod.so: undefined symbol:"
                         " undefined_function\n") != NULL);
    support_capture_subprocess_free (&proc);
  }

  {
    struct support_capture_subprocess proc
      = support_capture_subprocess (test_destructor, NULL);
    support_capture_subprocess_check (&proc, "destructor", 127,
                                      sc_allow_stderr);
    printf ("info: destructor failure output: [[%s]]\n", proc.err.buffer);
    TEST_VERIFY (strstr (proc.err.buffer,
                         "tst-initfinilazyfail: symbol lookup error: ")
                 != NULL);
    TEST_VERIFY (strstr (proc.err.buffer,
                         "tst-finilazyfailmod.so: undefined symbol:"
                         " undefined_function\n") != NULL);
    support_capture_subprocess_free (&proc);
  }

  return 0;
}

#include <support/test-driver.c>
