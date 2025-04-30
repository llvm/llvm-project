/* Test that throwing C++ exceptions works with the minimum stack size.
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

#include <stdexcept>

#include <limits.h>
#include <string.h>
#include <support/check.h>
#include <support/xthread.h>

/* Throw a std::runtime_exception.  */
__attribute__ ((noinline, noclone, weak))
void
do_throw_exception ()
{
  throw std::runtime_error ("test exception");
}

/* Class with a destructor, to trigger unwind handling.  */
struct class_with_destructor
{
  class_with_destructor ();
  ~class_with_destructor ();
};

__attribute__ ((noinline, noclone, weak))
class_with_destructor::class_with_destructor ()
{
}

__attribute__ ((noinline, noclone, weak))
class_with_destructor::~class_with_destructor ()
{
}

__attribute__ ((noinline, noclone, weak))
void
function_with_destructed_object ()
{
  class_with_destructor obj;
  do_throw_exception ();
}

static void *
threadfunc (void *closure)
{
  try
    {
      function_with_destructed_object ();
      FAIL_EXIT1 ("no exception thrown");
    }
  catch (std::exception &e)
    {
      TEST_COMPARE (strcmp (e.what (), "test exception"), 0);
      return reinterpret_cast<void *> (threadfunc);
    }
  FAIL_EXIT1 ("no exception caught");
}

static int
do_test (void)
{
  pthread_attr_t attr;
  xpthread_attr_init (&attr);
  xpthread_attr_setstacksize (&attr, PTHREAD_STACK_MIN);
  pthread_t thr = xpthread_create (&attr, threadfunc, NULL);
  TEST_VERIFY (xpthread_join (thr) == threadfunc);
  xpthread_attr_destroy (&attr);
  return 0;
}

#include <support/test-driver.c>
