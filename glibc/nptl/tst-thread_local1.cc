/* Test basic thread_local support.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

#include <functional>
#include <string>
#include <thread>

struct counter
{
  int constructed {};
  int destructed {};

  void reset ();
};

void
counter::reset ()
{
  constructed = 0;
  destructed = 0;
}

static std::string
to_string (const counter &c)
{
  char buf[128];
  snprintf (buf, sizeof (buf), "%d/%d",
            c.constructed, c.destructed);
  return buf;
}

template <counter *Counter>
struct counting
{
  counting () __attribute__ ((noinline, noclone));
  ~counting () __attribute__ ((noinline, noclone));
  void operation () __attribute__ ((noinline, noclone));
};

template<counter *Counter>
__attribute__ ((noinline, noclone))
counting<Counter>::counting ()
{
  ++Counter->constructed;
}

template<counter *Counter>
__attribute__ ((noinline, noclone))
counting<Counter>::~counting ()
{
  ++Counter->destructed;
}

template<counter *Counter>
void __attribute__ ((noinline, noclone))
counting<Counter>::operation ()
{
  // Optimization barrier.
  asm ("");
}

static counter counter_static;
static counter counter_anonymous_namespace;
static counter counter_extern;
static counter counter_function_local;
static bool errors (false);

static std::string
all_counters ()
{
  return to_string (counter_static)
    + ' ' + to_string (counter_anonymous_namespace)
    + ' ' + to_string (counter_extern)
    + ' ' + to_string (counter_function_local);
}

static void
check_counters (const char *name, const char *expected)
{
  std::string actual{all_counters ()};
  if (actual != expected)
    {
      printf ("error: %s: (%s) != (%s)\n",
              name, actual.c_str (), expected);
      errors = true;
    }
}

static void
reset_all ()
{
  counter_static.reset ();
  counter_anonymous_namespace.reset ();
  counter_extern.reset ();
  counter_function_local.reset ();
}

static thread_local counting<&counter_static> counting_static;
namespace {
  thread_local counting<&counter_anonymous_namespace>
    counting_anonymous_namespace;
}
extern thread_local counting<&counter_extern> counting_extern;
thread_local counting<&counter_extern> counting_extern;

static void *
thread_without_access (void *)
{
  return nullptr;
}

static void *
thread_with_access (void *)
{
  thread_local counting<&counter_function_local> counting_function_local;
  counting_function_local.operation ();
  check_counters ("early in thread_with_access", "0/0 0/0 0/0 1/0");
  counting_static.operation ();
  counting_anonymous_namespace.operation ();
  counting_extern.operation ();
  check_counters ("in thread_with_access", "1/0 1/0 1/0 1/0");
  return nullptr;
}

static int
do_test (void)
{
  std::function<void (void *(void *))> do_pthread =
    [](void *(func) (void *))
    {
      pthread_t thr;
      int ret = pthread_create (&thr, nullptr, func, nullptr);
      if (ret != 0)
        {
          errno = ret;
          printf ("error: pthread_create: %m\n");
          errors = true;
          return;
        }
      ret = pthread_join (thr, nullptr);
      if (ret != 0)
        {
          errno = ret;
          printf ("error: pthread_join: %m\n");
          errors = true;
          return;
        }
    };
  std::function<void (void *(void *))> do_std_thread =
    [](void *(func) (void *))
    {
      std::thread thr{[func] {func (nullptr);}};
      thr.join ();
    };

  std::array<std::pair<const char *, std::function<void (void *(void *))>>, 2>
    do_thread_X
      {{
        {"pthread_create", do_pthread},
        {"std::thread", do_std_thread},
      }};

  for (auto do_thread : do_thread_X)
    {
      printf ("info: testing %s\n", do_thread.first);
      check_counters ("initial", "0/0 0/0 0/0 0/0");
      do_thread.second (thread_without_access);
      check_counters ("after thread_without_access", "0/0 0/0 0/0 0/0");
      reset_all ();
      do_thread.second (thread_with_access);
      check_counters ("after thread_with_access", "1/1 1/1 1/1 1/1");
      reset_all ();
    }

  return errors;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
