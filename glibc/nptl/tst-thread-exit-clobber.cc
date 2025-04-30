/* Test that pthread_exit does not clobber callee-saved registers.
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

#include <stdio.h>
#include <support/check.h>
#include <support/xthread.h>

/* This test attempts to check that callee-saved registers are
   restored to their original values when destructors are run after
   pthread_exit is called.  GCC PR 83641 causes this test to fail.

   The constants have been chosen randomly and are magic values which
   are used to detect whether registers have been clobbered.  The idea
   is that these values are hidden behind a compiler barrier and only
   present in .rodata initially, so that it is less likely that they
   are in a register by accident.

   The checker class can be stored in registers, and the magic values
   are directly loaded into these registers.  The checker destructor
   is eventually invoked by pthread_exit and calls one of the
   check_magic functions to verify that the class contents (that is,
   register value) is correct.

   These tests are performed both for unsigned int and double values,
   to cover different calling conventions.  */

template <class T>
struct values
{
  T v0;
  T v1;
  T v2;
  T v3;
  T v4;
};

static const values<unsigned int> magic_values =
  {
    0x57f7fc72,
    0xe582daba,
    0x5f6ac994,
    0x35efddb7,
    0x1fbf5a74,
  };

static const values<double> magic_values_double =
  {
    0.6764041905675465,
    0.9533336788140494,
    0.6091161359041452,
    0.7668653957125336,
    0.010374520235509666,
  };

/* Special index value which tells check_magic that no check should be
   performed.  */
enum { no_check = -1 };

/* Check that VALUE is the magic value for INDEX, behind a compiler
   barrier.  */
__attribute__ ((noinline, noclone, weak))
void
check_magic (int index, unsigned int value)
{
  switch (index)
    {
    case 0:
      TEST_COMPARE (value, magic_values.v0);
      break;
    case 1:
      TEST_COMPARE (value, magic_values.v1);
      break;
    case 2:
      TEST_COMPARE (value, magic_values.v2);
      break;
    case 3:
      TEST_COMPARE (value, magic_values.v3);
      break;
    case 4:
      TEST_COMPARE (value, magic_values.v4);
      break;
    case no_check:
      break;
    default:
      FAIL_EXIT1 ("invalid magic value index %d", index);
    }
}

/* Check that VALUE is the magic value for INDEX, behind a compiler
   barrier.  Double variant.  */
__attribute__ ((noinline, noclone, weak))
void
check_magic (int index, double value)
{
  switch (index)
    {
    case 0:
      TEST_VERIFY (value == magic_values_double.v0);
      break;
    case 1:
      TEST_VERIFY (value == magic_values_double.v1);
      break;
    case 2:
      TEST_VERIFY (value == magic_values_double.v2);
      break;
    case 3:
      TEST_VERIFY (value == magic_values_double.v3);
      break;
    case 4:
      TEST_VERIFY (value == magic_values_double.v4);
      break;
    case no_check:
      break;
    default:
      FAIL_EXIT1 ("invalid magic value index %d", index);
    }
}

/* Store a magic value and check, via the destructor, that it has the
   expected value.  */
template <class T, int I>
struct checker
{
  T value;

  checker (T v)
    : value (v)
  {
  }

  ~checker ()
  {
    check_magic (I, value);
  }
};

/* The functions call_pthread_exit_0, call_pthread_exit_1,
   call_pthread_exit are used to call pthread_exit indirectly, with
   the intent of clobbering the register values.  */

__attribute__ ((noinline, noclone, weak))
void
call_pthread_exit_0 (const values<unsigned int> *pvalues)
{
  checker<unsigned int, no_check> c0 (pvalues->v0);
  checker<unsigned int, no_check> c1 (pvalues->v1);
  checker<unsigned int, no_check> c2 (pvalues->v2);
  checker<unsigned int, no_check> c3 (pvalues->v3);
  checker<unsigned int, no_check> c4 (pvalues->v4);

  pthread_exit (NULL);
}

__attribute__ ((noinline, noclone, weak))
void
call_pthread_exit_1 (const values<double> *pvalues)
{
  checker<double, no_check> c0 (pvalues->v0);
  checker<double, no_check> c1 (pvalues->v1);
  checker<double, no_check> c2 (pvalues->v2);
  checker<double, no_check> c3 (pvalues->v3);
  checker<double, no_check> c4 (pvalues->v4);

  values<unsigned int> other_values = { 0, };
  call_pthread_exit_0 (&other_values);
}

__attribute__ ((noinline, noclone, weak))
void
call_pthread_exit ()
{
  values<double> other_values = { 0, };
  call_pthread_exit_1 (&other_values);
}

/* Create on-stack objects and check that their values are restored by
   pthread_exit.  If Nested is true, call pthread_exit indirectly via
   call_pthread_exit.  */
template <class T, bool Nested>
__attribute__ ((noinline, noclone, weak))
void *
threadfunc (void *closure)
{
  const values<T> *pvalues = static_cast<const values<T> *> (closure);

  checker<T, 0> c0 (pvalues->v0);
  checker<T, 1> c1 (pvalues->v1);
  checker<T, 2> c2 (pvalues->v2);
  checker<T, 3> c3 (pvalues->v3);
  checker<T, 4> c4 (pvalues->v4);

  if (Nested)
    call_pthread_exit ();
  else
    pthread_exit (NULL);

  /* This should not be reached.  */
  return const_cast<char *> ("");
}

static int
do_test ()
{
  puts ("info: unsigned int, direct pthread_exit call");
  pthread_t thr
    = xpthread_create (NULL, &threadfunc<unsigned int, false>,
                       const_cast<values<unsigned int> *> (&magic_values));
  TEST_VERIFY (xpthread_join (thr) == NULL);

  puts ("info: double, direct pthread_exit call");
  thr = xpthread_create (NULL, &threadfunc<double, false>,
                         const_cast<values<double> *> (&magic_values_double));
  TEST_VERIFY (xpthread_join (thr) == NULL);

  puts ("info: unsigned int, indirect pthread_exit call");
  thr = xpthread_create (NULL, &threadfunc<unsigned int, true>,
                       const_cast<values<unsigned int> *> (&magic_values));
  TEST_VERIFY (xpthread_join (thr) == NULL);

  puts ("info: double, indirect pthread_exit call");
  thr = xpthread_create (NULL, &threadfunc<double, true>,
                         const_cast<values<double> *> (&magic_values_double));
  TEST_VERIFY (xpthread_join (thr) == NULL);

  return 0;
}

#include <support/test-driver.c>
