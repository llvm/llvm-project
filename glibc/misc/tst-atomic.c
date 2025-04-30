/* Tests for atomic.h macros.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2003.

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
#include <atomic.h>

#ifndef atomic_t
#ifdef __clang__
# define atomic_t long
#else
# define atomic_t int
#endif
#endif

/* Test various atomic.h macros.  */
static int
do_test (void)
{
  atomic_t mem, expected;
  int ret = 0;

#ifdef atomic_compare_and_exchange_val_acq
  mem = 24;
  if (atomic_compare_and_exchange_val_acq (&mem, 35, 24) != 24
      || mem != 35)
    {
      puts ("atomic_compare_and_exchange_val_acq test 1 failed");
      ret = 1;
    }

  mem = 12;
  if (atomic_compare_and_exchange_val_acq (&mem, 10, 15) != 12
      || mem != 12)
    {
      puts ("atomic_compare_and_exchange_val_acq test 2 failed");
      ret = 1;
    }

  mem = -15;
  if (atomic_compare_and_exchange_val_acq (&mem, -56, -15) != -15
      || mem != -56)
    {
      puts ("atomic_compare_and_exchange_val_acq test 3 failed");
      ret = 1;
    }

  mem = -1;
  if (atomic_compare_and_exchange_val_acq (&mem, 17, 0) != -1
      || mem != -1)
    {
      puts ("atomic_compare_and_exchange_val_acq test 4 failed");
      ret = 1;
    }
#endif

  mem = 24;
  if (atomic_compare_and_exchange_bool_acq (&mem, 35, 24)
      || mem != 35)
    {
      puts ("atomic_compare_and_exchange_bool_acq test 1 failed");
      ret = 1;
    }

  mem = 12;
  if (! atomic_compare_and_exchange_bool_acq (&mem, 10, 15)
      || mem != 12)
    {
      puts ("atomic_compare_and_exchange_bool_acq test 2 failed");
      ret = 1;
    }

  mem = -15;
  if (atomic_compare_and_exchange_bool_acq (&mem, -56, -15)
      || mem != -56)
    {
      puts ("atomic_compare_and_exchange_bool_acq test 3 failed");
      ret = 1;
    }

  mem = -1;
  if (! atomic_compare_and_exchange_bool_acq (&mem, 17, 0)
      || mem != -1)
    {
      puts ("atomic_compare_and_exchange_bool_acq test 4 failed");
      ret = 1;
    }

  mem = 64;
  if (atomic_exchange_acq (&mem, 31) != 64
      || mem != 31)
    {
      puts ("atomic_exchange_acq test failed");
      ret = 1;
    }

  mem = 2;
  if (atomic_exchange_and_add (&mem, 11) != 2
      || mem != 13)
    {
      puts ("atomic_exchange_and_add test failed");
      ret = 1;
    }

  mem = 2;
  if (atomic_exchange_and_add_acq (&mem, 11) != 2
      || mem != 13)
    {
      puts ("atomic_exchange_and_add test failed");
      ret = 1;
    }

  mem = 2;
  if (atomic_exchange_and_add_rel (&mem, 11) != 2
      || mem != 13)
    {
      puts ("atomic_exchange_and_add test failed");
      ret = 1;
    }

  mem = -21;
  atomic_add (&mem, 22);
  if (mem != 1)
    {
      puts ("atomic_add test failed");
      ret = 1;
    }

  mem = -1;
  atomic_increment (&mem);
  if (mem != 0)
    {
      puts ("atomic_increment test failed");
      ret = 1;
    }

  mem = 2;
  if (atomic_increment_val (&mem) != 3)
    {
      puts ("atomic_increment_val test failed");
      ret = 1;
    }

  mem = 0;
  if (atomic_increment_and_test (&mem)
      || mem != 1)
    {
      puts ("atomic_increment_and_test test 1 failed");
      ret = 1;
    }

  mem = 35;
  if (atomic_increment_and_test (&mem)
      || mem != 36)
    {
      puts ("atomic_increment_and_test test 2 failed");
      ret = 1;
    }

  mem = -1;
  if (! atomic_increment_and_test (&mem)
      || mem != 0)
    {
      puts ("atomic_increment_and_test test 3 failed");
      ret = 1;
    }

  mem = 17;
  atomic_decrement (&mem);
  if (mem != 16)
    {
      puts ("atomic_decrement test failed");
      ret = 1;
    }

  if (atomic_decrement_val (&mem) != 15)
    {
      puts ("atomic_decrement_val test failed");
      ret = 1;
    }

  mem = 0;
  if (atomic_decrement_and_test (&mem)
      || mem != -1)
    {
      puts ("atomic_decrement_and_test test 1 failed");
      ret = 1;
    }

  mem = 15;
  if (atomic_decrement_and_test (&mem)
      || mem != 14)
    {
      puts ("atomic_decrement_and_test test 2 failed");
      ret = 1;
    }

  mem = 1;
  if (! atomic_decrement_and_test (&mem)
      || mem != 0)
    {
      puts ("atomic_decrement_and_test test 3 failed");
      ret = 1;
    }

  mem = 1;
  if (atomic_decrement_if_positive (&mem) != 1
      || mem != 0)
    {
      puts ("atomic_decrement_if_positive test 1 failed");
      ret = 1;
    }

  mem = 0;
  if (atomic_decrement_if_positive (&mem) != 0
      || mem != 0)
    {
      puts ("atomic_decrement_if_positive test 2 failed");
      ret = 1;
    }

  mem = -1;
  if (atomic_decrement_if_positive (&mem) != -1
      || mem != -1)
    {
      puts ("atomic_decrement_if_positive test 3 failed");
      ret = 1;
    }

  mem = -12;
  if (! atomic_add_negative (&mem, 10)
      || mem != -2)
    {
      puts ("atomic_add_negative test 1 failed");
      ret = 1;
    }

  mem = 0;
  if (atomic_add_negative (&mem, 100)
      || mem != 100)
    {
      puts ("atomic_add_negative test 2 failed");
      ret = 1;
    }

  mem = 15;
  if (atomic_add_negative (&mem, -10)
      || mem != 5)
    {
      puts ("atomic_add_negative test 3 failed");
      ret = 1;
    }

  mem = -12;
  if (atomic_add_negative (&mem, 14)
      || mem != 2)
    {
      puts ("atomic_add_negative test 4 failed");
      ret = 1;
    }

  mem = 0;
  if (! atomic_add_negative (&mem, -1)
      || mem != -1)
    {
      puts ("atomic_add_negative test 5 failed");
      ret = 1;
    }

  mem = -31;
  if (atomic_add_negative (&mem, 31)
      || mem != 0)
    {
      puts ("atomic_add_negative test 6 failed");
      ret = 1;
    }

  mem = -34;
  if (atomic_add_zero (&mem, 31)
      || mem != -3)
    {
      puts ("atomic_add_zero test 1 failed");
      ret = 1;
    }

  mem = -36;
  if (! atomic_add_zero (&mem, 36)
      || mem != 0)
    {
      puts ("atomic_add_zero test 2 failed");
      ret = 1;
    }

  mem = 113;
  if (atomic_add_zero (&mem, -13)
      || mem != 100)
    {
      puts ("atomic_add_zero test 3 failed");
      ret = 1;
    }

  mem = -18;
  if (atomic_add_zero (&mem, 20)
      || mem != 2)
    {
      puts ("atomic_add_zero test 4 failed");
      ret = 1;
    }

  mem = 10;
  if (atomic_add_zero (&mem, -20)
      || mem != -10)
    {
      puts ("atomic_add_zero test 5 failed");
      ret = 1;
    }

  mem = 10;
  if (! atomic_add_zero (&mem, -10)
      || mem != 0)
    {
      puts ("atomic_add_zero test 6 failed");
      ret = 1;
    }

  mem = 0;
  atomic_bit_set (&mem, 1);
  if (mem != 2)
    {
      puts ("atomic_bit_set test 1 failed");
      ret = 1;
    }

  mem = 8;
  atomic_bit_set (&mem, 3);
  if (mem != 8)
    {
      puts ("atomic_bit_set test 2 failed");
      ret = 1;
    }

#ifdef TEST_ATOMIC64
  mem = 16;
  atomic_bit_set (&mem, 35);
  if (mem != 0x800000010LL)
    {
      puts ("atomic_bit_set test 3 failed");
      ret = 1;
    }
#endif

  mem = 0;
  if (atomic_bit_test_set (&mem, 1)
      || mem != 2)
    {
      puts ("atomic_bit_test_set test 1 failed");
      ret = 1;
    }

  mem = 8;
  if (! atomic_bit_test_set (&mem, 3)
      || mem != 8)
    {
      puts ("atomic_bit_test_set test 2 failed");
      ret = 1;
    }

#ifdef TEST_ATOMIC64
  mem = 16;
  if (atomic_bit_test_set (&mem, 35)
      || mem != 0x800000010LL)
    {
      puts ("atomic_bit_test_set test 3 failed");
      ret = 1;
    }

  mem = 0x100000000LL;
  if (! atomic_bit_test_set (&mem, 32)
      || mem != 0x100000000LL)
    {
      puts ("atomic_bit_test_set test 4 failed");
      ret = 1;
    }
#endif

#ifdef catomic_compare_and_exchange_val_acq
  mem = 24;
  if (catomic_compare_and_exchange_val_acq (&mem, 35, 24) != 24
      || mem != 35)
    {
      puts ("catomic_compare_and_exchange_val_acq test 1 failed");
      ret = 1;
    }

  mem = 12;
  if (catomic_compare_and_exchange_val_acq (&mem, 10, 15) != 12
      || mem != 12)
    {
      puts ("catomic_compare_and_exchange_val_acq test 2 failed");
      ret = 1;
    }

  mem = -15;
  if (catomic_compare_and_exchange_val_acq (&mem, -56, -15) != -15
      || mem != -56)
    {
      puts ("catomic_compare_and_exchange_val_acq test 3 failed");
      ret = 1;
    }

  mem = -1;
  if (catomic_compare_and_exchange_val_acq (&mem, 17, 0) != -1
      || mem != -1)
    {
      puts ("catomic_compare_and_exchange_val_acq test 4 failed");
      ret = 1;
    }
#endif

  mem = 24;
  if (catomic_compare_and_exchange_bool_acq (&mem, 35, 24)
      || mem != 35)
    {
      puts ("catomic_compare_and_exchange_bool_acq test 1 failed");
      ret = 1;
    }

  mem = 12;
  if (! catomic_compare_and_exchange_bool_acq (&mem, 10, 15)
      || mem != 12)
    {
      puts ("catomic_compare_and_exchange_bool_acq test 2 failed");
      ret = 1;
    }

  mem = -15;
  if (catomic_compare_and_exchange_bool_acq (&mem, -56, -15)
      || mem != -56)
    {
      puts ("catomic_compare_and_exchange_bool_acq test 3 failed");
      ret = 1;
    }

  mem = -1;
  if (! catomic_compare_and_exchange_bool_acq (&mem, 17, 0)
      || mem != -1)
    {
      puts ("catomic_compare_and_exchange_bool_acq test 4 failed");
      ret = 1;
    }

  mem = 2;
  if (catomic_exchange_and_add (&mem, 11) != 2
      || mem != 13)
    {
      puts ("catomic_exchange_and_add test failed");
      ret = 1;
    }

  mem = -21;
  catomic_add (&mem, 22);
  if (mem != 1)
    {
      puts ("catomic_add test failed");
      ret = 1;
    }

  mem = -1;
  catomic_increment (&mem);
  if (mem != 0)
    {
      puts ("catomic_increment test failed");
      ret = 1;
    }

  mem = 2;
  if (catomic_increment_val (&mem) != 3)
    {
      puts ("catomic_increment_val test failed");
      ret = 1;
    }

  mem = 17;
  catomic_decrement (&mem);
  if (mem != 16)
    {
      puts ("catomic_decrement test failed");
      ret = 1;
    }

  if (catomic_decrement_val (&mem) != 15)
    {
      puts ("catomic_decrement_val test failed");
      ret = 1;
    }

  /* Tests for C11-like atomics.  */
  mem = 11;
  if (atomic_load_relaxed (&mem) != 11 || atomic_load_acquire (&mem) != 11)
    {
      puts ("atomic_load_{relaxed,acquire} test failed");
      ret = 1;
    }

  atomic_store_relaxed (&mem, 12);
  if (mem != 12)
    {
      puts ("atomic_store_relaxed test failed");
      ret = 1;
    }
  atomic_store_release (&mem, 13);
  if (mem != 13)
    {
      puts ("atomic_store_release test failed");
      ret = 1;
    }

  mem = 14;
  expected = 14;
  if (!atomic_compare_exchange_weak_relaxed (&mem, &expected, 25)
      || mem != 25 || expected != 14)
    {
      puts ("atomic_compare_exchange_weak_relaxed test 1 failed");
      ret = 1;
    }
  if (atomic_compare_exchange_weak_relaxed (&mem, &expected, 14)
      || mem != 25 || expected != 25)
    {
      puts ("atomic_compare_exchange_weak_relaxed test 2 failed");
      ret = 1;
    }
  mem = 14;
  expected = 14;
  if (!atomic_compare_exchange_weak_acquire (&mem, &expected, 25)
      || mem != 25 || expected != 14)
    {
      puts ("atomic_compare_exchange_weak_acquire test 1 failed");
      ret = 1;
    }
  if (atomic_compare_exchange_weak_acquire (&mem, &expected, 14)
      || mem != 25 || expected != 25)
    {
      puts ("atomic_compare_exchange_weak_acquire test 2 failed");
      ret = 1;
    }
  mem = 14;
  expected = 14;
  if (!atomic_compare_exchange_weak_release (&mem, &expected, 25)
      || mem != 25 || expected != 14)
    {
      puts ("atomic_compare_exchange_weak_release test 1 failed");
      ret = 1;
    }
  if (atomic_compare_exchange_weak_release (&mem, &expected, 14)
      || mem != 25 || expected != 25)
    {
      puts ("atomic_compare_exchange_weak_release test 2 failed");
      ret = 1;
    }

  mem = 23;
  if (atomic_exchange_acquire (&mem, 42) != 23 || mem != 42)
    {
      puts ("atomic_exchange_acquire test failed");
      ret = 1;
    }
  mem = 23;
  if (atomic_exchange_release (&mem, 42) != 23 || mem != 42)
    {
      puts ("atomic_exchange_release test failed");
      ret = 1;
    }

  mem = 23;
  if (atomic_fetch_add_relaxed (&mem, 1) != 23 || mem != 24)
    {
      puts ("atomic_fetch_add_relaxed test failed");
      ret = 1;
    }
  mem = 23;
  if (atomic_fetch_add_acquire (&mem, 1) != 23 || mem != 24)
    {
      puts ("atomic_fetch_add_acquire test failed");
      ret = 1;
    }
  mem = 23;
  if (atomic_fetch_add_release (&mem, 1) != 23 || mem != 24)
    {
      puts ("atomic_fetch_add_release test failed");
      ret = 1;
    }
  mem = 23;
  if (atomic_fetch_add_acq_rel (&mem, 1) != 23 || mem != 24)
    {
      puts ("atomic_fetch_add_acq_rel test failed");
      ret = 1;
    }

  mem = 3;
  if (atomic_fetch_and_acquire (&mem, 2) != 3 || mem != 2)
    {
      puts ("atomic_fetch_and_acquire test failed");
      ret = 1;
    }

  mem = 4;
  if (atomic_fetch_or_relaxed (&mem, 2) != 4 || mem != 6)
    {
      puts ("atomic_fetch_or_relaxed test failed");
      ret = 1;
    }
  mem = 4;
  if (atomic_fetch_or_acquire (&mem, 2) != 4 || mem != 6)
    {
      puts ("atomic_fetch_or_acquire test failed");
      ret = 1;
    }

  /* This is a single-threaded test, so we can't test the effects of the
     fences.  */
  atomic_thread_fence_acquire ();
  atomic_thread_fence_release ();
  atomic_thread_fence_seq_cst ();

  return ret;
}

#include <support/test-driver.c>
