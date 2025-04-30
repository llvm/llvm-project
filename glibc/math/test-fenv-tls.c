/* Test floating-point environment is thread-local.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#include <fenv.h>
#include <pthread.h>
#include <stdio.h>
#include <stdint.h>

#define TEST_ONE_RM(RM)						\
  do								\
    {								\
      if (fesetround (RM) == 0)					\
	{							\
	  rm = fegetround ();					\
	  if (rm != RM)						\
	    {							\
	      printf ("expected " #RM ", got %d\n", rm);	\
	      ret = 1;						\
	    }							\
	}							\
    }								\
  while (0)

static void *
test_round (void *arg)
{
  intptr_t ret = 0;
  for (int i = 0; i < 10000; i++)
    {
      int rm;
#ifdef FE_DOWNWARD
      TEST_ONE_RM (FE_DOWNWARD);
#endif
#ifdef FE_TONEAREST
      TEST_ONE_RM (FE_TONEAREST);
#endif
#ifdef FE_TOWARDZERO
      TEST_ONE_RM (FE_TOWARDZERO);
#endif
#ifdef FE_UPWARD
      TEST_ONE_RM (FE_UPWARD);
#endif
    }
  return (void *) ret;
}

#define TEST_ONE_RAISE(EX)				\
  do							\
    {							\
      if (feraiseexcept (EX) == 0)			\
	if (fetestexcept (EX) != EX)			\
	  {						\
	    printf (#EX " not raised\n");		\
	    ret = 1;					\
	  }						\
      if (feclearexcept (FE_ALL_EXCEPT) == 0)		\
	if (fetestexcept (FE_ALL_EXCEPT) != 0)		\
	  {						\
	    printf ("exceptions not all cleared\n");	\
	    ret = 1;					\
	  }						\
    }							\
  while (0)

static void *
test_raise (void *arg)
{
  intptr_t ret = 0;
  for (int i = 0; i < 10000; i++)
    {
#ifdef FE_DIVBYZERO
      TEST_ONE_RAISE (FE_DIVBYZERO);
#endif
#ifdef FE_INEXACT
      TEST_ONE_RAISE (FE_INEXACT);
#endif
#ifdef FE_INVALID
      TEST_ONE_RAISE (FE_INVALID);
#endif
#ifdef FE_OVERFLOW
      TEST_ONE_RAISE (FE_OVERFLOW);
#endif
#ifdef UNDERFLOW
      TEST_ONE_RAISE (FE_UNDERFLOW);
#endif
    }
  return (void *) ret;
}

#define TEST_ONE_ENABLE(EX)				\
  do							\
    {							\
      if (feenableexcept (EX) != -1)			\
	if (fegetexcept () != EX)			\
	  {						\
	    printf (#EX " not enabled\n");		\
	    ret = 1;					\
	  }						\
      if (fedisableexcept (EX) != -1)			\
	if (fegetexcept () != 0)			\
	  {						\
	    printf ("exceptions not all disabled\n");	\
	    ret = 1;					\
	  }						\
    }							\
  while (0)

static void *
test_enable (void *arg)
{
  intptr_t ret = 0;
  for (int i = 0; i < 10000; i++)
    {
#ifdef FE_DIVBYZERO
      TEST_ONE_ENABLE (FE_DIVBYZERO);
#endif
#ifdef FE_INEXACT
      TEST_ONE_ENABLE (FE_INEXACT);
#endif
#ifdef FE_INVALID
      TEST_ONE_ENABLE (FE_INVALID);
#endif
#ifdef FE_OVERFLOW
      TEST_ONE_ENABLE (FE_OVERFLOW);
#endif
#ifdef UNDERFLOW
      TEST_ONE_ENABLE (FE_UNDERFLOW);
#endif
    }
  return (void *) ret;
}

static int
do_test (void)
{
  int ret = 0;
  void *vret;
  pthread_t thread_id;
  int pret;

  pret = pthread_create (&thread_id, NULL, test_round, NULL);
  if (pret != 0)
    {
      printf ("pthread_create failed: %d\n", pret);
      return 1;
    }
  vret = test_round (NULL);
  ret |= (intptr_t) vret;
  pret = pthread_join (thread_id, &vret);
  if (pret != 0)
    {
      printf ("pthread_join failed: %d\n", pret);
      return 1;
    }
  ret |= (intptr_t) vret;

  pret = pthread_create (&thread_id, NULL, test_raise, NULL);
  if (pret != 0)
    {
      printf ("pthread_create failed: %d\n", pret);
      return 1;
    }
  vret = test_raise (NULL);
  ret |= (intptr_t) vret;
  pret = pthread_join (thread_id, &vret);
  if (pret != 0)
    {
      printf ("pthread_join failed: %d\n", pret);
      return 1;
    }
  ret |= (intptr_t) vret;

  pret = pthread_create (&thread_id, NULL, test_enable, NULL);
  if (pret != 0)
    {
      printf ("pthread_create failed: %d\n", pret);
      return 1;
    }
  vret = test_enable (NULL);
  ret |= (intptr_t) vret;
  pret = pthread_join (thread_id, &vret);
  if (pret != 0)
    {
      printf ("pthread_join failed: %d\n", pret);
      return 1;
    }
  ret |= (intptr_t) vret;

  return ret;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
