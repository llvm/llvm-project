/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Jaeger <aj@suse.de> and
   Ulrich Drepper <drepper@cygnus.com>, 1997.

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

/* Tests for ISO C99 7.6: Floating-point environment  */

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif

#include <complex.h>
#include <math.h>
#include <float.h>
#include <fenv.h>

#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <math-tests.h>

/*
  Since not all architectures might define all exceptions, we define
  a private set and map accordingly.
*/
#define NO_EXC 0
#define INEXACT_EXC 0x1
#define DIVBYZERO_EXC 0x2
#define UNDERFLOW_EXC 0x04
#define OVERFLOW_EXC 0x08
#define INVALID_EXC 0x10
#define ALL_EXC \
        (INEXACT_EXC | DIVBYZERO_EXC | UNDERFLOW_EXC | OVERFLOW_EXC \
         | INVALID_EXC)

static int count_errors;

#if FE_ALL_EXCEPT
/* Test whether a given exception was raised.  */
static void
test_single_exception (short int exception,
                       short int exc_flag,
                       fexcept_t fe_flag,
                       const char *flag_name)
{
  if (exception & exc_flag)
    {
      if (fetestexcept (fe_flag))
        printf ("  Pass: Exception \"%s\" is set\n", flag_name);
      else
        {
          printf ("  Fail: Exception \"%s\" is not set\n", flag_name);
          ++count_errors;
        }
    }
  else
    {
      if (fetestexcept (fe_flag))
        {
          printf ("  Fail: Exception \"%s\" is set\n", flag_name);
          ++count_errors;
        }
      else
        {
          printf ("  Pass: Exception \"%s\" is not set\n", flag_name);
        }
    }
}
#endif

static void
test_exceptions (const char *test_name, short int exception,
		 int ignore_inexact)
{
  printf ("Test: %s\n", test_name);
#ifdef FE_DIVBYZERO
  test_single_exception (exception, DIVBYZERO_EXC, FE_DIVBYZERO,
                         "DIVBYZERO");
#endif
#ifdef FE_INVALID
  test_single_exception (exception, INVALID_EXC, FE_INVALID,
                         "INVALID");
#endif
#ifdef FE_INEXACT
  if (!ignore_inexact)
    test_single_exception (exception, INEXACT_EXC, FE_INEXACT,
			   "INEXACT");
#endif
#ifdef FE_UNDERFLOW
  test_single_exception (exception, UNDERFLOW_EXC, FE_UNDERFLOW,
                         "UNDERFLOW");
#endif
#ifdef FE_OVERFLOW
  test_single_exception (exception, OVERFLOW_EXC, FE_OVERFLOW,
                         "OVERFLOW");
#endif
}

static void
print_rounding (int rounding)
{

  switch (rounding)
    {
#ifdef FE_TONEAREST
    case FE_TONEAREST:
      printf ("TONEAREST");
      break;
#endif
#ifdef FE_UPWARD
    case FE_UPWARD:
      printf ("UPWARD");
      break;
#endif
#ifdef FE_DOWNWARD
    case FE_DOWNWARD:
      printf ("DOWNWARD");
      break;
#endif
#ifdef FE_TOWARDZERO
    case FE_TOWARDZERO:
      printf ("TOWARDZERO");
      break;
#endif
    }
  printf (".\n");
}


static void
test_rounding (const char *test_name, int rounding_mode)
{
  int curr_rounding = fegetround ();

  printf ("Test: %s\n", test_name);
  if (curr_rounding == rounding_mode)
    {
      printf ("  Pass: Rounding mode is ");
      print_rounding (curr_rounding);
    }
  else
    {
      ++count_errors;
      printf ("  Fail: Rounding mode is ");
      print_rounding (curr_rounding);
    }
}


#if FE_ALL_EXCEPT
static void
set_single_exc (const char *test_name, int fe_exc, fexcept_t exception)
{
  char str[200];
  /* The standard allows the inexact exception to be set together with the
     underflow and overflow exceptions.  So ignore the inexact flag if the
     others are raised.  */
  int ignore_inexact = (fe_exc & (UNDERFLOW_EXC | OVERFLOW_EXC)) != 0;

  strcpy (str, test_name);
  strcat (str, ": set flag, with rest not set");
  feclearexcept (FE_ALL_EXCEPT);
  feraiseexcept (exception);
  test_exceptions (str, fe_exc, ignore_inexact);

  strcpy (str, test_name);
  strcat (str, ": clear flag, rest also unset");
  feclearexcept (exception);
  test_exceptions (str, NO_EXC, ignore_inexact);

  strcpy (str, test_name);
  strcat (str, ": set flag, with rest set");
  feraiseexcept (FE_ALL_EXCEPT ^ exception);
  feraiseexcept (exception);
  test_exceptions (str, ALL_EXC, 0);

  strcpy (str, test_name);
  strcat (str, ": clear flag, leave rest set");
  feclearexcept (exception);
  test_exceptions (str, ALL_EXC ^ fe_exc, 0);
}
#endif

static void
fe_tests (void)
{
  /* clear all exceptions and test if all are cleared */
  feclearexcept (FE_ALL_EXCEPT);
  test_exceptions ("feclearexcept (FE_ALL_EXCEPT) clears all exceptions",
                   NO_EXC, 0);

  /* Skip further tests here if exceptions not supported.  */
  if (!EXCEPTION_TESTS (float) && FE_ALL_EXCEPT != 0)
    return;
  /* raise all exceptions and test if all are raised */
  feraiseexcept (FE_ALL_EXCEPT);
  test_exceptions ("feraiseexcept (FE_ALL_EXCEPT) raises all exceptions",
                   ALL_EXC, 0);
  feclearexcept (FE_ALL_EXCEPT);

#ifdef FE_DIVBYZERO
  set_single_exc ("Set/Clear FE_DIVBYZERO", DIVBYZERO_EXC, FE_DIVBYZERO);
#endif
#ifdef FE_INVALID
  set_single_exc ("Set/Clear FE_INVALID", INVALID_EXC, FE_INVALID);
#endif
#ifdef FE_INEXACT
  set_single_exc ("Set/Clear FE_INEXACT", INEXACT_EXC, FE_INEXACT);
#endif
#ifdef FE_UNDERFLOW
  set_single_exc ("Set/Clear FE_UNDERFLOW", UNDERFLOW_EXC, FE_UNDERFLOW);
#endif
#ifdef FE_OVERFLOW
  set_single_exc ("Set/Clear FE_OVERFLOW", OVERFLOW_EXC, FE_OVERFLOW);
#endif
}

#if FE_ALL_EXCEPT
/* Test that program aborts with no masked interrupts */
static void
feenv_nomask_test (const char *flag_name, int fe_exc)
{
# if defined FE_NOMASK_ENV
  int status;
  pid_t pid;

  if (!EXCEPTION_ENABLE_SUPPORTED (FE_ALL_EXCEPT)
      && fesetenv (FE_NOMASK_ENV) != 0)
    {
      printf ("Test: not testing FE_NOMASK_ENV, it isn't implemented.\n");
      return;
    }

  printf ("Test: after fesetenv (FE_NOMASK_ENV) processes will abort\n");
  printf ("      when feraiseexcept (%s) is called.\n", flag_name);
  pid = fork ();
  if (pid == 0)
    {
#  ifdef RLIMIT_CORE
      /* Try to avoid dumping core.  */
      struct rlimit core_limit;
      core_limit.rlim_cur = 0;
      core_limit.rlim_max = 0;
      setrlimit (RLIMIT_CORE, &core_limit);
#  endif

      fesetenv (FE_NOMASK_ENV);
      feraiseexcept (fe_exc);
      exit (2);
    }
  else if (pid < 0)
    {
      if (errno != ENOSYS)
	{
	  printf ("  Fail: Could not fork.\n");
	  ++count_errors;
	}
      else
	printf ("  `fork' not implemented, test ignored.\n");
    }
  else {
    if (waitpid (pid, &status, 0) != pid)
      {
	printf ("  Fail: waitpid call failed.\n");
	++count_errors;
      }
    else if (WIFSIGNALED (status) && WTERMSIG (status) == SIGFPE)
      printf ("  Pass: Process received SIGFPE.\n");
    else
      {
	printf ("  Fail: Process didn't receive signal and exited with status %d.\n",
		status);
	++count_errors;
      }
  }
# endif
}

/* Test that program doesn't abort with default environment */
static void
feenv_mask_test (const char *flag_name, int fe_exc)
{
  int status;
  pid_t pid;

  printf ("Test: after fesetenv (FE_DFL_ENV) processes will not abort\n");
  printf ("      when feraiseexcept (%s) is called.\n", flag_name);
  pid = fork ();
  if (pid == 0)
    {
#ifdef RLIMIT_CORE
      /* Try to avoid dumping core.  */
      struct rlimit core_limit;
      core_limit.rlim_cur = 0;
      core_limit.rlim_max = 0;
      setrlimit (RLIMIT_CORE, &core_limit);
#endif

      fesetenv (FE_DFL_ENV);
      feraiseexcept (fe_exc);
      exit (2);
    }
  else if (pid < 0)
    {
      if (errno != ENOSYS)
	{
	  printf ("  Fail: Could not fork.\n");
	  ++count_errors;
	}
      else
	printf ("  `fork' not implemented, test ignored.\n");
    }
  else {
    if (waitpid (pid, &status, 0) != pid)
      {
	printf ("  Fail: waitpid call failed.\n");
	++count_errors;
      }
    else if (WIFEXITED (status) && WEXITSTATUS (status) == 2)
      printf ("  Pass: Process exited normally.\n");
    else
      {
	printf ("  Fail: Process exited abnormally with status %d.\n",
		status);
	++count_errors;
      }
  }
}

/* Test that program aborts with no masked interrupts */
static void
feexcp_nomask_test (const char *flag_name, int fe_exc)
{
  int status;
  pid_t pid;

  if (!EXCEPTION_ENABLE_SUPPORTED (fe_exc) && feenableexcept (fe_exc) == -1)
    {
      printf ("Test: not testing feenableexcept, it isn't implemented.\n");
      return;
    }

  printf ("Test: after feenableexcept (%s) processes will abort\n",
	  flag_name);
  printf ("      when feraiseexcept (%s) is called.\n", flag_name);
  pid = fork ();
  if (pid == 0)
    {
#ifdef RLIMIT_CORE
      /* Try to avoid dumping core.  */
      struct rlimit core_limit;
      core_limit.rlim_cur = 0;
      core_limit.rlim_max = 0;
      setrlimit (RLIMIT_CORE, &core_limit);
#endif

      fedisableexcept (FE_ALL_EXCEPT);
      feenableexcept (fe_exc);
      feraiseexcept (fe_exc);
      exit (2);
    }
  else if (pid < 0)
    {
      if (errno != ENOSYS)
	{
	  printf ("  Fail: Could not fork.\n");
	  ++count_errors;
	}
      else
	printf ("  `fork' not implemented, test ignored.\n");
    }
  else {
    if (waitpid (pid, &status, 0) != pid)
      {
	printf ("  Fail: waitpid call failed.\n");
	++count_errors;
      }
    else if (WIFSIGNALED (status) && WTERMSIG (status) == SIGFPE)
      printf ("  Pass: Process received SIGFPE.\n");
    else
      {
	printf ("  Fail: Process didn't receive signal and exited with status %d.\n",
		status);
	++count_errors;
      }
  }
}

/* Test that program doesn't abort with exception.  */
static void
feexcp_mask_test (const char *flag_name, int fe_exc)
{
  int status;
  int exception;
  pid_t pid;

  printf ("Test: after fedisableexcept (%s) processes will not abort\n",
	  flag_name);
  printf ("      when feraiseexcept (%s) is called.\n", flag_name);
  pid = fork ();
  if (pid == 0)
    {
#ifdef RLIMIT_CORE
      /* Try to avoid dumping core.  */
      struct rlimit core_limit;
      core_limit.rlim_cur = 0;
      core_limit.rlim_max = 0;
      setrlimit (RLIMIT_CORE, &core_limit);
#endif
      feenableexcept (FE_ALL_EXCEPT);
      exception = fe_exc;
#ifdef FE_INEXACT
      /* The standard allows the inexact exception to be set together with the
	 underflow and overflow exceptions.  So add FE_INEXACT to the set of
	 exceptions to be disabled if we will be raising underflow or
	 overflow.  */
# ifdef FE_OVERFLOW
      if (fe_exc & FE_OVERFLOW)
	exception |= FE_INEXACT;
# endif
# ifdef FE_UNDERFLOW
      if (fe_exc & FE_UNDERFLOW)
	exception |= FE_INEXACT;
# endif
#endif
      fedisableexcept (exception);
      feraiseexcept (fe_exc);
      exit (2);
    }
  else if (pid < 0)
    {
      if (errno != ENOSYS)
	{
	  printf ("  Fail: Could not fork.\n");
	  ++count_errors;
	}
      else
	printf ("  `fork' not implemented, test ignored.\n");
    }
  else {
    if (waitpid (pid, &status, 0) != pid)
      {
	printf ("  Fail: waitpid call failed.\n");
	++count_errors;
      }
    else if (WIFEXITED (status) && WEXITSTATUS (status) == 2)
      printf ("  Pass: Process exited normally.\n");
    else
      {
	printf ("  Fail: Process exited abnormally with status %d.\n",
		status);
	++count_errors;
      }
  }
}


/* Tests for feenableexcept/fedisableexcept/fegetexcept.  */
static void
feenable_test (const char *flag_name, int fe_exc)
{
  int excepts;

  printf ("Tests for feenableexcepts etc. with flag %s\n", flag_name);

  /* First disable all exceptions.  */
  if (fedisableexcept (FE_ALL_EXCEPT) == -1)
    {
      printf ("Test: fedisableexcept (FE_ALL_EXCEPT) failed\n");
      ++count_errors;
      /* If this fails, the other tests don't make sense.  */
      return;
    }
  excepts = fegetexcept ();
  if (excepts != 0)
    {
      printf ("Test: fegetexcept (%s) failed, return should be 0, is %d\n",
	      flag_name, excepts);
      ++count_errors;
    }
  excepts = feenableexcept (fe_exc);
  if (!EXCEPTION_ENABLE_SUPPORTED (fe_exc) && excepts == -1)
    {
      printf ("Test: not testing feenableexcept, it isn't implemented.\n");
      return;
    }
  if (excepts == -1)
    {
      printf ("Test: feenableexcept (%s) failed\n", flag_name);
      ++count_errors;
      return;
    }
  if (excepts != 0)
    {
      printf ("Test: feenableexcept (%s) failed, return should be 0, is %x\n",
	      flag_name, excepts);
      ++count_errors;
    }

  excepts = fegetexcept ();
  if (excepts != fe_exc)
    {
      printf ("Test: fegetexcept (%s) failed, return should be 0x%x, is 0x%x\n",
	      flag_name, fe_exc, excepts);
      ++count_errors;
    }

  /* And now disable the exception again.  */
  excepts = fedisableexcept (fe_exc);
  if (excepts == -1)
    {
      printf ("Test: fedisableexcept (%s) failed\n", flag_name);
      ++count_errors;
      return;
    }
  if (excepts != fe_exc)
    {
      printf ("Test: fedisableexcept (%s) failed, return should be 0x%x, is 0x%x\n",
	      flag_name, fe_exc, excepts);
      ++count_errors;
    }

  excepts = fegetexcept ();
  if (excepts != 0)
    {
      printf ("Test: fegetexcept (%s) failed, return should be 0, is 0x%x\n",
	      flag_name, excepts);
      ++count_errors;
    }

  /* Now the other way round: Enable all exceptions and disable just this one.  */
  if (feenableexcept (FE_ALL_EXCEPT) == -1)
    {
      printf ("Test: feenableexcept (FE_ALL_EXCEPT) failed\n");
      ++count_errors;
      /* If this fails, the other tests don't make sense.  */
      return;
    }

  excepts = fegetexcept ();
  if (excepts != FE_ALL_EXCEPT)
    {
      printf ("Test: fegetexcept (%s) failed, return should be 0x%x, is 0x%x\n",
	      flag_name, FE_ALL_EXCEPT, excepts);
      ++count_errors;
    }

  excepts = fedisableexcept (fe_exc);
  if (excepts == -1)
    {
      printf ("Test: fedisableexcept (%s) failed\n", flag_name);
      ++count_errors;
      return;
    }
  if (excepts != FE_ALL_EXCEPT)
    {
      printf ("Test: fedisableexcept (%s) failed, return should be 0, is 0x%x\n",
	      flag_name, excepts);
      ++count_errors;
    }

  excepts = fegetexcept ();
  if (excepts != (FE_ALL_EXCEPT & ~fe_exc))
    {
      printf ("Test: fegetexcept (%s) failed, return should be 0x%x, is 0x%x\n",
	      flag_name, (FE_ALL_EXCEPT & ~fe_exc), excepts);
      ++count_errors;
    }

  /* And now enable the exception again.  */
  excepts = feenableexcept (fe_exc);
  if (excepts == -1)
    {
      printf ("Test: feenableexcept (%s) failed\n", flag_name);
      ++count_errors;
      return;
    }
  if (excepts != (FE_ALL_EXCEPT & ~fe_exc))
    {
      printf ("Test: feenableexcept (%s) failed, return should be 0, is 0x%x\n",
	      flag_name, excepts);
      ++count_errors;
    }

  excepts = fegetexcept ();
  if (excepts != FE_ALL_EXCEPT)
    {
      printf ("Test: fegetexcept (%s) failed, return should be 0x%x, is 0x%x\n",
	      flag_name, FE_ALL_EXCEPT, excepts);
      ++count_errors;
    }
  feexcp_nomask_test (flag_name, fe_exc);
  feexcp_mask_test (flag_name, fe_exc);

}


static void
fe_single_test (const char *flag_name, int fe_exc)
{
  feenv_nomask_test (flag_name, fe_exc);
  feenv_mask_test (flag_name, fe_exc);
  feenable_test (flag_name, fe_exc);
}
#endif


static void
feenv_tests (void)
{
  /* We might have some exceptions still set.  */
  feclearexcept (FE_ALL_EXCEPT);

#ifdef FE_DIVBYZERO
  fe_single_test ("FE_DIVBYZERO", FE_DIVBYZERO);
#endif
#ifdef FE_INVALID
  fe_single_test ("FE_INVALID", FE_INVALID);
#endif
#ifdef FE_INEXACT
  fe_single_test ("FE_INEXACT", FE_INEXACT);
#endif
#ifdef FE_UNDERFLOW
  fe_single_test ("FE_UNDERFLOW", FE_UNDERFLOW);
#endif
#ifdef FE_OVERFLOW
  fe_single_test ("FE_OVERFLOW", FE_OVERFLOW);
#endif
  fesetenv (FE_DFL_ENV);
}


static void
feholdexcept_tests (void)
{
  fenv_t saved, saved2;
  int res;

  feclearexcept (FE_ALL_EXCEPT);
  fedisableexcept (FE_ALL_EXCEPT);
#ifdef FE_DIVBYZERO
  feraiseexcept (FE_DIVBYZERO);
#endif
  if (EXCEPTION_TESTS (float))
    test_exceptions ("feholdexcept_tests FE_DIVBYZERO test",
		     DIVBYZERO_EXC, 0);
  res = feholdexcept (&saved);
  if (res != 0)
    {
      printf ("feholdexcept failed: %d\n", res);
      ++count_errors;
    }
#if defined FE_TONEAREST && defined FE_TOWARDZERO
  res = fesetround (FE_TOWARDZERO);
  if (res != 0 && ROUNDING_TESTS (float, FE_TOWARDZERO))
    {
      printf ("fesetround failed: %d\n", res);
      ++count_errors;
    }
#endif
  test_exceptions ("feholdexcept_tests 0 test", NO_EXC, 0);
#ifdef FE_INVALID
  feraiseexcept (FE_INVALID);
  if (EXCEPTION_TESTS (float))
    test_exceptions ("feholdexcept_tests FE_INVALID test",
		     INVALID_EXC, 0);
#endif
  res = feupdateenv (&saved);
  if (res != 0)
    {
      printf ("feupdateenv failed: %d\n", res);
      ++count_errors;
    }
#if defined FE_TONEAREST && defined FE_TOWARDZERO
  res = fegetround ();
  if (res != FE_TONEAREST)
    {
      printf ("feupdateenv didn't restore rounding mode: %d\n", res);
      ++count_errors;
    }
#endif
  if (EXCEPTION_TESTS (float))
    test_exceptions ("feholdexcept_tests FE_DIVBYZERO|FE_INVALID test",
		     DIVBYZERO_EXC | INVALID_EXC, 0);
  feclearexcept (FE_ALL_EXCEPT);
#ifdef FE_INVALID
  feraiseexcept (FE_INVALID);
#endif
#if defined FE_TONEAREST && defined FE_UPWARD
  res = fesetround (FE_UPWARD);
  if (res != 0 && ROUNDING_TESTS (float, FE_UPWARD))
    {
      printf ("fesetround failed: %d\n", res);
      ++count_errors;
    }
#endif
  res = feholdexcept (&saved2);
  if (res != 0)
    {
      printf ("feholdexcept failed: %d\n", res);
      ++count_errors;
    }
#if defined FE_TONEAREST && defined FE_UPWARD
  res = fesetround (FE_TONEAREST);
  if (res != 0)
    {
      printf ("fesetround failed: %d\n", res);
      ++count_errors;
    }
#endif
  test_exceptions ("feholdexcept_tests 0 2nd test", NO_EXC, 0);
#ifdef FE_INEXACT
  feraiseexcept (FE_INEXACT);
  if (EXCEPTION_TESTS (float))
    test_exceptions ("feholdexcept_tests FE_INEXACT test",
		     INEXACT_EXC, 0);
#endif
  res = feupdateenv (&saved2);
  if (res != 0)
    {
      printf ("feupdateenv failed: %d\n", res);
      ++count_errors;
    }
#if defined FE_TONEAREST && defined FE_UPWARD
  res = fegetround ();
  if (res != FE_UPWARD && ROUNDING_TESTS (float, FE_UPWARD))
    {
      printf ("feupdateenv didn't restore rounding mode: %d\n", res);
      ++count_errors;
    }
  fesetround (FE_TONEAREST);
#endif
  if (EXCEPTION_TESTS (float))
    test_exceptions ("feholdexcept_tests FE_INEXACT|FE_INVALID test",
		     INVALID_EXC | INEXACT_EXC, 0);
  feclearexcept (FE_ALL_EXCEPT);
}


/* IEC 559 and ISO C99 define a default startup environment */
static void
initial_tests (void)
{
  test_exceptions ("Initially all exceptions should be cleared",
                   NO_EXC, 0);
#ifdef FE_TONEAREST
  test_rounding ("Rounding direction should be initalized to nearest",
                 FE_TONEAREST);
#endif
}

int
main (void)
{
  initial_tests ();
  fe_tests ();
  feenv_tests ();
  feholdexcept_tests ();

  if (count_errors)
    {
      printf ("\n%d errors occurred.\n", count_errors);
      exit (1);
    }
  printf ("\n All tests passed successfully.\n");
  return 0;
}
