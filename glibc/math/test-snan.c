/* Test signaling NaNs in issignaling, isnan, isinf, and similar functions.
   Copyright (C) 2008-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Jaeger <aj@suse.de>, 2005.

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

#define _GNU_SOURCE 1
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <fenv.h>
#include <signal.h>
#include <setjmp.h>

#include <math-tests.h>


static sigjmp_buf sigfpe_buf;

static void
myFPsighandler (int signal)
{
  siglongjmp (sigfpe_buf, 1);
}

static int errors = 0;

#define CHECK(testname, expr)						      \
  do {									      \
    feclearexcept (FE_ALL_EXCEPT);					      \
    feenableexcept (FE_ALL_EXCEPT);					      \
    if (sigsetjmp (sigfpe_buf, 0))					      \
      {									      \
	printf ("%s raised SIGFPE\n", testname);			      \
	++errors;							      \
      }									      \
    else if (!(expr))							      \
      {									      \
        printf ("Failure: %s\n", testname);				      \
        ++errors;							      \
      }									      \
  } while (0)

#define TEST_FUNC(NAME, FLOAT, SUFFIX)					      \
static void								      \
NAME (void)								      \
{									      \
  /* Variables are declared volatile to forbid some compiler		      \
     optimizations.  */							      \
  volatile FLOAT Inf_var, qNaN_var, zero_var, one_var;			      \
  /* A sNaN is only guaranteed to be representable in variables with */	      \
  /* static (or thread-local) storage duration.  */			      \
  static volatile FLOAT sNaN_var = __builtin_nans ## SUFFIX ("");	      \
  static volatile FLOAT minus_sNaN_var = -__builtin_nans ## SUFFIX ("");      \
  fenv_t saved_fenv;							      \
									      \
  zero_var = 0.0;							      \
  one_var = 1.0;							      \
  qNaN_var = __builtin_nan ## SUFFIX ("");				      \
  Inf_var = one_var / zero_var;						      \
									      \
  (void) &zero_var;							      \
  (void) &one_var;							      \
  (void) &qNaN_var;							      \
  (void) &sNaN_var;							      \
  (void) &minus_sNaN_var;						      \
  (void) &Inf_var;							      \
									      \
  fegetenv (&saved_fenv);						      \
									      \
  CHECK (#FLOAT " issignaling (qNaN)", !issignaling (qNaN_var));	      \
  CHECK (#FLOAT " issignaling (-qNaN)", !issignaling (-qNaN_var));	      \
  CHECK (#FLOAT " issignaling (sNaN)",					      \
         SNAN_TESTS (FLOAT) ? issignaling (sNaN_var) : 1);		      \
  CHECK (#FLOAT " issignaling (-sNaN)",					      \
         SNAN_TESTS (FLOAT) ? issignaling (minus_sNaN_var) : 1);	      \
  CHECK (#FLOAT " isnan (qNaN)", isnan (qNaN_var));			      \
  CHECK (#FLOAT " isnan (-qNaN)", isnan (-qNaN_var));			      \
  CHECK (#FLOAT " isnan (sNaN)",					      \
         SNAN_TESTS (FLOAT) ? isnan (sNaN_var) : 1);			      \
  CHECK (#FLOAT " isnan (-sNaN)",					      \
         SNAN_TESTS (FLOAT) ? isnan (minus_sNaN_var) : 1);		      \
  CHECK (#FLOAT " isinf (qNaN)", !isinf (qNaN_var));			      \
  CHECK (#FLOAT " isinf (-qNaN)", !isinf (-qNaN_var));			      \
  CHECK (#FLOAT " isinf (sNaN)",					      \
         SNAN_TESTS (FLOAT) ? !isinf (sNaN_var) : 1);			      \
  CHECK (#FLOAT " isinf (-sNaN)",					      \
         SNAN_TESTS (FLOAT) ? !isinf (minus_sNaN_var) : 1);		      \
  CHECK (#FLOAT " isfinite (qNaN)", !isfinite (qNaN_var));		      \
  CHECK (#FLOAT " isfinite (-qNaN)", !isfinite (-qNaN_var));		      \
  CHECK (#FLOAT " isfinite (sNaN)",					      \
         SNAN_TESTS (FLOAT) ? !isfinite (sNaN_var) : 1);		      \
  CHECK (#FLOAT " isfinite (-sNaN)",					      \
         SNAN_TESTS (FLOAT) ? !isfinite (minus_sNaN_var) : 1);		      \
  CHECK (#FLOAT " isnormal (qNaN)", !isnormal (qNaN_var));		      \
  CHECK (#FLOAT " isnormal (-qNaN)", !isnormal (-qNaN_var));		      \
  CHECK (#FLOAT " isnormal (sNaN)",					      \
         SNAN_TESTS (FLOAT) ? !isnormal (sNaN_var) : 1);		      \
  CHECK (#FLOAT " isnormal (-sNaN)",					      \
         SNAN_TESTS (FLOAT) ? !isnormal (minus_sNaN_var) : 1);		      \
  CHECK (#FLOAT " fpclassify (qNaN)", (fpclassify (qNaN_var)==FP_NAN));	      \
  CHECK (#FLOAT " fpclassify (-qNaN)", (fpclassify (-qNaN_var)==FP_NAN));     \
  CHECK (#FLOAT " fpclassify (sNaN)",					      \
         SNAN_TESTS (FLOAT) ? fpclassify (sNaN_var) == FP_NAN : 1);	      \
  CHECK (#FLOAT " fpclassify (-sNaN)",					      \
         SNAN_TESTS (FLOAT) ? fpclassify (minus_sNaN_var) == FP_NAN : 1);     \
									      \
  fesetenv (&saved_fenv); /* restore saved fenv */			      \
}									      \

TEST_FUNC (float_test, float, f)
TEST_FUNC (double_test, double, )
TEST_FUNC (ldouble_test, long double, l)

static int
do_test (void)
{
  signal (SIGFPE, &myFPsighandler);

  float_test ();
  double_test ();
  ldouble_test ();

  return errors != 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
