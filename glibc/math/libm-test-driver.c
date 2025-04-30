/* Support code for testing libm functions (driver).
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#include "libm-test-support.h"

#include <math-tests-arch.h>

/* Flags set by the including file.  */
const int flag_test_errno = TEST_ERRNO;
const int flag_test_exceptions = TEST_EXCEPTIONS;
const int flag_test_mathvec = TEST_MATHVEC;

#if TEST_NARROW
const int snan_tests_arg = SNAN_TESTS (ARG_FLOAT);
#else
const int snan_tests_arg = SNAN_TESTS (FLOAT);
#endif

#define STRX(x) #x
#define STR(x) STRX (x)
#define STR_FLOAT STR (FLOAT)
#define STR_ARG_FLOAT STR (ARG_FLOAT)
#define STR_VEC_LEN STR (VEC_LEN)

/* Informal description of the functions being tested.  */
#if TEST_MATHVEC
# define TEST_MSG "testing " STR_FLOAT " (vector length " STR_VEC_LEN ")\n"
#elif TEST_NARROW
# define TEST_MSG "testing " STR_FLOAT " (argument " STR_ARG_FLOAT ")\n"
#else
# define TEST_MSG "testing " STR_FLOAT " (without inline functions)\n"
#endif
const char test_msg[] = TEST_MSG;

/* Allow platforms without all rounding modes to test properly,
   assuming they provide an __FE_UNDEFINED in <bits/fenv.h> which
   causes fesetround() to return failure.  */
#ifndef FE_TONEAREST
# define FE_TONEAREST	__FE_UNDEFINED
#endif
#ifndef FE_TOWARDZERO
# define FE_TOWARDZERO	__FE_UNDEFINED
#endif
#ifndef FE_UPWARD
# define FE_UPWARD	__FE_UNDEFINED
#endif
#ifndef FE_DOWNWARD
# define FE_DOWNWARD	__FE_UNDEFINED
#endif

#define TEST_NAN_PAYLOAD_CANONICALIZE	(SNAN_TESTS_PRESERVE_PAYLOAD	\
					 ? TEST_NAN_PAYLOAD		\
					 : 0)

const char qtype_str[] = TYPE_STR;

/* Various constants derived from pi.  We must supply them precalculated for
   accuracy.  They are written as a series of postfix operations to keep
   them concise yet somewhat readable.  */

/* (pi * 3) / 4 */
#define lit_pi_3_m_4_d		LIT (2.356194490192344928846982537459627163)
/* pi * 3 / (4 * ln(10)) */
#define lit_pi_3_m_4_ln10_m_d	LIT (1.023282265381381010614337719073516828)
/* pi / (2 * ln(10)) */
#define lit_pi_2_ln10_m_d	LIT (0.682188176920920673742891812715677885)
/* pi / (4 * ln(10)) */
#define lit_pi_4_ln10_m_d	LIT (0.341094088460460336871445906357838943)
/* pi / ln(10) */
#define lit_pi_ln10_d		LIT (1.364376353841841347485783625431355770)
/* pi / 2 */
#define lit_pi_2_d		LITM (M_PI_2)
/* pi / 4 */
#define lit_pi_4_d		LITM (M_PI_4)
/* pi */
#define lit_pi			LITM (M_PI)

/* Other useful constants.  */

/* e */
#define lit_e			LITM (M_E)

#define plus_zero	LIT (0.0)
#define minus_zero	LIT (-0.0)
#define plus_infty	FUNC (__builtin_inf) ()
#define minus_infty	-(FUNC (__builtin_inf) ())
#define qnan_value_pl(S)	FUNC (__builtin_nan) (S)
#define qnan_value	qnan_value_pl ("")
#define snan_value_pl(S)	FUNC (__builtin_nans) (S)
#define snan_value	snan_value_pl ("")
#define max_value	TYPE_MAX
#define min_value	TYPE_MIN
#define min_subnorm_value TYPE_TRUE_MIN

#define arg_plus_zero	ARG_LIT (0.0)
#define arg_minus_zero	ARG_LIT (-0.0)
#define arg_plus_infty	ARG_FUNC (__builtin_inf) ()
#define arg_minus_infty	-(ARG_FUNC (__builtin_inf) ())
#define arg_qnan_value_pl(S)	ARG_FUNC (__builtin_nan) (S)
#define arg_qnan_value	arg_qnan_value_pl ("")
#define arg_snan_value_pl(S)	ARG_FUNC (__builtin_nans) (S)
#define arg_snan_value	arg_snan_value_pl ("")
#define arg_max_value	ARG_TYPE_MAX
#define arg_min_value	ARG_TYPE_MIN
#define arg_min_subnorm_value ARG_TYPE_TRUE_MIN

/* For nexttoward tests.  */
#define snan_value_ld	__builtin_nansl ("")

/* For pseudo-normal number tests.  */
#if TEST_COND_intel96
# include <math_ldbl.h>
#define pseudo_inf { .parts = { 0x00000000, 0x00000000, 0x7fff }}
#define pseudo_zero { .parts = { 0x00000000, 0x00000000, 0x0100 }}
#define pseudo_qnan { .parts = { 0x00000001, 0x00000000, 0x7fff }}
#define pseudo_snan { .parts = { 0x00000001, 0x40000000, 0x7fff }}
#define pseudo_unnormal { .parts = { 0x00000001, 0x40000000, 0x0100 }}
#endif

/* Structures for each kind of test.  */
/* Used for both RUN_TEST_LOOP_f_f and RUN_TEST_LOOP_fp_f.  */
struct test_f_f_data
{
  const char *arg_str;
  FLOAT arg;
  struct
  {
    FLOAT expected;
    int exceptions;
  } rd, rn, rz, ru;
};
struct test_ff_f_data
{
  const char *arg_str;
  FLOAT arg1, arg2;
  struct
  {
    FLOAT expected;
    int exceptions;
  } rd, rn, rz, ru;
};
/* Strictly speaking, a j type argument is one gen-libm-test.py will not
   attempt to muck with.  For now, it is only used to prevent it from
   mucking up an explicitly long double argument.  */
struct test_fj_f_data
{
  const char *arg_str;
  FLOAT arg1;
  long double arg2;
  struct
  {
    FLOAT expected;
    int exceptions;
  } rd, rn, rz, ru;
};
#ifdef ARG_FLOAT
struct test_aa_f_data
{
  const char *arg_str;
  ARG_FLOAT arg1, arg2;
  struct
  {
    FLOAT expected;
    int exceptions;
  } rd, rn, rz, ru;
};
#endif
struct test_fi_f_data
{
  const char *arg_str;
  FLOAT arg1;
  int arg2;
  struct
  {
    FLOAT expected;
    int exceptions;
  } rd, rn, rz, ru;
};
struct test_fl_f_data
{
  const char *arg_str;
  FLOAT arg1;
  long int arg2;
  struct
  {
    FLOAT expected;
    int exceptions;
  } rd, rn, rz, ru;
};
struct test_if_f_data
{
  const char *arg_str;
  int arg1;
  FLOAT arg2;
  struct
  {
    FLOAT expected;
    int exceptions;
  } rd, rn, rz, ru;
};
struct test_fff_f_data
{
  const char *arg_str;
  FLOAT arg1, arg2, arg3;
  struct
  {
    FLOAT expected;
    int exceptions;
  } rd, rn, rz, ru;
};
struct test_fiu_M_data
{
  const char *arg_str;
  FLOAT arg1;
  int arg2;
  unsigned int arg3;
  struct
  {
    intmax_t expected;
    int exceptions;
  } rd, rn, rz, ru;
};
struct test_fiu_U_data
{
  const char *arg_str;
  FLOAT arg1;
  int arg2;
  unsigned int arg3;
  struct
  {
    uintmax_t expected;
    int exceptions;
  } rd, rn, rz, ru;
};
struct test_c_f_data
{
  const char *arg_str;
  FLOAT argr, argc;
  struct
  {
    FLOAT expected;
    int exceptions;
  } rd, rn, rz, ru;
};
/* Used for both RUN_TEST_LOOP_f_f1 and RUN_TEST_LOOP_fI_f1.  */
struct test_f_f1_data
{
  const char *arg_str;
  FLOAT arg;
  struct
  {
    FLOAT expected;
    int exceptions;
    int extra_test;
    int extra_expected;
  } rd, rn, rz, ru;
};
struct test_fF_f1_data
{
  const char *arg_str;
  FLOAT arg;
  struct
  {
    FLOAT expected;
    int exceptions;
    int extra_test;
    FLOAT extra_expected;
  } rd, rn, rz, ru;
};
struct test_ffI_f1_data
{
  const char *arg_str;
  FLOAT arg1, arg2;
  struct
  {
    FLOAT expected;
    int exceptions;
    int extra_test;
    int extra_expected;
  } rd, rn, rz, ru;
};
struct test_c_c_data
{
  const char *arg_str;
  FLOAT argr, argc;
  struct
  {
    FLOAT expr, expc;
    int exceptions;
  } rd, rn, rz, ru;
};
struct test_cc_c_data
{
  const char *arg_str;
  FLOAT arg1r, arg1c, arg2r, arg2c;
  struct
  {
    FLOAT expr, expc;
    int exceptions;
  } rd, rn, rz, ru;
};
/* Used for all of RUN_TEST_LOOP_f_i, RUN_TEST_LOOP_f_i_tg,
   RUN_TEST_LOOP_f_b and RUN_TEST_LOOP_f_b_tg.  */
struct test_f_i_data
{
  const char *arg_str;
  FLOAT arg;
  struct
  {
    int expected;
    int exceptions;
  } rd, rn, rz, ru;
};
/* Used for RUN_TEST_LOOP_f_i_tg_u and RUN_TEST_LOOP_f_b_tg_u.  */
#if TEST_COND_intel96
struct test_j_i_data_u
{
  const char *arg_str;
  ieee_long_double_shape_type arg;
  struct
  {
    int expected;
    int exceptions;
  } rd, rn, rz, ru;
};
#endif
/* Used for RUN_TEST_LOOP_ff_b, RUN_TEST_LOOP_fpfp_b and
   RUN_TEST_LOOP_ff_i_tg.  */
struct test_ff_i_data
{
  const char *arg_str;
  FLOAT arg1, arg2;
  struct
  {
    int expected;
    int exceptions;
  } rd, rn, rz, ru;
};
struct test_f_l_data
{
  const char *arg_str;
  FLOAT arg;
  struct
  {
    long int expected;
    int exceptions;
  } rd, rn, rz, ru;
};
struct test_f_L_data
{
  const char *arg_str;
  FLOAT arg;
  struct
  {
    long long int expected;
    int exceptions;
  } rd, rn, rz, ru;
};
struct test_fFF_11_data
{
  const char *arg_str;
  FLOAT arg;
  struct
  {
    int exceptions;
    int extra1_test;
    FLOAT extra1_expected;
    int extra2_test;
    FLOAT extra2_expected;
  } rd, rn, rz, ru;
};
/* Used for both RUN_TEST_LOOP_Ff_b1 and RUN_TEST_LOOP_Ffp_b1.  */
struct test_Ff_b1_data
{
  const char *arg_str;
  FLOAT arg;
  struct
  {
    int expected;
    int exceptions;
    int extra_test;
    FLOAT extra_expected;
  } rd, rn, rz, ru;
};

/* Set the rounding mode, or restore the saved value.  */
#define IF_ROUND_INIT_	/* Empty.  */
#define IF_ROUND_INIT_FE_DOWNWARD		\
  int save_round_mode = fegetround ();		\
  if (ROUNDING_TESTS (FLOAT, FE_DOWNWARD)	\
      && !TEST_MATHVEC				\
      && fesetround (FE_DOWNWARD) == 0)
#define IF_ROUND_INIT_FE_TONEAREST		\
  int save_round_mode = fegetround ();		\
  if (ROUNDING_TESTS (FLOAT, FE_TONEAREST)	\
      && fesetround (FE_TONEAREST) == 0)
#define IF_ROUND_INIT_FE_TOWARDZERO		\
  int save_round_mode = fegetround ();		\
  if (ROUNDING_TESTS (FLOAT, FE_TOWARDZERO)	\
      && !TEST_MATHVEC				\
      && fesetround (FE_TOWARDZERO) == 0)
#define IF_ROUND_INIT_FE_UPWARD			\
  int save_round_mode = fegetround ();		\
  if (ROUNDING_TESTS (FLOAT, FE_UPWARD)		\
      && !TEST_MATHVEC				\
      && fesetround (FE_UPWARD) == 0)
#define ROUND_RESTORE_	/* Empty.  */
#define ROUND_RESTORE_FE_DOWNWARD		\
  fesetround (save_round_mode)
#define ROUND_RESTORE_FE_TONEAREST		\
  fesetround (save_round_mode)
#define ROUND_RESTORE_FE_TOWARDZERO		\
  fesetround (save_round_mode)
#define ROUND_RESTORE_FE_UPWARD			\
  fesetround (save_round_mode)

/* Field name to use for a given rounding mode.  */
#define RM_			rn
#define RM_FE_DOWNWARD		rd
#define RM_FE_TONEAREST		rn
#define RM_FE_TOWARDZERO	rz
#define RM_FE_UPWARD		ru

/* Common setup for an individual test.  */
#define COMMON_TEST_SETUP(ARG_STR)					\
  char *test_name;							\
  if (asprintf (&test_name, "%s (%s)", this_func, (ARG_STR)) == -1)	\
    abort ()

/* Setup for a test with an extra output.  */
#define EXTRA_OUTPUT_TEST_SETUP(ARG_STR, N)			\
  char *extra##N##_name;					\
  if (asprintf (&extra##N##_name, "%s (%s) extra output " #N,	\
		this_func, (ARG_STR)) == -1)			\
    abort ()

/* Common cleanup after an individual test.  */
#define COMMON_TEST_CLEANUP			\
  free (test_name)

/* Cleanup for a test with an extra output.  */
#define EXTRA_OUTPUT_TEST_CLEANUP(N)		\
  free (extra##N##_name)

/* Run an individual test, including any required setup and checking
   of results, or loop over all tests in an array.  */
#define RUN_TEST_f_f(ARG_STR, FUNC_NAME, ARG, EXPECTED,			\
		     EXCEPTIONS)					\
  do									\
    if (enable_test (EXCEPTIONS))					\
      {									\
	COMMON_TEST_SETUP (ARG_STR);					\
	check_float (test_name,	FUNC_TEST (FUNC_NAME) (ARG),		\
		     EXPECTED, EXCEPTIONS);				\
	COMMON_TEST_CLEANUP;						\
      }									\
  while (0)
#define RUN_TEST_LOOP_f_f(FUNC_NAME, ARRAY, ROUNDING_MODE)		\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
    for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
      RUN_TEST_f_f ((ARRAY)[i].arg_str, FUNC_NAME, (ARRAY)[i].arg,	\
		    (ARRAY)[i].RM_##ROUNDING_MODE.expected,		\
		    (ARRAY)[i].RM_##ROUNDING_MODE.exceptions);		\
  ROUND_RESTORE_ ## ROUNDING_MODE
#define RUN_TEST_fp_f(ARG_STR, FUNC_NAME, ARG, EXPECTED,		\
		     EXCEPTIONS)					\
  do									\
    if (enable_test (EXCEPTIONS))					\
      {									\
	COMMON_TEST_SETUP (ARG_STR);					\
	check_float (test_name,	FUNC_TEST (FUNC_NAME) (&(ARG)),		\
		     EXPECTED, EXCEPTIONS);				\
	COMMON_TEST_CLEANUP;						\
      }									\
  while (0)
#define RUN_TEST_LOOP_fp_f(FUNC_NAME, ARRAY, ROUNDING_MODE)		\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
    for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
      RUN_TEST_fp_f ((ARRAY)[i].arg_str, FUNC_NAME, (ARRAY)[i].arg,	\
		    (ARRAY)[i].RM_##ROUNDING_MODE.expected,		\
		    (ARRAY)[i].RM_##ROUNDING_MODE.exceptions);		\
  ROUND_RESTORE_ ## ROUNDING_MODE
#define RUN_TEST_2_f(ARG_STR, FUNC_NAME, ARG1, ARG2, EXPECTED,	\
		     EXCEPTIONS)				\
  do								\
    if (enable_test (EXCEPTIONS))				\
      {								\
	COMMON_TEST_SETUP (ARG_STR);				\
	check_float (test_name, FUNC_TEST (FUNC_NAME) (ARG1, ARG2),	\
		     EXPECTED, EXCEPTIONS);			\
	COMMON_TEST_CLEANUP;					\
      }								\
  while (0)
#define RUN_TEST_LOOP_2_f(FUNC_NAME, ARRAY, ROUNDING_MODE)		\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
    for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
      RUN_TEST_2_f ((ARRAY)[i].arg_str, FUNC_NAME, (ARRAY)[i].arg1,	\
		    (ARRAY)[i].arg2,					\
		    (ARRAY)[i].RM_##ROUNDING_MODE.expected,		\
		    (ARRAY)[i].RM_##ROUNDING_MODE.exceptions);		\
  ROUND_RESTORE_ ## ROUNDING_MODE
#define RUN_TEST_ff_f RUN_TEST_2_f
#define RUN_TEST_LOOP_ff_f RUN_TEST_LOOP_2_f
#define RUN_TEST_LOOP_fj_f RUN_TEST_LOOP_2_f
#define RUN_TEST_LOOP_aa_f RUN_TEST_LOOP_2_f
#define RUN_TEST_fi_f RUN_TEST_2_f
#define RUN_TEST_LOOP_fi_f RUN_TEST_LOOP_2_f
#define RUN_TEST_fl_f RUN_TEST_2_f
#define RUN_TEST_LOOP_fl_f RUN_TEST_LOOP_2_f
#define RUN_TEST_if_f RUN_TEST_2_f
#define RUN_TEST_LOOP_if_f RUN_TEST_LOOP_2_f
#define RUN_TEST_fff_f(ARG_STR, FUNC_NAME, ARG1, ARG2, ARG3,		\
		       EXPECTED, EXCEPTIONS)				\
  do									\
    if (enable_test (EXCEPTIONS))					\
      {									\
	COMMON_TEST_SETUP (ARG_STR);					\
	check_float (test_name, FUNC_TEST (FUNC_NAME) (ARG1, ARG2, ARG3),	\
		     EXPECTED, EXCEPTIONS);				\
	COMMON_TEST_CLEANUP;						\
      }									\
  while (0)
#define RUN_TEST_LOOP_fff_f(FUNC_NAME, ARRAY, ROUNDING_MODE)		\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
    for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
      RUN_TEST_fff_f ((ARRAY)[i].arg_str, FUNC_NAME, (ARRAY)[i].arg1,	\
		      (ARRAY)[i].arg2, (ARRAY)[i].arg3,			\
		      (ARRAY)[i].RM_##ROUNDING_MODE.expected,		\
		      (ARRAY)[i].RM_##ROUNDING_MODE.exceptions);	\
  ROUND_RESTORE_ ## ROUNDING_MODE
#define RUN_TEST_fiu_M(ARG_STR, FUNC_NAME, ARG1, ARG2, ARG3,		\
		       EXPECTED, EXCEPTIONS)				\
  do									\
    if (enable_test (EXCEPTIONS))					\
      {									\
	COMMON_TEST_SETUP (ARG_STR);					\
	check_intmax_t (test_name,					\
			FUNC_TEST (FUNC_NAME) (ARG1, ARG2, ARG3),	\
			EXPECTED, EXCEPTIONS);				\
	COMMON_TEST_CLEANUP;						\
      }									\
  while (0)
#define RUN_TEST_LOOP_fiu_M(FUNC_NAME, ARRAY, ROUNDING_MODE)		\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
    for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
      RUN_TEST_fiu_M ((ARRAY)[i].arg_str, FUNC_NAME, (ARRAY)[i].arg1,	\
		      (ARRAY)[i].arg2, (ARRAY)[i].arg3,			\
		      (ARRAY)[i].RM_##ROUNDING_MODE.expected,		\
		      (ARRAY)[i].RM_##ROUNDING_MODE.exceptions);	\
  ROUND_RESTORE_ ## ROUNDING_MODE
#define RUN_TEST_fiu_U(ARG_STR, FUNC_NAME, ARG1, ARG2, ARG3,		\
		       EXPECTED, EXCEPTIONS)				\
  do									\
    if (enable_test (EXCEPTIONS))					\
      {									\
	COMMON_TEST_SETUP (ARG_STR);					\
	check_uintmax_t (test_name,					\
			 FUNC_TEST (FUNC_NAME) (ARG1, ARG2, ARG3),	\
			 EXPECTED, EXCEPTIONS);				\
	COMMON_TEST_CLEANUP;						\
      }									\
  while (0)
#define RUN_TEST_LOOP_fiu_U(FUNC_NAME, ARRAY, ROUNDING_MODE)		\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
    for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
      RUN_TEST_fiu_U ((ARRAY)[i].arg_str, FUNC_NAME, (ARRAY)[i].arg1,	\
		      (ARRAY)[i].arg2, (ARRAY)[i].arg3,			\
		      (ARRAY)[i].RM_##ROUNDING_MODE.expected,		\
		      (ARRAY)[i].RM_##ROUNDING_MODE.exceptions);	\
  ROUND_RESTORE_ ## ROUNDING_MODE
#define RUN_TEST_c_f(ARG_STR, FUNC_NAME, ARG1, ARG2, EXPECTED,		\
		     EXCEPTIONS)					\
  do									\
    if (enable_test (EXCEPTIONS))					\
      {									\
	COMMON_TEST_SETUP (ARG_STR);					\
	check_float (test_name,						\
		     FUNC_TEST (FUNC_NAME) (BUILD_COMPLEX (ARG1, ARG2)),\
		     EXPECTED, EXCEPTIONS);				\
	COMMON_TEST_CLEANUP;						\
      }									\
  while (0)
#define RUN_TEST_LOOP_c_f(FUNC_NAME, ARRAY, ROUNDING_MODE)		\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
    for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
      RUN_TEST_c_f ((ARRAY)[i].arg_str, FUNC_NAME, (ARRAY)[i].argr,	\
		    (ARRAY)[i].argc,					\
		    (ARRAY)[i].RM_##ROUNDING_MODE.expected,		\
		    (ARRAY)[i].RM_##ROUNDING_MODE.exceptions);		\
  ROUND_RESTORE_ ## ROUNDING_MODE
#define RUN_TEST_f_f1(ARG_STR, FUNC_NAME, ARG, EXPECTED,		\
		      EXCEPTIONS, EXTRA_VAR, EXTRA_TEST,		\
		      EXTRA_EXPECTED)					\
  do									\
    if (enable_test (EXCEPTIONS))					\
      {									\
	COMMON_TEST_SETUP (ARG_STR);					\
	(EXTRA_VAR) = (EXTRA_EXPECTED) == 0 ? 1 : 0;			\
	check_float (test_name, FUNC_TEST (FUNC_NAME) (ARG), EXPECTED,	\
		     EXCEPTIONS);					\
	EXTRA_OUTPUT_TEST_SETUP (ARG_STR, 1);				\
	if (EXTRA_TEST)							\
	  check_int (extra1_name, EXTRA_VAR, EXTRA_EXPECTED, 0);	\
	EXTRA_OUTPUT_TEST_CLEANUP (1);					\
	COMMON_TEST_CLEANUP;						\
      }									\
  while (0)
#define RUN_TEST_LOOP_f_f1(FUNC_NAME, ARRAY, ROUNDING_MODE, EXTRA_VAR)	\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
    for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
      RUN_TEST_f_f1 ((ARRAY)[i].arg_str, FUNC_NAME, (ARRAY)[i].arg,	\
		     (ARRAY)[i].RM_##ROUNDING_MODE.expected,		\
		     (ARRAY)[i].RM_##ROUNDING_MODE.exceptions,		\
		     EXTRA_VAR,						\
		     (ARRAY)[i].RM_##ROUNDING_MODE.extra_test,		\
		     (ARRAY)[i].RM_##ROUNDING_MODE.extra_expected);	\
  ROUND_RESTORE_ ## ROUNDING_MODE
#define RUN_TEST_fF_f1(ARG_STR, FUNC_NAME, ARG, EXPECTED,		\
		       EXCEPTIONS, EXTRA_VAR, EXTRA_TEST,		\
		       EXTRA_EXPECTED)					\
  do									\
    if (enable_test (EXCEPTIONS))					\
      {									\
	COMMON_TEST_SETUP (ARG_STR);					\
	(EXTRA_VAR) = (EXTRA_EXPECTED) == 0 ? 1 : 0;			\
	check_float (test_name, FUNC_TEST (FUNC_NAME) (ARG, &(EXTRA_VAR)),	\
		     EXPECTED, EXCEPTIONS);				\
	EXTRA_OUTPUT_TEST_SETUP (ARG_STR, 1);				\
	if (EXTRA_TEST)							\
	  check_float (extra1_name, EXTRA_VAR, EXTRA_EXPECTED, 0);	\
	EXTRA_OUTPUT_TEST_CLEANUP (1);					\
	COMMON_TEST_CLEANUP;						\
      }									\
  while (0)
#define RUN_TEST_LOOP_fF_f1(FUNC_NAME, ARRAY, ROUNDING_MODE, EXTRA_VAR)	\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
    for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
      RUN_TEST_fF_f1 ((ARRAY)[i].arg_str, FUNC_NAME, (ARRAY)[i].arg,	\
		      (ARRAY)[i].RM_##ROUNDING_MODE.expected,		\
		      (ARRAY)[i].RM_##ROUNDING_MODE.exceptions,		\
		      EXTRA_VAR,					\
		      (ARRAY)[i].RM_##ROUNDING_MODE.extra_test,		\
		      (ARRAY)[i].RM_##ROUNDING_MODE.extra_expected);	\
  ROUND_RESTORE_ ## ROUNDING_MODE
#define RUN_TEST_fI_f1(ARG_STR, FUNC_NAME, ARG, EXPECTED,		\
		       EXCEPTIONS, EXTRA_VAR, EXTRA_TEST,		\
		       EXTRA_EXPECTED)					\
  do									\
    if (enable_test (EXCEPTIONS))					\
      {									\
	COMMON_TEST_SETUP (ARG_STR);					\
	(EXTRA_VAR) = (EXTRA_EXPECTED) == 0 ? 1 : 0;			\
	check_float (test_name, FUNC_TEST (FUNC_NAME) (ARG, &(EXTRA_VAR)),	\
		     EXPECTED, EXCEPTIONS);				\
	EXTRA_OUTPUT_TEST_SETUP (ARG_STR, 1);				\
	if (EXTRA_TEST)							\
	  check_int (extra1_name, EXTRA_VAR, EXTRA_EXPECTED, 0);	\
	EXTRA_OUTPUT_TEST_CLEANUP (1);					\
	COMMON_TEST_CLEANUP;						\
      }									\
  while (0)
#define RUN_TEST_LOOP_fI_f1(FUNC_NAME, ARRAY, ROUNDING_MODE, EXTRA_VAR)	\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
    for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
      RUN_TEST_fI_f1 ((ARRAY)[i].arg_str, FUNC_NAME, (ARRAY)[i].arg,	\
		      (ARRAY)[i].RM_##ROUNDING_MODE.expected,		\
		      (ARRAY)[i].RM_##ROUNDING_MODE.exceptions,		\
		      EXTRA_VAR,					\
		      (ARRAY)[i].RM_##ROUNDING_MODE.extra_test,		\
		      (ARRAY)[i].RM_##ROUNDING_MODE.extra_expected);	\
  ROUND_RESTORE_ ## ROUNDING_MODE
#define RUN_TEST_ffI_f1_mod8(ARG_STR, FUNC_NAME, ARG1, ARG2, EXPECTED,	\
			     EXCEPTIONS, EXTRA_VAR, EXTRA_TEST,		\
			     EXTRA_EXPECTED)				\
  do									\
    if (enable_test (EXCEPTIONS))					\
      {									\
	COMMON_TEST_SETUP (ARG_STR);					\
	(EXTRA_VAR) = (EXTRA_EXPECTED) == 0 ? 1 : 0;			\
	check_float (test_name,						\
		     FUNC_TEST (FUNC_NAME) (ARG1, ARG2, &(EXTRA_VAR)),	\
		     EXPECTED, EXCEPTIONS);				\
	EXTRA_OUTPUT_TEST_SETUP (ARG_STR, 1);				\
	if (EXTRA_TEST)							\
	  check_int (extra1_name, (EXTRA_VAR) % 8, EXTRA_EXPECTED, 0);	\
	EXTRA_OUTPUT_TEST_CLEANUP (1);					\
	COMMON_TEST_CLEANUP;						\
      }									\
  while (0)
#define RUN_TEST_LOOP_ffI_f1_mod8(FUNC_NAME, ARRAY, ROUNDING_MODE,	\
				  EXTRA_VAR)				\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
    for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
      RUN_TEST_ffI_f1_mod8 ((ARRAY)[i].arg_str, FUNC_NAME,		\
			    (ARRAY)[i].arg1, (ARRAY)[i].arg2,		\
			    (ARRAY)[i].RM_##ROUNDING_MODE.expected,	\
			    (ARRAY)[i].RM_##ROUNDING_MODE.exceptions,	\
			    EXTRA_VAR,					\
			    (ARRAY)[i].RM_##ROUNDING_MODE.extra_test,	\
			    (ARRAY)[i].RM_##ROUNDING_MODE.extra_expected); \
  ROUND_RESTORE_ ## ROUNDING_MODE
#define RUN_TEST_Ff_b1(ARG_STR, FUNC_NAME, ARG, EXPECTED,		\
		       EXCEPTIONS, EXTRA_VAR, EXTRA_TEST,		\
		       EXTRA_EXPECTED)					\
  do									\
    if (enable_test (EXCEPTIONS))					\
      {									\
	COMMON_TEST_SETUP (ARG_STR);					\
	(EXTRA_VAR) = (EXTRA_EXPECTED) == 0 ? 1 : 0;			\
	/* Clear any exceptions from comparison involving sNaN		\
	   EXTRA_EXPECTED.  */						\
	feclearexcept (FE_ALL_EXCEPT);					\
	check_bool (test_name, FUNC_TEST (FUNC_NAME) (&(EXTRA_VAR),	\
						      (ARG)),		\
		    EXPECTED, EXCEPTIONS);				\
	EXTRA_OUTPUT_TEST_SETUP (ARG_STR, 1);				\
	if (EXTRA_TEST)							\
	  check_float (extra1_name, EXTRA_VAR, EXTRA_EXPECTED,		\
		       (EXCEPTIONS) & TEST_NAN_PAYLOAD);		\
	EXTRA_OUTPUT_TEST_CLEANUP (1);					\
	COMMON_TEST_CLEANUP;						\
      }									\
  while (0)
#define RUN_TEST_LOOP_Ff_b1(FUNC_NAME, ARRAY, ROUNDING_MODE,		\
			    EXTRA_VAR)					\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
    for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
      RUN_TEST_Ff_b1 ((ARRAY)[i].arg_str, FUNC_NAME, (ARRAY)[i].arg,	\
		      (ARRAY)[i].RM_##ROUNDING_MODE.expected,		\
		      (ARRAY)[i].RM_##ROUNDING_MODE.exceptions,		\
		      EXTRA_VAR,					\
		      (ARRAY)[i].RM_##ROUNDING_MODE.extra_test,		\
		      (ARRAY)[i].RM_##ROUNDING_MODE.extra_expected);	\
  ROUND_RESTORE_ ## ROUNDING_MODE
#define RUN_TEST_Ffp_b1(ARG_STR, FUNC_NAME, ARG, EXPECTED,		\
			EXCEPTIONS, EXTRA_VAR, EXTRA_TEST,		\
			EXTRA_EXPECTED)					\
  do									\
    if (enable_test (EXCEPTIONS))					\
      {									\
	COMMON_TEST_SETUP (ARG_STR);					\
	(EXTRA_VAR) = (EXTRA_EXPECTED) == 0 ? 1 : 0;			\
	check_bool (test_name, FUNC_TEST (FUNC_NAME) (&(EXTRA_VAR),	\
						      &(ARG)),		\
		    EXPECTED, EXCEPTIONS);				\
	EXTRA_OUTPUT_TEST_SETUP (ARG_STR, 1);				\
	if (EXTRA_TEST)							\
	  check_float (extra1_name, EXTRA_VAR, EXTRA_EXPECTED,		\
		       (EXCEPTIONS) & TEST_NAN_PAYLOAD);		\
	EXTRA_OUTPUT_TEST_CLEANUP (1);					\
	COMMON_TEST_CLEANUP;						\
      }									\
  while (0)
#define RUN_TEST_LOOP_Ffp_b1(FUNC_NAME, ARRAY, ROUNDING_MODE,		\
			     EXTRA_VAR)					\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
    for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
      RUN_TEST_Ffp_b1 ((ARRAY)[i].arg_str, FUNC_NAME, (ARRAY)[i].arg,	\
		       (ARRAY)[i].RM_##ROUNDING_MODE.expected,		\
		       (ARRAY)[i].RM_##ROUNDING_MODE.exceptions,	\
		       EXTRA_VAR,					\
		       (ARRAY)[i].RM_##ROUNDING_MODE.extra_test,	\
		       (ARRAY)[i].RM_##ROUNDING_MODE.extra_expected);	\
  ROUND_RESTORE_ ## ROUNDING_MODE
#define RUN_TEST_c_c(ARG_STR, FUNC_NAME, ARGR, ARGC, EXPR, EXPC,	\
		     EXCEPTIONS)					\
  do									\
    if (enable_test (EXCEPTIONS))					\
      {									\
	COMMON_TEST_SETUP (ARG_STR);					\
	check_complex (test_name,					\
		       FUNC_TEST (FUNC_NAME) (BUILD_COMPLEX (ARGR, ARGC)),	\
		       BUILD_COMPLEX (EXPR, EXPC), EXCEPTIONS);		\
	COMMON_TEST_CLEANUP;						\
      }									\
  while (0)
#define RUN_TEST_LOOP_c_c(FUNC_NAME, ARRAY, ROUNDING_MODE)		\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
    for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
      RUN_TEST_c_c ((ARRAY)[i].arg_str, FUNC_NAME, (ARRAY)[i].argr,	\
		    (ARRAY)[i].argc,					\
		    (ARRAY)[i].RM_##ROUNDING_MODE.expr,			\
		    (ARRAY)[i].RM_##ROUNDING_MODE.expc,			\
		    (ARRAY)[i].RM_##ROUNDING_MODE.exceptions);		\
  ROUND_RESTORE_ ## ROUNDING_MODE
#define RUN_TEST_cc_c(ARG_STR, FUNC_NAME, ARG1R, ARG1C, ARG2R, ARG2C,	\
		      EXPR, EXPC, EXCEPTIONS)				\
  do									\
    if (enable_test (EXCEPTIONS))					\
      {									\
	COMMON_TEST_SETUP (ARG_STR);					\
	check_complex (test_name,					\
		       FUNC_TEST (FUNC_NAME) (BUILD_COMPLEX (ARG1R, ARG1C),	\
					      BUILD_COMPLEX (ARG2R, ARG2C)),	\
		       BUILD_COMPLEX (EXPR, EXPC), EXCEPTIONS);		\
	COMMON_TEST_CLEANUP;						\
      }									\
  while (0)
#define RUN_TEST_LOOP_cc_c(FUNC_NAME, ARRAY, ROUNDING_MODE)		\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
    for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
      RUN_TEST_cc_c ((ARRAY)[i].arg_str, FUNC_NAME, (ARRAY)[i].arg1r,	\
		     (ARRAY)[i].arg1c, (ARRAY)[i].arg2r,		\
		     (ARRAY)[i].arg2c,					\
		     (ARRAY)[i].RM_##ROUNDING_MODE.expr,		\
		     (ARRAY)[i].RM_##ROUNDING_MODE.expc,		\
		     (ARRAY)[i].RM_##ROUNDING_MODE.exceptions);		\
  ROUND_RESTORE_ ## ROUNDING_MODE
#define RUN_TEST_f_i(ARG_STR, FUNC_NAME, ARG, EXPECTED, EXCEPTIONS)	\
  do									\
    if (enable_test (EXCEPTIONS))					\
      {									\
	COMMON_TEST_SETUP (ARG_STR);					\
	check_int (test_name, FUNC_TEST (FUNC_NAME) (ARG), EXPECTED,	\
		   EXCEPTIONS);						\
	COMMON_TEST_CLEANUP;						\
      }									\
  while (0)
#define RUN_TEST_LOOP_f_i(FUNC_NAME, ARRAY, ROUNDING_MODE)		\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
    for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
      RUN_TEST_f_i ((ARRAY)[i].arg_str, FUNC_NAME, (ARRAY)[i].arg,	\
		    (ARRAY)[i].RM_##ROUNDING_MODE.expected,		\
		    (ARRAY)[i].RM_##ROUNDING_MODE.exceptions);		\
  ROUND_RESTORE_ ## ROUNDING_MODE
#define RUN_TEST_f_i_tg(ARG_STR, FUNC_NAME, ARG, EXPECTED,		\
			EXCEPTIONS)					\
  do									\
    if (enable_test (EXCEPTIONS))					\
      {									\
	COMMON_TEST_SETUP (ARG_STR);					\
	check_int (test_name, FUNC_NAME (ARG), EXPECTED, EXCEPTIONS);	\
	COMMON_TEST_CLEANUP;						\
      }									\
  while (0)
#define RUN_TEST_LOOP_f_i_tg(FUNC_NAME, ARRAY, ROUNDING_MODE)		\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
    for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
      RUN_TEST_f_i_tg ((ARRAY)[i].arg_str, FUNC_NAME, (ARRAY)[i].arg,	\
		       (ARRAY)[i].RM_##ROUNDING_MODE.expected,		\
		       (ARRAY)[i].RM_##ROUNDING_MODE.exceptions);	\
  ROUND_RESTORE_ ## ROUNDING_MODE
#define RUN_TEST_LOOP_j_b_tg_u(FUNC_NAME, ARRAY, ROUNDING_MODE)		\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
  for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
  RUN_TEST_f_b_tg ((ARRAY)[i].arg_str, FUNC_NAME,			\
		   (FLOAT)(ARRAY)[i].arg.value,				\
		   (ARRAY)[i].RM_##ROUNDING_MODE.expected,		\
		   (ARRAY)[i].RM_##ROUNDING_MODE.exceptions);		\
  ROUND_RESTORE_ ## ROUNDING_MODE
#define RUN_TEST_LOOP_j_i_tg_u(FUNC_NAME, ARRAY, ROUNDING_MODE)		\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
  for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
  RUN_TEST_f_i_tg ((ARRAY)[i].arg_str, FUNC_NAME,			\
		   (FLOAT)(ARRAY)[i].arg.value,				\
		   (ARRAY)[i].RM_##ROUNDING_MODE.expected,		\
		   (ARRAY)[i].RM_##ROUNDING_MODE.exceptions);		\
  ROUND_RESTORE_ ## ROUNDING_MODE
#define RUN_TEST_ff_b(ARG_STR, FUNC_NAME, ARG1, ARG2, EXPECTED,		\
		      EXCEPTIONS)					\
  do									\
    if (enable_test (EXCEPTIONS))					\
      {									\
	COMMON_TEST_SETUP (ARG_STR);					\
	check_bool (test_name, FUNC_TEST (FUNC_NAME) (ARG1, ARG2),	\
		    EXPECTED, EXCEPTIONS);				\
	COMMON_TEST_CLEANUP;						\
      }									\
  while (0)
#define RUN_TEST_LOOP_ff_b(FUNC_NAME, ARRAY, ROUNDING_MODE)		\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
    for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
      RUN_TEST_ff_b ((ARRAY)[i].arg_str, FUNC_NAME,			\
		     (ARRAY)[i].arg1, (ARRAY)[i].arg2,			\
		     (ARRAY)[i].RM_##ROUNDING_MODE.expected,		\
		     (ARRAY)[i].RM_##ROUNDING_MODE.exceptions);		\
  ROUND_RESTORE_ ## ROUNDING_MODE
#define RUN_TEST_fpfp_b(ARG_STR, FUNC_NAME, ARG1, ARG2, EXPECTED,	\
			EXCEPTIONS)					\
  do									\
    if (enable_test (EXCEPTIONS))					\
      {									\
	COMMON_TEST_SETUP (ARG_STR);					\
	check_bool (test_name,						\
		    FUNC_TEST (FUNC_NAME) (&(ARG1), &(ARG2)),		\
		    EXPECTED, EXCEPTIONS);				\
	COMMON_TEST_CLEANUP;						\
      }									\
  while (0)
#define RUN_TEST_LOOP_fpfp_b(FUNC_NAME, ARRAY, ROUNDING_MODE)		\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
    for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
      RUN_TEST_fpfp_b ((ARRAY)[i].arg_str, FUNC_NAME,			\
		       (ARRAY)[i].arg1, (ARRAY)[i].arg2,		\
		       (ARRAY)[i].RM_##ROUNDING_MODE.expected,		\
		       (ARRAY)[i].RM_##ROUNDING_MODE.exceptions);	\
  ROUND_RESTORE_ ## ROUNDING_MODE
#define RUN_TEST_ff_i_tg(ARG_STR, FUNC_NAME, ARG1, ARG2, EXPECTED,	\
			 EXCEPTIONS)					\
  do									\
    if (enable_test (EXCEPTIONS))					\
      {									\
	COMMON_TEST_SETUP (ARG_STR);					\
	check_int (test_name, FUNC_NAME (ARG1, ARG2), EXPECTED,		\
		   EXCEPTIONS);						\
	COMMON_TEST_CLEANUP;						\
      }									\
  while (0)
#define RUN_TEST_LOOP_ff_i_tg(FUNC_NAME, ARRAY, ROUNDING_MODE)		\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
    for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
      RUN_TEST_ff_i_tg ((ARRAY)[i].arg_str, FUNC_NAME,			\
			(ARRAY)[i].arg1, (ARRAY)[i].arg2,		\
			(ARRAY)[i].RM_##ROUNDING_MODE.expected,		\
			(ARRAY)[i].RM_##ROUNDING_MODE.exceptions);	\
  ROUND_RESTORE_ ## ROUNDING_MODE
#define RUN_TEST_f_b(ARG_STR, FUNC_NAME, ARG, EXPECTED, EXCEPTIONS)	\
  do									\
    if (enable_test (EXCEPTIONS))					\
      {									\
	COMMON_TEST_SETUP (ARG_STR);					\
	check_bool (test_name, FUNC_TEST (FUNC_NAME) (ARG), EXPECTED,	\
		    EXCEPTIONS);					\
	COMMON_TEST_CLEANUP;						\
      }									\
  while (0)
#define RUN_TEST_LOOP_f_b(FUNC_NAME, ARRAY, ROUNDING_MODE)		\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
    for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
      RUN_TEST_f_b ((ARRAY)[i].arg_str, FUNC_NAME, (ARRAY)[i].arg,	\
		    (ARRAY)[i].RM_##ROUNDING_MODE.expected,		\
		    (ARRAY)[i].RM_##ROUNDING_MODE.exceptions);		\
  ROUND_RESTORE_ ## ROUNDING_MODE
#define RUN_TEST_f_b_tg(ARG_STR, FUNC_NAME, ARG, EXPECTED,		\
			EXCEPTIONS)					\
  do									\
    if (enable_test (EXCEPTIONS))					\
      {									\
	COMMON_TEST_SETUP (ARG_STR);					\
	check_bool (test_name, FUNC_NAME (ARG), EXPECTED, EXCEPTIONS);	\
	COMMON_TEST_CLEANUP;						\
      }									\
  while (0)
#define RUN_TEST_LOOP_f_b_tg(FUNC_NAME, ARRAY, ROUNDING_MODE)		\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
    for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
      RUN_TEST_f_b_tg ((ARRAY)[i].arg_str, FUNC_NAME, (ARRAY)[i].arg,	\
		       (ARRAY)[i].RM_##ROUNDING_MODE.expected,		\
		       (ARRAY)[i].RM_##ROUNDING_MODE.exceptions);	\
  ROUND_RESTORE_ ## ROUNDING_MODE
#define RUN_TEST_f_l(ARG_STR, FUNC_NAME, ARG, EXPECTED, EXCEPTIONS)	\
  do									\
    if (enable_test (EXCEPTIONS))					\
      {									\
	COMMON_TEST_SETUP (ARG_STR);					\
	check_long (test_name, FUNC_TEST (FUNC_NAME) (ARG), EXPECTED,	\
		    EXCEPTIONS);					\
	COMMON_TEST_CLEANUP;						\
      }									\
  while (0)
#define RUN_TEST_LOOP_f_l(FUNC_NAME, ARRAY, ROUNDING_MODE)		\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
    for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
      RUN_TEST_f_l ((ARRAY)[i].arg_str, FUNC_NAME, (ARRAY)[i].arg,	\
		    (ARRAY)[i].RM_##ROUNDING_MODE.expected,		\
		    (ARRAY)[i].RM_##ROUNDING_MODE.exceptions);		\
  ROUND_RESTORE_ ## ROUNDING_MODE
#define RUN_TEST_f_L(ARG_STR, FUNC_NAME, ARG, EXPECTED, EXCEPTIONS)	\
  do									\
    if (enable_test (EXCEPTIONS))					\
      {									\
	COMMON_TEST_SETUP (ARG_STR);					\
	check_longlong (test_name, FUNC_TEST (FUNC_NAME) (ARG),		\
			EXPECTED, EXCEPTIONS);				\
	COMMON_TEST_CLEANUP;						\
      }									\
  while (0)
#define RUN_TEST_LOOP_f_L(FUNC_NAME, ARRAY, ROUNDING_MODE)		\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
    for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
      RUN_TEST_f_L ((ARRAY)[i].arg_str, FUNC_NAME, (ARRAY)[i].arg,	\
		    (ARRAY)[i].RM_##ROUNDING_MODE.expected,		\
		    (ARRAY)[i].RM_##ROUNDING_MODE.exceptions);		\
  ROUND_RESTORE_ ## ROUNDING_MODE
#define RUN_TEST_fFF_11(ARG_STR, FUNC_NAME, ARG, EXCEPTIONS,		\
			EXTRA1_VAR, EXTRA1_TEST,			\
			EXTRA1_EXPECTED, EXTRA2_VAR,			\
			EXTRA2_TEST, EXTRA2_EXPECTED)			\
  do									\
    if (enable_test (EXCEPTIONS))					\
      {									\
	COMMON_TEST_SETUP (ARG_STR);					\
	FUNC_TEST (FUNC_NAME) (ARG, &(EXTRA1_VAR), &(EXTRA2_VAR));	\
	EXTRA_OUTPUT_TEST_SETUP (ARG_STR, 1);				\
	if (EXTRA1_TEST)						\
	  check_float (extra1_name, EXTRA1_VAR, EXTRA1_EXPECTED,	\
		       EXCEPTIONS);					\
	EXTRA_OUTPUT_TEST_CLEANUP (1);					\
	EXTRA_OUTPUT_TEST_SETUP (ARG_STR, 2);				\
	if (EXTRA2_TEST)						\
	  check_float (extra2_name, EXTRA2_VAR, EXTRA2_EXPECTED, 0);	\
	EXTRA_OUTPUT_TEST_CLEANUP (2);					\
	COMMON_TEST_CLEANUP;						\
      }									\
  while (0)
#define RUN_TEST_LOOP_fFF_11(FUNC_NAME, ARRAY, ROUNDING_MODE,		\
			     EXTRA1_VAR, EXTRA2_VAR)			\
  IF_ROUND_INIT_ ## ROUNDING_MODE					\
    for (size_t i = 0; i < sizeof (ARRAY) / sizeof (ARRAY)[0]; i++)	\
      RUN_TEST_fFF_11 ((ARRAY)[i].arg_str, FUNC_NAME, (ARRAY)[i].arg,	\
		       (ARRAY)[i].RM_##ROUNDING_MODE.exceptions,	\
		       EXTRA1_VAR,					\
		       (ARRAY)[i].RM_##ROUNDING_MODE.extra1_test,	\
		       (ARRAY)[i].RM_##ROUNDING_MODE.extra1_expected,	\
		       EXTRA2_VAR,					\
		       (ARRAY)[i].RM_##ROUNDING_MODE.extra2_test,	\
		       (ARRAY)[i].RM_##ROUNDING_MODE.extra2_expected);	\
  ROUND_RESTORE_ ## ROUNDING_MODE

#if TEST_MATHVEC
# define TEST_SUFF VEC_SUFF
# define TEST_SUFF_STR
#elif TEST_NARROW
# define TEST_SUFF
# define TEST_SUFF_STR "_" ARG_TYPE_STR
#else
# define TEST_SUFF
# define TEST_SUFF_STR
#endif

#define STR_CONCAT(a, b, c) __STRING (a##b##c)
#define STR_CON3(a, b, c) STR_CONCAT (a, b, c)

#if TEST_NARROW
# define TEST_COND_any_ibm128 (TEST_COND_ibm128 || TEST_COND_arg_ibm128)
#else
# define TEST_COND_any_ibm128 TEST_COND_ibm128
#endif

/* Start and end the tests for a given function.  */
#define START(FUN, SUFF, EXACT)					\
  CHECK_ARCH_EXT;						\
  const char *this_func						\
    = STR_CON3 (FUN, SUFF, TEST_SUFF) TEST_SUFF_STR;		\
  init_max_error (this_func, EXACT, TEST_COND_any_ibm128)
#define END					\
  print_max_error (this_func)
#define END_COMPLEX				\
  print_complex_max_error (this_func)

/* Run tests for a given function in all rounding modes.  */
#define ALL_RM_TEST(FUNC, EXACT, ARRAY, LOOP_MACRO, END_MACRO, ...)	\
  do									\
    {									\
      do								\
	{								\
	  START (FUNC,, EXACT);						\
	  LOOP_MACRO (FUNC, ARRAY, , ## __VA_ARGS__);			\
	  END_MACRO;							\
	}								\
      while (0);							\
      do								\
	{								\
	  START (FUNC, _downward, EXACT);				\
	  LOOP_MACRO (FUNC, ARRAY, FE_DOWNWARD, ## __VA_ARGS__);	\
	  END_MACRO;							\
	}								\
      while (0);							\
      do								\
	{								\
	  START (FUNC, _towardzero, EXACT);				\
	  LOOP_MACRO (FUNC, ARRAY, FE_TOWARDZERO, ## __VA_ARGS__);	\
	  END_MACRO;							\
	}								\
      while (0);							\
      do								\
	{								\
	  START (FUNC, _upward, EXACT);				\
	  LOOP_MACRO (FUNC, ARRAY, FE_UPWARD, ## __VA_ARGS__);		\
	  END_MACRO;							\
	}								\
      while (0);							\
    }									\
  while (0);

/* Short description of program.  */
const char doc[] = "Math test suite: " TEST_MSG ;

static void do_test (void);

int
main (int argc, char **argv)
{
  libm_test_init (argc, argv);
  INIT_ARCH_EXT;
  do_test ();
  return libm_test_finish ();
}
