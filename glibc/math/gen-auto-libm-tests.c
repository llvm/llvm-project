/* Generate expected output for libm tests with MPFR and MPC.
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

/* Compile this program as:

   gcc -std=gnu11 -O2 -Wall -Wextra gen-auto-libm-tests.c -lmpc -lmpfr -lgmp \
     -o gen-auto-libm-tests

   (use of current MPC and MPFR versions recommended) and run it as:

   gen-auto-libm-tests auto-libm-test-in <func> auto-libm-test-out-<func>

   to generate results for normal libm functions, or

   gen-auto-libm-tests --narrow auto-libm-test-in <func> \
     auto-libm-test-out-narrow-<func>

   to generate results for a function rounding results to a narrower
   type (in the case of fma and sqrt, both output files are generated
   from the same test inputs).

   The input file auto-libm-test-in contains three kinds of lines:

   Lines beginning with "#" are comments, and are ignored, as are
   empty lines.

   Other lines are test lines, of the form "function input1 input2
   ... [flag1 flag2 ...]".  Inputs are either finite real numbers or
   integers, depending on the function under test.  Real numbers may
   be in any form acceptable to mpfr_strtofr (base 0); integers in any
   form acceptable to mpz_set_str (base 0).  In addition, real numbers
   may be certain special strings such as "pi", as listed in the
   special_real_inputs array.

   Each flag is a flag name possibly followed by a series of
   ":condition".  Conditions may be any of the names of floating-point
   formats in the floating_point_formats array, "long32" and "long64"
   to indicate the number of bits in the "long" type, or other strings
   for which libm-test.inc defines a TEST_COND_<condition> macro (with
   "-"- changed to "_" in the condition name) evaluating to nonzero
   when the condition is true and zero when the condition is false.
   The meaning is that the flag applies to the test if all the listed
   conditions are true.  "flag:cond1:cond2 flag:cond3:cond4" means the
   flag applies if ((cond1 && cond2) || (cond3 && cond4)).

   A real number specified as an input is considered to represent the
   set of real numbers arising from rounding the given number in any
   direction for any supported floating-point format; any roundings
   that give infinity are ignored.  Each input on a test line has all
   the possible roundings considered independently.  Each resulting
   choice of the tuple of inputs to the function is ignored if the
   mathematical result of the function involves a NaN or an exact
   infinity, and is otherwise considered for each floating-point
   format for which all those inputs are exactly representable.  Thus
   tests may result in "overflow", "underflow" and "inexact"
   exceptions; "invalid" may arise only when the final result type is
   an integer type and it is the conversion of a mathematically
   defined finite result to integer type that results in that
   exception.

   By default, it is assumed that "overflow" and "underflow"
   exceptions should be correct, but that "inexact" exceptions should
   only be correct for functions listed as exactly determined.  For
   such functions, "underflow" exceptions should respect whether the
   machine has before-rounding or after-rounding tininess detection.
   For other functions, it is considered that if the exact result is
   somewhere between the greatest magnitude subnormal of a given sign
   (exclusive) and the least magnitude normal of that sign
   (inclusive), underflow exceptions are permitted but optional on all
   machines, and they are also permitted but optional for smaller
   subnormal exact results for functions that are not exactly
   determined.  errno setting is expected for overflow to infinity and
   underflow to zero (for real functions), and for out-of-range
   conversion of a finite result to integer type, and is considered
   permitted but optional for all other cases where overflow
   exceptions occur, and where underflow exceptions occur or are
   permitted.  In other cases (where no overflow or underflow is
   permitted), errno is expected to be left unchanged.

   The flag "ignore-zero-inf-sign" indicates the the signs of
   zero and infinite results should be ignored; "xfail" indicates the
   test is disabled as expected to produce incorrect results,
   "xfail-rounding" indicates the test is disabled only in rounding
   modes other than round-to-nearest.  Otherwise, test flags are of
   the form "spurious-<exception>" and "missing-<exception>", for any
   exception ("overflow", "underflow", "inexact", "invalid",
   "divbyzero"), "spurious-errno" and "missing-errno", to indicate
   when tests are expected to deviate from the exception and errno
   settings corresponding to the mathematical results.  "xfail",
   "xfail-rounding", "spurious-" and "missing-" flags should be
   accompanied by a comment referring to an open bug in glibc
   Bugzilla.

   The output file auto-libm-test-out-<func> contains the test lines from
   auto-libm-test-in, and, after the line for a given test, some
   number of output test lines.  An output test line is of the form "=
   function rounding-mode format input1 input2 ... : output1 output2
   ... : flags".  rounding-mode is "tonearest", "towardzero", "upward"
   or "downward".  format is a name from the floating_point_formats
   array, possibly followed by a sequence of ":flag" for flags from
   "long32" and "long64".  Inputs and outputs are specified as hex
   floats with the required suffix for the floating-point type, or
   plus_infty or minus_infty for infinite expected results, or as
   integer constant expressions (not necessarily with the right type)
   or IGNORE for integer inputs and outputs.  Flags are
   "ignore-zero-info-sign", "xfail", "<exception>",
   "<exception>-ok", "errno-<value>", "errno-<value>-ok", which may be
   unconditional or conditional.  "<exception>" indicates that a
   correct result means the given exception should be raised.
   "errno-<value>" indicates that a correct result means errno should
   be set to the given value.  "-ok" means not to test for the given
   exception or errno value (whether because it was marked as possibly
   missing or spurious, or because the calculation of correct results
   indicated it was optional).  Conditions "before-rounding" and
   "after-rounding" indicate tests where expectations for underflow
   exceptions depend on how the architecture detects tininess.

   For functions rounding their results to a narrower type, the format
   given on an output test line is the result format followed by
   information about the requirements on the argument format to be
   able to represent the argument values, in the form
   "format:arg_fmt(MAX_EXP,NUM_ONES,MIN_EXP,MAX_PREC)".  Instead of
   separate lines for separate argument formats, an output test line
   relates to all argument formats that can represent the values.
   MAX_EXP is the maximum exponent of a nonzero bit in any argument,
   or 0 if all arguments are zero; NUM_ONES is the maximum number of
   leading bits with value 1 in an argument with exponent MAX_EXP, or
   0 if all arguments are zero; MIN_EXP is the minimum exponent of a
   nonzero bit in any argument, or 0 if all arguments are zero;
   MAX_PREC is the maximum precision required to represent all
   arguments, or 0 if all arguments are zero.  */

#define _GNU_SOURCE

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <error.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gmp.h>
#include <mpfr.h>
#include <mpc.h>

#define ARRAY_SIZE(A) (sizeof (A) / sizeof ((A)[0]))

/* The supported floating-point formats.  */
typedef enum
  {
    fp_flt_32,
    fp_dbl_64,
    fp_ldbl_96_intel,
    fp_ldbl_96_m68k,
    fp_ldbl_128,
    fp_ldbl_128ibm,
    fp_num_formats,
    fp_first_format = 0
  } fp_format;

/* Structure describing a single floating-point format.  */
typedef struct
{
  /* The name of the format.  */
  const char *name;
  /* A string for the largest normal value, or NULL for IEEE formats
     where this can be determined automatically.  */
  const char *max_string;
  /* The number of mantissa bits.  */
  int mant_dig;
  /* The least N such that 2^N overflows.  */
  int max_exp;
  /* One more than the least N such that 2^N is normal.  */
  int min_exp;
  /* The largest normal value.  */
  mpfr_t max;
  /* The value 0.5ulp above the least positive normal value.  */
  mpfr_t min_plus_half;
  /* The least positive normal value, 2^(MIN_EXP-1).  */
  mpfr_t min;
  /* The greatest positive subnormal value.  */
  mpfr_t subnorm_max;
  /* The least positive subnormal value, 2^(MIN_EXP-MANT_DIG).  */
  mpfr_t subnorm_min;
} fp_format_desc;

/* List of floating-point formats, in the same order as the fp_format
   enumeration.  */
static fp_format_desc fp_formats[fp_num_formats] =
  {
    { "binary32", NULL, 24, 128, -125, {}, {}, {}, {}, {} },
    { "binary64", NULL, 53, 1024, -1021, {}, {}, {}, {}, {} },
    { "intel96", NULL, 64, 16384, -16381, {}, {}, {}, {}, {} },
    { "m68k96", NULL, 64, 16384, -16382, {}, {}, {}, {}, {} },
    { "binary128", NULL, 113, 16384, -16381, {}, {}, {}, {}, {} },
    { "ibm128", "0x1.fffffffffffff7ffffffffffff8p+1023",
      106, 1024, -968, {}, {}, {}, {}, {} },
  };

/* The supported rounding modes.  */
typedef enum
  {
    rm_downward,
    rm_tonearest,
    rm_towardzero,
    rm_upward,
    rm_num_modes,
    rm_first_mode = 0
  } rounding_mode;

/* Structure describing a single rounding mode.  */
typedef struct
{
  /* The name of the rounding mode.  */
  const char *name;
  /* The MPFR rounding mode.  */
  mpfr_rnd_t mpfr_mode;
  /* The MPC rounding mode.  */
  mpc_rnd_t mpc_mode;
} rounding_mode_desc;

/* List of rounding modes, in the same order as the rounding_mode
   enumeration.  */
static const rounding_mode_desc rounding_modes[rm_num_modes] =
  {
    { "downward", MPFR_RNDD, MPC_RNDDD },
    { "tonearest", MPFR_RNDN, MPC_RNDNN },
    { "towardzero", MPFR_RNDZ, MPC_RNDZZ },
    { "upward", MPFR_RNDU, MPC_RNDUU },
  };

/* The supported exceptions.  */
typedef enum
  {
    exc_divbyzero,
    exc_inexact,
    exc_invalid,
    exc_overflow,
    exc_underflow,
    exc_num_exceptions,
    exc_first_exception = 0
  } fp_exception;

/* List of exceptions, in the same order as the fp_exception
   enumeration.  */
static const char *const exceptions[exc_num_exceptions] =
  {
    "divbyzero",
    "inexact",
    "invalid",
    "overflow",
    "underflow",
  };

/* The internal precision to use for most MPFR calculations, which
   must be at least 2 more than the greatest precision of any
   supported floating-point format.  */
static int internal_precision;

/* A value that overflows all supported floating-point formats.  */
static mpfr_t global_max;

/* A value that is at most half the least subnormal in any
   floating-point format and so is rounded the same way as all
   sufficiently small positive values.  */
static mpfr_t global_min;

/* The maximum number of (real or integer) arguments to a function
   handled by this program (complex arguments count as two real
   arguments).  */
#define MAX_NARGS 4

/* The maximum number of (real or integer) return values from a
   function handled by this program.  */
#define MAX_NRET 2

/* A type of a function argument or return value.  */
typedef enum
  {
    /* No type (not a valid argument or return value).  */
    type_none,
    /* A floating-point value with the type corresponding to that of
       the function.  */
    type_fp,
    /* An integer value of type int.  */
    type_int,
    /* An integer value of type long.  */
    type_long,
    /* An integer value of type long long.  */
    type_long_long,
  } arg_ret_type;

/* A type of a generic real or integer value.  */
typedef enum
  {
    /* No type.  */
    gtype_none,
    /* Floating-point (represented with MPFR).  */
    gtype_fp,
    /* Integer (represented with GMP).  */
    gtype_int,
  } generic_value_type;

/* A generic value (argument or result).  */
typedef struct
{
  /* The type of this value.  */
  generic_value_type type;
  /* Its value.  */
  union
  {
    mpfr_t f;
    mpz_t i;
  } value;
} generic_value;

/* A type of input flag.  */
typedef enum
  {
    flag_ignore_zero_inf_sign,
    flag_xfail,
    flag_xfail_rounding,
    /* The "spurious" and "missing" flags must be in the same order as
       the fp_exception enumeration.  */
    flag_spurious_divbyzero,
    flag_spurious_inexact,
    flag_spurious_invalid,
    flag_spurious_overflow,
    flag_spurious_underflow,
    flag_spurious_errno,
    flag_missing_divbyzero,
    flag_missing_inexact,
    flag_missing_invalid,
    flag_missing_overflow,
    flag_missing_underflow,
    flag_missing_errno,
    num_input_flag_types,
    flag_first_flag = 0,
    flag_spurious_first = flag_spurious_divbyzero,
    flag_missing_first = flag_missing_divbyzero
  } input_flag_type;

/* List of flags, in the same order as the input_flag_type
   enumeration.  */
static const char *const input_flags[num_input_flag_types] =
  {
    "ignore-zero-inf-sign",
    "xfail",
    "xfail-rounding",
    "spurious-divbyzero",
    "spurious-inexact",
    "spurious-invalid",
    "spurious-overflow",
    "spurious-underflow",
    "spurious-errno",
    "missing-divbyzero",
    "missing-inexact",
    "missing-invalid",
    "missing-overflow",
    "missing-underflow",
    "missing-errno",
  };

/* An input flag, possibly conditional.  */
typedef struct
{
  /* The type of this flag.  */
  input_flag_type type;
  /* The conditions on this flag, as a string ":cond1:cond2..." or
     NULL.  */
  const char *cond;
} input_flag;

/* Structure describing a single test from the input file (which may
   expand into many tests in the output).  The choice of function,
   which implies the numbers and types of arguments and results, is
   implicit rather than stored in this structure (except as part of
   the source line).  */
typedef struct
{
  /* The text of the input line describing the test, including the
     trailing newline.  */
  const char *line;
  /* The number of combinations of interpretations of input values for
     different floating-point formats and rounding modes.  */
  size_t num_input_cases;
  /* The corresponding lists of inputs.  */
  generic_value **inputs;
  /* The number of flags for this test.  */
  size_t num_flags;
  /* The corresponding list of flags.  */
  input_flag *flags;
  /* The old output for this test.  */
  const char *old_output;
} input_test;

/* Ways to calculate a function.  */
typedef enum
  {
    /* MPFR function with a single argument and result.  */
    mpfr_f_f,
    /* MPFR function with two arguments and one result.  */
    mpfr_ff_f,
    /* MPFR function with three arguments and one result.  */
    mpfr_fff_f,
    /* MPFR function with a single argument and floating-point and
       integer results.  */
    mpfr_f_f1,
    /* MPFR function with integer and floating-point arguments and one
       result.  */
    mpfr_if_f,
    /* MPFR function with a single argument and two floating-point
       results.  */
    mpfr_f_11,
    /* MPC function with a single complex argument and one real
       result.  */
    mpc_c_f,
    /* MPC function with a single complex argument and one complex
       result.  */
    mpc_c_c,
    /* MPC function with two complex arguments and one complex
       result.  */
    mpc_cc_c,
  } func_calc_method;

/* Description of how to calculate a function.  */
typedef struct
{
  /* Which method is used to calculate the function.  */
  func_calc_method method;
  /* The specific function called.  */
  union
  {
    int (*mpfr_f_f) (mpfr_t, const mpfr_t, mpfr_rnd_t);
    int (*mpfr_ff_f) (mpfr_t, const mpfr_t, const mpfr_t, mpfr_rnd_t);
    int (*mpfr_fff_f) (mpfr_t, const mpfr_t, const mpfr_t, const mpfr_t,
		       mpfr_rnd_t);
    int (*mpfr_f_f1) (mpfr_t, int *, const mpfr_t, mpfr_rnd_t);
    int (*mpfr_if_f) (mpfr_t, long, const mpfr_t, mpfr_rnd_t);
    int (*mpfr_f_11) (mpfr_t, mpfr_t, const mpfr_t, mpfr_rnd_t);
    int (*mpc_c_f) (mpfr_t, const mpc_t, mpfr_rnd_t);
    int (*mpc_c_c) (mpc_t, const mpc_t, mpc_rnd_t);
    int (*mpc_cc_c) (mpc_t, const mpc_t, const mpc_t, mpc_rnd_t);
  } func;
} func_calc_desc;

/* Structure describing a function handled by this program.  */
typedef struct
{
  /* The name of the function.  */
  const char *name;
  /* The number of arguments.  */
  size_t num_args;
  /* The types of the arguments.  */
  arg_ret_type arg_types[MAX_NARGS];
  /* The number of return values.  */
  size_t num_ret;
  /* The types of the return values.  */
  arg_ret_type ret_types[MAX_NRET];
  /* Whether the function has exactly determined results and
     exceptions.  */
  bool exact;
  /* Whether the function is a complex function, so errno setting is
     optional.  */
  bool complex_fn;
  /* Whether to treat arguments given as floating-point constants as
     exact only, rather than rounding them up and down to all
     formats.  */
  bool exact_args;
  /* How to calculate this function.  */
  func_calc_desc calc;
  /* The number of tests allocated for this function.  */
  size_t num_tests_alloc;
  /* The number of tests for this function.  */
  size_t num_tests;
  /* The tests themselves.  */
  input_test *tests;
} test_function;

#define ARGS1(T1) 1, { T1 }
#define ARGS2(T1, T2) 2, { T1, T2 }
#define ARGS3(T1, T2, T3) 3, { T1, T2, T3 }
#define ARGS4(T1, T2, T3, T4) 4, { T1, T2, T3, T4 }
#define RET1(T1) 1, { T1 }
#define RET2(T1, T2) 2, { T1, T2 }
#define CALC(TYPE, FN) { TYPE, { .TYPE = FN } }
#define FUNC(NAME, ARGS, RET, EXACT, COMPLEX_FN, EXACT_ARGS, CALC)	\
  {									\
    NAME, ARGS, RET, EXACT, COMPLEX_FN, EXACT_ARGS, CALC, 0, 0, NULL	\
  }

#define FUNC_mpfr_f_f(NAME, MPFR_FUNC, EXACT)				\
  FUNC (NAME, ARGS1 (type_fp), RET1 (type_fp), EXACT, false, false,	\
	CALC (mpfr_f_f, MPFR_FUNC))
#define FUNC_mpfr_ff_f(NAME, MPFR_FUNC, EXACT)				\
  FUNC (NAME, ARGS2 (type_fp, type_fp), RET1 (type_fp), EXACT, false,	\
	false, CALC (mpfr_ff_f, MPFR_FUNC))
#define FUNC_mpfr_if_f(NAME, MPFR_FUNC, EXACT)				\
  FUNC (NAME, ARGS2 (type_int, type_fp), RET1 (type_fp), EXACT, false,	\
	false, CALC (mpfr_if_f, MPFR_FUNC))
#define FUNC_mpc_c_f(NAME, MPFR_FUNC, EXACT)				\
  FUNC (NAME, ARGS2 (type_fp, type_fp), RET1 (type_fp), EXACT, true,	\
	false, CALC (mpc_c_f, MPFR_FUNC))
#define FUNC_mpc_c_c(NAME, MPFR_FUNC, EXACT)				\
  FUNC (NAME, ARGS2 (type_fp, type_fp), RET2 (type_fp, type_fp), EXACT, \
	true, false, CALC (mpc_c_c, MPFR_FUNC))

/* List of functions handled by this program.  */
static test_function test_functions[] =
  {
    FUNC_mpfr_f_f ("acos", mpfr_acos, false),
    FUNC_mpfr_f_f ("acosh", mpfr_acosh, false),
    FUNC_mpfr_ff_f ("add", mpfr_add, true),
    FUNC_mpfr_f_f ("asin", mpfr_asin, false),
    FUNC_mpfr_f_f ("asinh", mpfr_asinh, false),
    FUNC_mpfr_f_f ("atan", mpfr_atan, false),
    FUNC_mpfr_ff_f ("atan2", mpfr_atan2, false),
    FUNC_mpfr_f_f ("atanh", mpfr_atanh, false),
    FUNC_mpc_c_f ("cabs", mpc_abs, false),
    FUNC_mpc_c_c ("cacos", mpc_acos, false),
    FUNC_mpc_c_c ("cacosh", mpc_acosh, false),
    FUNC_mpc_c_f ("carg", mpc_arg, false),
    FUNC_mpc_c_c ("casin", mpc_asin, false),
    FUNC_mpc_c_c ("casinh", mpc_asinh, false),
    FUNC_mpc_c_c ("catan", mpc_atan, false),
    FUNC_mpc_c_c ("catanh", mpc_atanh, false),
    FUNC_mpfr_f_f ("cbrt", mpfr_cbrt, false),
    FUNC_mpc_c_c ("ccos", mpc_cos, false),
    FUNC_mpc_c_c ("ccosh", mpc_cosh, false),
    FUNC_mpc_c_c ("cexp", mpc_exp, false),
    FUNC_mpc_c_c ("clog", mpc_log, false),
    FUNC_mpc_c_c ("clog10", mpc_log10, false),
    FUNC_mpfr_f_f ("cos", mpfr_cos, false),
    FUNC_mpfr_f_f ("cosh", mpfr_cosh, false),
    FUNC ("cpow", ARGS4 (type_fp, type_fp, type_fp, type_fp),
	  RET2 (type_fp, type_fp), false, true, false,
	  CALC (mpc_cc_c, mpc_pow)),
    FUNC_mpc_c_c ("csin", mpc_sin, false),
    FUNC_mpc_c_c ("csinh", mpc_sinh, false),
    FUNC_mpc_c_c ("csqrt", mpc_sqrt, false),
    FUNC_mpc_c_c ("ctan", mpc_tan, false),
    FUNC_mpc_c_c ("ctanh", mpc_tanh, false),
    FUNC_mpfr_ff_f ("div", mpfr_div, true),
    FUNC_mpfr_f_f ("erf", mpfr_erf, false),
    FUNC_mpfr_f_f ("erfc", mpfr_erfc, false),
    FUNC_mpfr_f_f ("exp", mpfr_exp, false),
    FUNC_mpfr_f_f ("exp10", mpfr_exp10, false),
    FUNC_mpfr_f_f ("exp2", mpfr_exp2, false),
    FUNC_mpfr_f_f ("expm1", mpfr_expm1, false),
    FUNC ("fma", ARGS3 (type_fp, type_fp, type_fp), RET1 (type_fp),
	  true, false, true, CALC (mpfr_fff_f, mpfr_fma)),
    FUNC_mpfr_ff_f ("hypot", mpfr_hypot, false),
    FUNC_mpfr_f_f ("j0", mpfr_j0, false),
    FUNC_mpfr_f_f ("j1", mpfr_j1, false),
    FUNC_mpfr_if_f ("jn", mpfr_jn, false),
    FUNC ("lgamma", ARGS1 (type_fp), RET2 (type_fp, type_int), false, false,
	  false, CALC (mpfr_f_f1, mpfr_lgamma)),
    FUNC_mpfr_f_f ("log", mpfr_log, false),
    FUNC_mpfr_f_f ("log10", mpfr_log10, false),
    FUNC_mpfr_f_f ("log1p", mpfr_log1p, false),
    FUNC_mpfr_f_f ("log2", mpfr_log2, false),
    FUNC_mpfr_ff_f ("mul", mpfr_mul, true),
    FUNC_mpfr_ff_f ("pow", mpfr_pow, false),
    FUNC_mpfr_f_f ("sin", mpfr_sin, false),
    FUNC ("sincos", ARGS1 (type_fp), RET2 (type_fp, type_fp), false, false,
	  false, CALC (mpfr_f_11, mpfr_sin_cos)),
    FUNC_mpfr_f_f ("sinh", mpfr_sinh, false),
    FUNC_mpfr_ff_f ("sub", mpfr_sub, true),
    FUNC_mpfr_f_f ("sqrt", mpfr_sqrt, true),
    FUNC_mpfr_f_f ("tan", mpfr_tan, false),
    FUNC_mpfr_f_f ("tanh", mpfr_tanh, false),
    FUNC_mpfr_f_f ("tgamma", mpfr_gamma, false),
    FUNC_mpfr_f_f ("y0", mpfr_y0, false),
    FUNC_mpfr_f_f ("y1", mpfr_y1, false),
    FUNC_mpfr_if_f ("yn", mpfr_yn, false),
  };

/* Allocate memory, with error checking.  */

static void *
xmalloc (size_t n)
{
  void *p = malloc (n);
  if (p == NULL)
    error (EXIT_FAILURE, errno, "xmalloc failed");
  return p;
}

static void *
xrealloc (void *p, size_t n)
{
  p = realloc (p, n);
  if (p == NULL)
    error (EXIT_FAILURE, errno, "xrealloc failed");
  return p;
}

static char *
xstrdup (const char *s)
{
  char *p = strdup (s);
  if (p == NULL)
    error (EXIT_FAILURE, errno, "xstrdup failed");
  return p;
}

/* Assert that the result of an MPFR operation was exact; that is,
   that the returned ternary value was 0.  */

static void
assert_exact (int i)
{
  assert (i == 0);
}

/* Return the generic type of an argument or return value type T.  */

static generic_value_type
generic_arg_ret_type (arg_ret_type t)
{
  switch (t)
    {
    case type_fp:
      return gtype_fp;

    case type_int:
    case type_long:
    case type_long_long:
      return gtype_int;

    default:
      abort ();
    }
}

/* Free a generic_value *V.  */

static void
generic_value_free (generic_value *v)
{
  switch (v->type)
    {
    case gtype_fp:
      mpfr_clear (v->value.f);
      break;

    case gtype_int:
      mpz_clear (v->value.i);
      break;

    default:
      abort ();
    }
}

/* Copy a generic_value *SRC to *DEST.  */

static void
generic_value_copy (generic_value *dest, const generic_value *src)
{
  dest->type = src->type;
  switch (src->type)
    {
    case gtype_fp:
      mpfr_init (dest->value.f);
      assert_exact (mpfr_set (dest->value.f, src->value.f, MPFR_RNDN));
      break;

    case gtype_int:
      mpz_init (dest->value.i);
      mpz_set (dest->value.i, src->value.i);
      break;

    default:
      abort ();
    }
}

/* Initialize data for floating-point formats.  */

static void
init_fp_formats (void)
{
  int global_max_exp = 0, global_min_subnorm_exp = 0;
  for (fp_format f = fp_first_format; f < fp_num_formats; f++)
    {
      if (fp_formats[f].mant_dig + 2 > internal_precision)
	internal_precision = fp_formats[f].mant_dig + 2;
      if (fp_formats[f].max_exp > global_max_exp)
	global_max_exp = fp_formats[f].max_exp;
      int min_subnorm_exp = fp_formats[f].min_exp - fp_formats[f].mant_dig;
      if (min_subnorm_exp < global_min_subnorm_exp)
	global_min_subnorm_exp = min_subnorm_exp;
      mpfr_init2 (fp_formats[f].max, fp_formats[f].mant_dig);
      if (fp_formats[f].max_string != NULL)
	{
	  char *ep = NULL;
	  assert_exact (mpfr_strtofr (fp_formats[f].max,
				      fp_formats[f].max_string,
				      &ep, 0, MPFR_RNDN));
	  assert (*ep == 0);
	}
      else
	{
	  assert_exact (mpfr_set_ui_2exp (fp_formats[f].max, 1,
					  fp_formats[f].max_exp,
					  MPFR_RNDN));
	  mpfr_nextbelow (fp_formats[f].max);
	}
      mpfr_init2 (fp_formats[f].min, fp_formats[f].mant_dig);
      assert_exact (mpfr_set_ui_2exp (fp_formats[f].min, 1,
				      fp_formats[f].min_exp - 1,
				      MPFR_RNDN));
      mpfr_init2 (fp_formats[f].min_plus_half, fp_formats[f].mant_dig + 1);
      assert_exact (mpfr_set (fp_formats[f].min_plus_half,
			      fp_formats[f].min, MPFR_RNDN));
      mpfr_nextabove (fp_formats[f].min_plus_half);
      mpfr_init2 (fp_formats[f].subnorm_max, fp_formats[f].mant_dig);
      assert_exact (mpfr_set (fp_formats[f].subnorm_max, fp_formats[f].min,
			      MPFR_RNDN));
      mpfr_nextbelow (fp_formats[f].subnorm_max);
      mpfr_nextbelow (fp_formats[f].subnorm_max);
      mpfr_init2 (fp_formats[f].subnorm_min, fp_formats[f].mant_dig);
      assert_exact (mpfr_set_ui_2exp (fp_formats[f].subnorm_min, 1,
				      min_subnorm_exp, MPFR_RNDN));
    }
  mpfr_set_default_prec (internal_precision);
  mpfr_init (global_max);
  assert_exact (mpfr_set_ui_2exp (global_max, 1, global_max_exp, MPFR_RNDN));
  mpfr_init (global_min);
  assert_exact (mpfr_set_ui_2exp (global_min, 1, global_min_subnorm_exp - 1,
				  MPFR_RNDN));
}

/* Fill in mpfr_t values for special strings in input arguments.  */

static size_t
special_fill_max (mpfr_t res0, mpfr_t res1 __attribute__ ((unused)),
		  fp_format format)
{
  mpfr_init2 (res0, fp_formats[format].mant_dig);
  assert_exact (mpfr_set (res0, fp_formats[format].max, MPFR_RNDN));
  return 1;
}

static size_t
special_fill_minus_max (mpfr_t res0, mpfr_t res1 __attribute__ ((unused)),
			fp_format format)
{
  mpfr_init2 (res0, fp_formats[format].mant_dig);
  assert_exact (mpfr_neg (res0, fp_formats[format].max, MPFR_RNDN));
  return 1;
}

static size_t
special_fill_min (mpfr_t res0, mpfr_t res1 __attribute__ ((unused)),
		  fp_format format)
{
  mpfr_init2 (res0, fp_formats[format].mant_dig);
  assert_exact (mpfr_set (res0, fp_formats[format].min, MPFR_RNDN));
  return 1;
}

static size_t
special_fill_minus_min (mpfr_t res0, mpfr_t res1 __attribute__ ((unused)),
			fp_format format)
{
  mpfr_init2 (res0, fp_formats[format].mant_dig);
  assert_exact (mpfr_neg (res0, fp_formats[format].min, MPFR_RNDN));
  return 1;
}

static size_t
special_fill_min_subnorm (mpfr_t res0, mpfr_t res1 __attribute__ ((unused)),
			  fp_format format)
{
  mpfr_init2 (res0, fp_formats[format].mant_dig);
  assert_exact (mpfr_set (res0, fp_formats[format].subnorm_min, MPFR_RNDN));
  return 1;
}

static size_t
special_fill_minus_min_subnorm (mpfr_t res0,
				mpfr_t res1 __attribute__ ((unused)),
				fp_format format)
{
  mpfr_init2 (res0, fp_formats[format].mant_dig);
  assert_exact (mpfr_neg (res0, fp_formats[format].subnorm_min, MPFR_RNDN));
  return 1;
}

static size_t
special_fill_min_subnorm_p120 (mpfr_t res0,
			       mpfr_t res1 __attribute__ ((unused)),
			       fp_format format)
{
  mpfr_init2 (res0, fp_formats[format].mant_dig);
  assert_exact (mpfr_mul_2ui (res0, fp_formats[format].subnorm_min,
			      120, MPFR_RNDN));
  return 1;
}

static size_t
special_fill_pi (mpfr_t res0, mpfr_t res1, fp_format format)
{
  mpfr_init2 (res0, fp_formats[format].mant_dig);
  mpfr_const_pi (res0, MPFR_RNDU);
  mpfr_init2 (res1, fp_formats[format].mant_dig);
  mpfr_const_pi (res1, MPFR_RNDD);
  return 2;
}

static size_t
special_fill_minus_pi (mpfr_t res0, mpfr_t res1, fp_format format)
{
  mpfr_init2 (res0, fp_formats[format].mant_dig);
  mpfr_const_pi (res0, MPFR_RNDU);
  assert_exact (mpfr_neg (res0, res0, MPFR_RNDN));
  mpfr_init2 (res1, fp_formats[format].mant_dig);
  mpfr_const_pi (res1, MPFR_RNDD);
  assert_exact (mpfr_neg (res1, res1, MPFR_RNDN));
  return 2;
}

static size_t
special_fill_pi_2 (mpfr_t res0, mpfr_t res1, fp_format format)
{
  mpfr_init2 (res0, fp_formats[format].mant_dig);
  mpfr_const_pi (res0, MPFR_RNDU);
  assert_exact (mpfr_div_ui (res0, res0, 2, MPFR_RNDN));
  mpfr_init2 (res1, fp_formats[format].mant_dig);
  mpfr_const_pi (res1, MPFR_RNDD);
  assert_exact (mpfr_div_ui (res1, res1, 2, MPFR_RNDN));
  return 2;
}

static size_t
special_fill_minus_pi_2 (mpfr_t res0, mpfr_t res1, fp_format format)
{
  mpfr_init2 (res0, fp_formats[format].mant_dig);
  mpfr_const_pi (res0, MPFR_RNDU);
  assert_exact (mpfr_div_ui (res0, res0, 2, MPFR_RNDN));
  assert_exact (mpfr_neg (res0, res0, MPFR_RNDN));
  mpfr_init2 (res1, fp_formats[format].mant_dig);
  mpfr_const_pi (res1, MPFR_RNDD);
  assert_exact (mpfr_div_ui (res1, res1, 2, MPFR_RNDN));
  assert_exact (mpfr_neg (res1, res1, MPFR_RNDN));
  return 2;
}

static size_t
special_fill_pi_4 (mpfr_t res0, mpfr_t res1, fp_format format)
{
  mpfr_init2 (res0, fp_formats[format].mant_dig);
  assert_exact (mpfr_set_si (res0, 1, MPFR_RNDN));
  mpfr_atan (res0, res0, MPFR_RNDU);
  mpfr_init2 (res1, fp_formats[format].mant_dig);
  assert_exact (mpfr_set_si (res1, 1, MPFR_RNDN));
  mpfr_atan (res1, res1, MPFR_RNDD);
  return 2;
}

static size_t
special_fill_pi_6 (mpfr_t res0, mpfr_t res1, fp_format format)
{
  mpfr_init2 (res0, fp_formats[format].mant_dig);
  assert_exact (mpfr_set_si_2exp (res0, 1, -1, MPFR_RNDN));
  mpfr_asin (res0, res0, MPFR_RNDU);
  mpfr_init2 (res1, fp_formats[format].mant_dig);
  assert_exact (mpfr_set_si_2exp (res1, 1, -1, MPFR_RNDN));
  mpfr_asin (res1, res1, MPFR_RNDD);
  return 2;
}

static size_t
special_fill_minus_pi_6 (mpfr_t res0, mpfr_t res1, fp_format format)
{
  mpfr_init2 (res0, fp_formats[format].mant_dig);
  assert_exact (mpfr_set_si_2exp (res0, -1, -1, MPFR_RNDN));
  mpfr_asin (res0, res0, MPFR_RNDU);
  mpfr_init2 (res1, fp_formats[format].mant_dig);
  assert_exact (mpfr_set_si_2exp (res1, -1, -1, MPFR_RNDN));
  mpfr_asin (res1, res1, MPFR_RNDD);
  return 2;
}

static size_t
special_fill_pi_3 (mpfr_t res0, mpfr_t res1, fp_format format)
{
  mpfr_init2 (res0, fp_formats[format].mant_dig);
  assert_exact (mpfr_set_si_2exp (res0, 1, -1, MPFR_RNDN));
  mpfr_acos (res0, res0, MPFR_RNDU);
  mpfr_init2 (res1, fp_formats[format].mant_dig);
  assert_exact (mpfr_set_si_2exp (res1, 1, -1, MPFR_RNDN));
  mpfr_acos (res1, res1, MPFR_RNDD);
  return 2;
}

static size_t
special_fill_2pi_3 (mpfr_t res0, mpfr_t res1, fp_format format)
{
  mpfr_init2 (res0, fp_formats[format].mant_dig);
  assert_exact (mpfr_set_si_2exp (res0, -1, -1, MPFR_RNDN));
  mpfr_acos (res0, res0, MPFR_RNDU);
  mpfr_init2 (res1, fp_formats[format].mant_dig);
  assert_exact (mpfr_set_si_2exp (res1, -1, -1, MPFR_RNDN));
  mpfr_acos (res1, res1, MPFR_RNDD);
  return 2;
}

static size_t
special_fill_2pi (mpfr_t res0, mpfr_t res1, fp_format format)
{
  mpfr_init2 (res0, fp_formats[format].mant_dig);
  mpfr_const_pi (res0, MPFR_RNDU);
  assert_exact (mpfr_mul_ui (res0, res0, 2, MPFR_RNDN));
  mpfr_init2 (res1, fp_formats[format].mant_dig);
  mpfr_const_pi (res1, MPFR_RNDD);
  assert_exact (mpfr_mul_ui (res1, res1, 2, MPFR_RNDN));
  return 2;
}

static size_t
special_fill_e (mpfr_t res0, mpfr_t res1, fp_format format)
{
  mpfr_init2 (res0, fp_formats[format].mant_dig);
  assert_exact (mpfr_set_si (res0, 1, MPFR_RNDN));
  mpfr_exp (res0, res0, MPFR_RNDU);
  mpfr_init2 (res1, fp_formats[format].mant_dig);
  assert_exact (mpfr_set_si (res1, 1, MPFR_RNDN));
  mpfr_exp (res1, res1, MPFR_RNDD);
  return 2;
}

static size_t
special_fill_1_e (mpfr_t res0, mpfr_t res1, fp_format format)
{
  mpfr_init2 (res0, fp_formats[format].mant_dig);
  assert_exact (mpfr_set_si (res0, -1, MPFR_RNDN));
  mpfr_exp (res0, res0, MPFR_RNDU);
  mpfr_init2 (res1, fp_formats[format].mant_dig);
  assert_exact (mpfr_set_si (res1, -1, MPFR_RNDN));
  mpfr_exp (res1, res1, MPFR_RNDD);
  return 2;
}

static size_t
special_fill_e_minus_1 (mpfr_t res0, mpfr_t res1, fp_format format)
{
  mpfr_init2 (res0, fp_formats[format].mant_dig);
  assert_exact (mpfr_set_si (res0, 1, MPFR_RNDN));
  mpfr_expm1 (res0, res0, MPFR_RNDU);
  mpfr_init2 (res1, fp_formats[format].mant_dig);
  assert_exact (mpfr_set_si (res1, 1, MPFR_RNDN));
  mpfr_expm1 (res1, res1, MPFR_RNDD);
  return 2;
}

/* A special string accepted in input arguments.  */
typedef struct
{
  /* The string.  */
  const char *str;
  /* The function that interprets it for a given floating-point
     format, filling in up to two mpfr_t values and returning the
     number of values filled.  */
  size_t (*func) (mpfr_t, mpfr_t, fp_format);
} special_real_input;

/* List of special strings accepted in input arguments.  */

static const special_real_input special_real_inputs[] =
  {
    { "max", special_fill_max },
    { "-max", special_fill_minus_max },
    { "min", special_fill_min },
    { "-min", special_fill_minus_min },
    { "min_subnorm", special_fill_min_subnorm },
    { "-min_subnorm", special_fill_minus_min_subnorm },
    { "min_subnorm_p120", special_fill_min_subnorm_p120 },
    { "pi", special_fill_pi },
    { "-pi", special_fill_minus_pi },
    { "pi/2", special_fill_pi_2 },
    { "-pi/2", special_fill_minus_pi_2 },
    { "pi/4", special_fill_pi_4 },
    { "pi/6", special_fill_pi_6 },
    { "-pi/6", special_fill_minus_pi_6 },
    { "pi/3", special_fill_pi_3 },
    { "2pi/3", special_fill_2pi_3 },
    { "2pi", special_fill_2pi },
    { "e", special_fill_e },
    { "1/e", special_fill_1_e },
    { "e-1", special_fill_e_minus_1 },
  };

/* Given a real number R computed in round-to-zero mode, set the
   lowest bit as a sticky bit if INEXACT, and saturate the exponent
   range for very large or small values.  */

static void
adjust_real (mpfr_t r, bool inexact)
{
  if (!inexact)
    return;
  /* NaNs are exact, as are infinities in round-to-zero mode.  */
  assert (mpfr_number_p (r));
  if (mpfr_cmpabs (r, global_min) < 0)
    assert_exact (mpfr_copysign (r, global_min, r, MPFR_RNDN));
  else if (mpfr_cmpabs (r, global_max) > 0)
    assert_exact (mpfr_copysign (r, global_max, r, MPFR_RNDN));
  else
    {
      mpz_t tmp;
      mpz_init (tmp);
      mpfr_exp_t e = mpfr_get_z_2exp (tmp, r);
      if (mpz_sgn (tmp) < 0)
	{
	  mpz_neg (tmp, tmp);
	  mpz_setbit (tmp, 0);
	  mpz_neg (tmp, tmp);
	}
      else
	mpz_setbit (tmp, 0);
      assert_exact (mpfr_set_z_2exp (r, tmp, e, MPFR_RNDN));
      mpz_clear (tmp);
    }
}

/* Given a finite real number R with sticky bit, compute the roundings
   to FORMAT in each rounding mode, storing the results in RES, the
   before-rounding exceptions in EXC_BEFORE and the after-rounding
   exceptions in EXC_AFTER.  */

static void
round_real (mpfr_t res[rm_num_modes],
	    unsigned int exc_before[rm_num_modes],
	    unsigned int exc_after[rm_num_modes],
	    mpfr_t r, fp_format format)
{
  assert (mpfr_number_p (r));
  for (rounding_mode m = rm_first_mode; m < rm_num_modes; m++)
    {
      mpfr_init2 (res[m], fp_formats[format].mant_dig);
      exc_before[m] = exc_after[m] = 0;
      bool inexact = mpfr_set (res[m], r, rounding_modes[m].mpfr_mode);
      if (mpfr_cmpabs (res[m], fp_formats[format].max) > 0)
	{
	  inexact = true;
	  exc_before[m] |= 1U << exc_overflow;
	  exc_after[m] |= 1U << exc_overflow;
	  bool overflow_inf;
	  switch (m)
	    {
	    case rm_tonearest:
	      overflow_inf = true;
	      break;
	    case rm_towardzero:
	      overflow_inf = false;
	      break;
	    case rm_downward:
	      overflow_inf = mpfr_signbit (res[m]);
	      break;
	    case rm_upward:
	      overflow_inf = !mpfr_signbit (res[m]);
	      break;
	    default:
	      abort ();
	    }
	  if (overflow_inf)
	    mpfr_set_inf (res[m], mpfr_signbit (res[m]) ? -1 : 1);
	  else
	    assert_exact (mpfr_copysign (res[m], fp_formats[format].max,
					 res[m], MPFR_RNDN));
	}
      if (mpfr_cmpabs (r, fp_formats[format].min) < 0)
	{
	  /* Tiny before rounding; may or may not be tiny after
	     rounding, and underflow applies only if also inexact
	     around rounding to a possibly subnormal value.  */
	  bool tiny_after_rounding
	    = mpfr_cmpabs (res[m], fp_formats[format].min) < 0;
	  /* To round to a possibly subnormal value, and determine
	     inexactness as a subnormal in the process, scale up and
	     round to integer, then scale back down.  */
	  mpfr_t tmp;
	  mpfr_init (tmp);
	  assert_exact (mpfr_mul_2si (tmp, r, (fp_formats[format].mant_dig
					       - fp_formats[format].min_exp),
				      MPFR_RNDN));
	  int rint_res = mpfr_rint (tmp, tmp, rounding_modes[m].mpfr_mode);
	  /* The integer must be representable.  */
	  assert (rint_res == 0 || rint_res == 2 || rint_res == -2);
	  /* If rounding to full precision was inexact, so must
	     rounding to subnormal precision be inexact.  */
	  if (inexact)
	    assert (rint_res != 0);
	  else
	    inexact = rint_res != 0;
	  assert_exact (mpfr_mul_2si (res[m], tmp,
				      (fp_formats[format].min_exp
				       - fp_formats[format].mant_dig),
				      MPFR_RNDN));
	  mpfr_clear (tmp);
	  if (inexact)
	    {
	      exc_before[m] |= 1U << exc_underflow;
	      if (tiny_after_rounding)
		exc_after[m] |= 1U << exc_underflow;
	    }
	}
      if (inexact)
	{
	  exc_before[m] |= 1U << exc_inexact;
	  exc_after[m] |= 1U << exc_inexact;
	}
    }
}

/* Handle the input argument at ARG (NUL-terminated), updating the
   lists of test inputs in IT accordingly.  NUM_PREV_ARGS arguments
   are already in those lists.  If EXACT_ARGS, interpret a value given
   as a floating-point constant exactly (it must be exact for some
   supported format) rather than rounding up and down.  The argument,
   of type GTYPE, comes from file FILENAME, line LINENO.  */

static void
handle_input_arg (const char *arg, input_test *it, size_t num_prev_args,
		  generic_value_type gtype, bool exact_args,
		  const char *filename, unsigned int lineno)
{
  size_t num_values = 0;
  generic_value values[2 * fp_num_formats];
  bool check_empty_list = false;
  switch (gtype)
    {
    case gtype_fp:
      for (fp_format f = fp_first_format; f < fp_num_formats; f++)
	{
	  mpfr_t extra_values[2];
	  size_t num_extra_values = 0;
	  for (size_t i = 0; i < ARRAY_SIZE (special_real_inputs); i++)
	    {
	      if (strcmp (arg, special_real_inputs[i].str) == 0)
		{
		  num_extra_values
		    = special_real_inputs[i].func (extra_values[0],
						   extra_values[1], f);
		  assert (num_extra_values > 0
			  && num_extra_values <= ARRAY_SIZE (extra_values));
		  break;
		}
	    }
	  if (num_extra_values == 0)
	    {
	      mpfr_t tmp;
	      char *ep;
	      if (exact_args)
		check_empty_list = true;
	      mpfr_init (tmp);
	      bool inexact = mpfr_strtofr (tmp, arg, &ep, 0, MPFR_RNDZ);
	      if (*ep != 0 || !mpfr_number_p (tmp))
		error_at_line (EXIT_FAILURE, 0, filename, lineno,
			       "bad floating-point argument: '%s'", arg);
	      adjust_real (tmp, inexact);
	      mpfr_t rounded[rm_num_modes];
	      unsigned int exc_before[rm_num_modes];
	      unsigned int exc_after[rm_num_modes];
	      round_real (rounded, exc_before, exc_after, tmp, f);
	      mpfr_clear (tmp);
	      if (mpfr_number_p (rounded[rm_upward])
		  && (!exact_args || mpfr_equal_p (rounded[rm_upward],
						   rounded[rm_downward])))
		{
		  mpfr_init2 (extra_values[num_extra_values],
			      fp_formats[f].mant_dig);
		  assert_exact (mpfr_set (extra_values[num_extra_values],
					  rounded[rm_upward], MPFR_RNDN));
		  num_extra_values++;
		}
	      if (mpfr_number_p (rounded[rm_downward]) && !exact_args)
		{
		  mpfr_init2 (extra_values[num_extra_values],
			      fp_formats[f].mant_dig);
		  assert_exact (mpfr_set (extra_values[num_extra_values],
					  rounded[rm_downward], MPFR_RNDN));
		  num_extra_values++;
		}
	      for (rounding_mode m = rm_first_mode; m < rm_num_modes; m++)
		mpfr_clear (rounded[m]);
	    }
	  for (size_t i = 0; i < num_extra_values; i++)
	    {
	      bool found = false;
	      for (size_t j = 0; j < num_values; j++)
		{
		  if (mpfr_equal_p (values[j].value.f, extra_values[i])
		      && ((mpfr_signbit (values[j].value.f) != 0)
			  == (mpfr_signbit (extra_values[i]) != 0)))
		    {
		      found = true;
		      break;
		    }
		}
	      if (!found)
		{
		  assert (num_values < ARRAY_SIZE (values));
		  values[num_values].type = gtype_fp;
		  mpfr_init2 (values[num_values].value.f,
			      fp_formats[f].mant_dig);
		  assert_exact (mpfr_set (values[num_values].value.f,
					  extra_values[i], MPFR_RNDN));
		  num_values++;
		}
	      mpfr_clear (extra_values[i]);
	    }
	}
      break;

    case gtype_int:
      num_values = 1;
      values[0].type = gtype_int;
      int ret = mpz_init_set_str (values[0].value.i, arg, 0);
      if (ret != 0)
	error_at_line (EXIT_FAILURE, 0, filename, lineno,
		       "bad integer argument: '%s'", arg);
      break;

    default:
      abort ();
    }
  if (check_empty_list && num_values == 0)
    error_at_line (EXIT_FAILURE, 0, filename, lineno,
		   "floating-point argument not exact for any format: '%s'",
		   arg);
  assert (num_values > 0 && num_values <= ARRAY_SIZE (values));
  if (it->num_input_cases >= SIZE_MAX / num_values)
    error_at_line (EXIT_FAILURE, 0, filename, lineno, "too many input cases");
  generic_value **old_inputs = it->inputs;
  size_t new_num_input_cases = it->num_input_cases * num_values;
  generic_value **new_inputs = xmalloc (new_num_input_cases
					* sizeof (new_inputs[0]));
  for (size_t i = 0; i < it->num_input_cases; i++)
    {
      for (size_t j = 0; j < num_values; j++)
	{
	  size_t idx = i * num_values + j;
	  new_inputs[idx] = xmalloc ((num_prev_args + 1)
				     * sizeof (new_inputs[idx][0]));
	  for (size_t k = 0; k < num_prev_args; k++)
	    generic_value_copy (&new_inputs[idx][k], &old_inputs[i][k]);
	  generic_value_copy (&new_inputs[idx][num_prev_args], &values[j]);
	}
      for (size_t j = 0; j < num_prev_args; j++)
	generic_value_free (&old_inputs[i][j]);
      free (old_inputs[i]);
    }
  free (old_inputs);
  for (size_t i = 0; i < num_values; i++)
    generic_value_free (&values[i]);
  it->inputs = new_inputs;
  it->num_input_cases = new_num_input_cases;
}

/* Handle the input flag ARG (NUL-terminated), storing it in *FLAG.
   The flag comes from file FILENAME, line LINENO.  */

static void
handle_input_flag (char *arg, input_flag *flag,
		   const char *filename, unsigned int lineno)
{
  char *ep = strchr (arg, ':');
  if (ep == NULL)
    {
      ep = strchr (arg, 0);
      assert (ep != NULL);
    }
  char c = *ep;
  *ep = 0;
  bool found = false;
  for (input_flag_type i = flag_first_flag; i < num_input_flag_types; i++)
    {
      if (strcmp (arg, input_flags[i]) == 0)
	{
	  found = true;
	  flag->type = i;
	  break;
	}
    }
  if (!found)
    error_at_line (EXIT_FAILURE, 0, filename, lineno, "unknown flag: '%s'",
		   arg);
  *ep = c;
  if (c == 0)
    flag->cond = NULL;
  else
    flag->cond = xstrdup (ep);
}

/* Add the test LINE (file FILENAME, line LINENO) to the test
   data.  */

static void
add_test (char *line, const char *filename, unsigned int lineno)
{
  size_t num_tokens = 1;
  char *p = line;
  while ((p = strchr (p, ' ')) != NULL)
    {
      num_tokens++;
      p++;
    }
  if (num_tokens < 2)
    error_at_line (EXIT_FAILURE, 0, filename, lineno,
		   "line too short: '%s'", line);
  p = strchr (line, ' ');
  size_t func_name_len = p - line;
  for (size_t i = 0; i < ARRAY_SIZE (test_functions); i++)
    {
      if (func_name_len == strlen (test_functions[i].name)
	  && strncmp (line, test_functions[i].name, func_name_len) == 0)
	{
	  test_function *tf = &test_functions[i];
	  if (num_tokens < 1 + tf->num_args)
	    error_at_line (EXIT_FAILURE, 0, filename, lineno,
			   "line too short: '%s'", line);
	  if (tf->num_tests == tf->num_tests_alloc)
	    {
	      tf->num_tests_alloc = 2 * tf->num_tests_alloc + 16;
	      tf->tests
		= xrealloc (tf->tests,
			    tf->num_tests_alloc * sizeof (tf->tests[0]));
	    }
	  input_test *it = &tf->tests[tf->num_tests];
	  it->line = line;
	  it->num_input_cases = 1;
	  it->inputs = xmalloc (sizeof (it->inputs[0]));
	  it->inputs[0] = NULL;
	  it->old_output = NULL;
	  p++;
	  for (size_t j = 0; j < tf->num_args; j++)
	    {
	      char *ep = strchr (p, ' ');
	      if (ep == NULL)
		{
		  ep = strchr (p, '\n');
		  assert (ep != NULL);
		}
	      if (ep == p)
		error_at_line (EXIT_FAILURE, 0, filename, lineno,
			       "empty token in line: '%s'", line);
	      for (char *t = p; t < ep; t++)
		if (isspace ((unsigned char) *t))
		  error_at_line (EXIT_FAILURE, 0, filename, lineno,
				 "whitespace in token in line: '%s'", line);
	      char c = *ep;
	      *ep = 0;
	      handle_input_arg (p, it, j,
				generic_arg_ret_type (tf->arg_types[j]),
				tf->exact_args, filename, lineno);
	      *ep = c;
	      p = ep + 1;
	    }
	  it->num_flags = num_tokens - 1 - tf->num_args;
	  it->flags = xmalloc (it->num_flags * sizeof (it->flags[0]));
	  for (size_t j = 0; j < it->num_flags; j++)
	    {
	      char *ep = strchr (p, ' ');
	      if (ep == NULL)
		{
		  ep = strchr (p, '\n');
		  assert (ep != NULL);
		}
	      if (ep == p)
		error_at_line (EXIT_FAILURE, 0, filename, lineno,
			       "empty token in line: '%s'", line);
	      for (char *t = p; t < ep; t++)
		if (isspace ((unsigned char) *t))
		  error_at_line (EXIT_FAILURE, 0, filename, lineno,
				 "whitespace in token in line: '%s'", line);
	      char c = *ep;
	      *ep = 0;
	      handle_input_flag (p, &it->flags[j], filename, lineno);
	      *ep = c;
	      p = ep + 1;
	    }
	  assert (*p == 0);
	  tf->num_tests++;
	  return;
	}
    }
  error_at_line (EXIT_FAILURE, 0, filename, lineno,
		 "unknown function in line: '%s'", line);
}

/* Read in the test input data from FILENAME.  */

static void
read_input (const char *filename)
{
  FILE *fp = fopen (filename, "r");
  if (fp == NULL)
    error (EXIT_FAILURE, errno, "open '%s'", filename);
  unsigned int lineno = 0;
  for (;;)
    {
      size_t size = 0;
      char *line = NULL;
      ssize_t ret = getline (&line, &size, fp);
      if (ret == -1)
	break;
      lineno++;
      if (line[0] == '#' || line[0] == '\n')
	continue;
      add_test (line, filename, lineno);
    }
  if (ferror (fp))
    error (EXIT_FAILURE, errno, "read from '%s'", filename);
  if (fclose (fp) != 0)
    error (EXIT_FAILURE, errno, "close '%s'", filename);
}

/* Calculate the generic results (round-to-zero with sticky bit) for
   the function described by CALC, with inputs INPUTS, if MODE is
   rm_towardzero; for other modes, calculate results in that mode,
   which must be exact zero results.  */

static void
calc_generic_results (generic_value *outputs, generic_value *inputs,
		      const func_calc_desc *calc, rounding_mode mode)
{
  bool inexact;
  int mpc_ternary;
  mpc_t ci1, ci2, co;
  mpfr_rnd_t mode_mpfr = rounding_modes[mode].mpfr_mode;
  mpc_rnd_t mode_mpc = rounding_modes[mode].mpc_mode;

  switch (calc->method)
    {
    case mpfr_f_f:
      assert (inputs[0].type == gtype_fp);
      outputs[0].type = gtype_fp;
      mpfr_init (outputs[0].value.f);
      inexact = calc->func.mpfr_f_f (outputs[0].value.f, inputs[0].value.f,
				     mode_mpfr);
      if (mode != rm_towardzero)
	assert (!inexact && mpfr_zero_p (outputs[0].value.f));
      adjust_real (outputs[0].value.f, inexact);
      break;

    case mpfr_ff_f:
      assert (inputs[0].type == gtype_fp);
      assert (inputs[1].type == gtype_fp);
      outputs[0].type = gtype_fp;
      mpfr_init (outputs[0].value.f);
      inexact = calc->func.mpfr_ff_f (outputs[0].value.f, inputs[0].value.f,
				      inputs[1].value.f, mode_mpfr);
      if (mode != rm_towardzero)
	assert (!inexact && mpfr_zero_p (outputs[0].value.f));
      adjust_real (outputs[0].value.f, inexact);
      break;

    case mpfr_fff_f:
      assert (inputs[0].type == gtype_fp);
      assert (inputs[1].type == gtype_fp);
      assert (inputs[2].type == gtype_fp);
      outputs[0].type = gtype_fp;
      mpfr_init (outputs[0].value.f);
      inexact = calc->func.mpfr_fff_f (outputs[0].value.f, inputs[0].value.f,
				       inputs[1].value.f, inputs[2].value.f,
				       mode_mpfr);
      if (mode != rm_towardzero)
	assert (!inexact && mpfr_zero_p (outputs[0].value.f));
      adjust_real (outputs[0].value.f, inexact);
      break;

    case mpfr_f_f1:
      assert (inputs[0].type == gtype_fp);
      outputs[0].type = gtype_fp;
      outputs[1].type = gtype_int;
      mpfr_init (outputs[0].value.f);
      int i = 0;
      inexact = calc->func.mpfr_f_f1 (outputs[0].value.f, &i,
				      inputs[0].value.f, mode_mpfr);
      if (mode != rm_towardzero)
	assert (!inexact && mpfr_zero_p (outputs[0].value.f));
      adjust_real (outputs[0].value.f, inexact);
      mpz_init_set_si (outputs[1].value.i, i);
      break;

    case mpfr_if_f:
      assert (inputs[0].type == gtype_int);
      assert (inputs[1].type == gtype_fp);
      outputs[0].type = gtype_fp;
      mpfr_init (outputs[0].value.f);
      assert (mpz_fits_slong_p (inputs[0].value.i));
      long l = mpz_get_si (inputs[0].value.i);
      inexact = calc->func.mpfr_if_f (outputs[0].value.f, l,
				      inputs[1].value.f, mode_mpfr);
      if (mode != rm_towardzero)
	assert (!inexact && mpfr_zero_p (outputs[0].value.f));
      adjust_real (outputs[0].value.f, inexact);
      break;

    case mpfr_f_11:
      assert (inputs[0].type == gtype_fp);
      outputs[0].type = gtype_fp;
      mpfr_init (outputs[0].value.f);
      outputs[1].type = gtype_fp;
      mpfr_init (outputs[1].value.f);
      int comb_ternary = calc->func.mpfr_f_11 (outputs[0].value.f,
					       outputs[1].value.f,
					       inputs[0].value.f,
					       mode_mpfr);
      if (mode != rm_towardzero)
	assert (((comb_ternary & 0x3) == 0
		 && mpfr_zero_p (outputs[0].value.f))
		|| ((comb_ternary & 0xc) == 0
		    && mpfr_zero_p (outputs[1].value.f)));
      adjust_real (outputs[0].value.f, (comb_ternary & 0x3) != 0);
      adjust_real (outputs[1].value.f, (comb_ternary & 0xc) != 0);
      break;

    case mpc_c_f:
      assert (inputs[0].type == gtype_fp);
      assert (inputs[1].type == gtype_fp);
      outputs[0].type = gtype_fp;
      mpfr_init (outputs[0].value.f);
      mpc_init2 (ci1, internal_precision);
      assert_exact (mpc_set_fr_fr (ci1, inputs[0].value.f, inputs[1].value.f,
				   MPC_RNDNN));
      inexact = calc->func.mpc_c_f (outputs[0].value.f, ci1, mode_mpfr);
      if (mode != rm_towardzero)
	assert (!inexact && mpfr_zero_p (outputs[0].value.f));
      adjust_real (outputs[0].value.f, inexact);
      mpc_clear (ci1);
      break;

    case mpc_c_c:
      assert (inputs[0].type == gtype_fp);
      assert (inputs[1].type == gtype_fp);
      outputs[0].type = gtype_fp;
      mpfr_init (outputs[0].value.f);
      outputs[1].type = gtype_fp;
      mpfr_init (outputs[1].value.f);
      mpc_init2 (ci1, internal_precision);
      mpc_init2 (co, internal_precision);
      assert_exact (mpc_set_fr_fr (ci1, inputs[0].value.f, inputs[1].value.f,
				   MPC_RNDNN));
      mpc_ternary = calc->func.mpc_c_c (co, ci1, mode_mpc);
      if (mode != rm_towardzero)
	assert ((!MPC_INEX_RE (mpc_ternary)
		 && mpfr_zero_p (mpc_realref (co)))
		|| (!MPC_INEX_IM (mpc_ternary)
		    && mpfr_zero_p (mpc_imagref (co))));
      assert_exact (mpfr_set (outputs[0].value.f, mpc_realref (co),
			      MPFR_RNDN));
      assert_exact (mpfr_set (outputs[1].value.f, mpc_imagref (co),
			      MPFR_RNDN));
      adjust_real (outputs[0].value.f, MPC_INEX_RE (mpc_ternary));
      adjust_real (outputs[1].value.f, MPC_INEX_IM (mpc_ternary));
      mpc_clear (ci1);
      mpc_clear (co);
      break;

    case mpc_cc_c:
      assert (inputs[0].type == gtype_fp);
      assert (inputs[1].type == gtype_fp);
      assert (inputs[2].type == gtype_fp);
      assert (inputs[3].type == gtype_fp);
      outputs[0].type = gtype_fp;
      mpfr_init (outputs[0].value.f);
      outputs[1].type = gtype_fp;
      mpfr_init (outputs[1].value.f);
      mpc_init2 (ci1, internal_precision);
      mpc_init2 (ci2, internal_precision);
      mpc_init2 (co, internal_precision);
      assert_exact (mpc_set_fr_fr (ci1, inputs[0].value.f, inputs[1].value.f,
				   MPC_RNDNN));
      assert_exact (mpc_set_fr_fr (ci2, inputs[2].value.f, inputs[3].value.f,
				   MPC_RNDNN));
      mpc_ternary = calc->func.mpc_cc_c (co, ci1, ci2, mode_mpc);
      if (mode != rm_towardzero)
	assert ((!MPC_INEX_RE (mpc_ternary)
		 && mpfr_zero_p (mpc_realref (co)))
		|| (!MPC_INEX_IM (mpc_ternary)
		    && mpfr_zero_p (mpc_imagref (co))));
      assert_exact (mpfr_set (outputs[0].value.f, mpc_realref (co),
			      MPFR_RNDN));
      assert_exact (mpfr_set (outputs[1].value.f, mpc_imagref (co),
			      MPFR_RNDN));
      adjust_real (outputs[0].value.f, MPC_INEX_RE (mpc_ternary));
      adjust_real (outputs[1].value.f, MPC_INEX_IM (mpc_ternary));
      mpc_clear (ci1);
      mpc_clear (ci2);
      mpc_clear (co);
      break;

    default:
      abort ();
    }
}

/* Return the number of bits for integer type TYPE, where "long" has
   LONG_BITS bits (32 or 64).  */

static int
int_type_bits (arg_ret_type type, int long_bits)
{
  assert (long_bits == 32 || long_bits == 64);
  switch (type)
    {
    case type_int:
      return 32;
      break;

    case type_long:
      return long_bits;
      break;

    case type_long_long:
      return 64;
      break;

    default:
      abort ();
    }
}

/* Check whether an integer Z fits a given type TYPE, where "long" has
   LONG_BITS bits (32 or 64).  */

static bool
int_fits_type (mpz_t z, arg_ret_type type, int long_bits)
{
  int bits = int_type_bits (type, long_bits);
  bool ret = true;
  mpz_t t;
  mpz_init (t);
  mpz_ui_pow_ui (t, 2, bits - 1);
  if (mpz_cmp (z, t) >= 0)
    ret = false;
  mpz_neg (t, t);
  if (mpz_cmp (z, t) < 0)
    ret = false;
  mpz_clear (t);
  return ret;
}

/* Print a generic value V to FP (name FILENAME), preceded by a space,
   for type TYPE, LONG_BITS bits per long, printing " IGNORE" instead
   if IGNORE.  */

static void
output_generic_value (FILE *fp, const char *filename, const generic_value *v,
		      bool ignore, arg_ret_type type, int long_bits)
{
  if (ignore)
    {
      if (fputs (" IGNORE", fp) < 0)
	error (EXIT_FAILURE, errno, "write to '%s'", filename);
      return;
    }
  assert (v->type == generic_arg_ret_type (type));
  const char *suffix;
  switch (type)
    {
    case type_fp:
      suffix = "";
      break;

    case type_int:
      suffix = "";
      break;

    case type_long:
      suffix = "L";
      break;

    case type_long_long:
      suffix = "LL";
      break;

    default:
      abort ();
    }
  switch (v->type)
    {
    case gtype_fp:
      if (mpfr_inf_p (v->value.f))
	{
	  if (fputs ((mpfr_signbit (v->value.f)
		      ? " minus_infty" : " plus_infty"), fp) < 0)
	    error (EXIT_FAILURE, errno, "write to '%s'", filename);
	}
      else
	{
	  assert (mpfr_number_p (v->value.f));
	  if (mpfr_fprintf (fp, " %Ra%s", v->value.f, suffix) < 0)
	    error (EXIT_FAILURE, errno, "mpfr_fprintf to '%s'", filename);
	}
      break;

    case gtype_int: ;
      int bits = int_type_bits (type, long_bits);
      mpz_t tmp;
      mpz_init (tmp);
      mpz_ui_pow_ui (tmp, 2, bits - 1);
      mpz_neg (tmp, tmp);
      if (mpz_cmp (v->value.i, tmp) == 0)
	{
	  mpz_add_ui (tmp, tmp, 1);
	  if (mpfr_fprintf (fp, " (%Zd%s-1)", tmp, suffix) < 0)
	    error (EXIT_FAILURE, errno, "mpfr_fprintf to '%s'", filename);
	}
      else
	{
	  if (mpfr_fprintf (fp, " %Zd%s", v->value.i, suffix) < 0)
	    error (EXIT_FAILURE, errno, "mpfr_fprintf to '%s'", filename);
	}
      mpz_clear (tmp);
      break;

    default:
      abort ();
    }
}

/* Generate test output to FP (name FILENAME) for test function TF
   (rounding results to a narrower type if NARROW), input test IT,
   choice of input values INPUTS.  */

static void
output_for_one_input_case (FILE *fp, const char *filename, test_function *tf,
			   bool narrow, input_test *it, generic_value *inputs)
{
  bool long_bits_matters = false;
  bool fits_long32 = true;
  for (size_t i = 0; i < tf->num_args; i++)
    {
      generic_value_type gtype = generic_arg_ret_type (tf->arg_types[i]);
      assert (inputs[i].type == gtype);
      if (gtype == gtype_int)
	{
	  bool fits_64 = int_fits_type (inputs[i].value.i, tf->arg_types[i],
					64);
	  if (!fits_64)
	    return;
	  if (tf->arg_types[i] == type_long
	      && !int_fits_type (inputs[i].value.i, tf->arg_types[i], 32))
	    {
	      long_bits_matters = true;
	      fits_long32 = false;
	    }
	}
    }
  generic_value generic_outputs[MAX_NRET];
  calc_generic_results (generic_outputs, inputs, &tf->calc, rm_towardzero);
  bool ignore_output_long32[MAX_NRET] = { false };
  bool ignore_output_long64[MAX_NRET] = { false };
  for (size_t i = 0; i < tf->num_ret; i++)
    {
      assert (generic_outputs[i].type
	      == generic_arg_ret_type (tf->ret_types[i]));
      switch (generic_outputs[i].type)
	{
	case gtype_fp:
	  if (!mpfr_number_p (generic_outputs[i].value.f))
	    goto out; /* Result is NaN or exact infinity.  */
	  break;

	case gtype_int:
	  ignore_output_long32[i] = !int_fits_type (generic_outputs[i].value.i,
						    tf->ret_types[i], 32);
	  ignore_output_long64[i] = !int_fits_type (generic_outputs[i].value.i,
						    tf->ret_types[i], 64);
	  if (ignore_output_long32[i] != ignore_output_long64[i])
	    long_bits_matters = true;
	  break;

	default:
	  abort ();
	}
    }
  /* Iterate over relevant sizes of long and floating-point formats.  */
  for (int long_bits = 32; long_bits <= 64; long_bits += 32)
    {
      if (long_bits == 32 && !fits_long32)
	continue;
      if (long_bits == 64 && !long_bits_matters)
	continue;
      const char *long_cond;
      if (long_bits_matters)
	long_cond = (long_bits == 32 ? ":long32" : ":long64");
      else
	long_cond = "";
      bool *ignore_output = (long_bits == 32
			     ? ignore_output_long32
			     : ignore_output_long64);
      for (fp_format f = fp_first_format; f < fp_num_formats; f++)
	{
	  bool fits = true;
	  mpfr_t res[rm_num_modes];
	  unsigned int exc_before[rm_num_modes];
	  unsigned int exc_after[rm_num_modes];
	  bool have_fp_arg = false;
	  int max_exp = 0;
	  int num_ones = 0;
	  int min_exp = 0;
	  int max_prec = 0;
	  for (size_t i = 0; i < tf->num_args; i++)
	    {
	      if (inputs[i].type == gtype_fp)
		{
		  if (narrow)
		    {
		      if (mpfr_zero_p (inputs[i].value.f))
			continue;
		      assert (mpfr_regular_p (inputs[i].value.f));
		      int this_exp, this_num_ones, this_min_exp, this_prec;
		      mpz_t tmp;
		      mpz_init (tmp);
		      mpfr_exp_t e = mpfr_get_z_2exp (tmp, inputs[i].value.f);
		      if (mpz_sgn (tmp) < 0)
			mpz_neg (tmp, tmp);
		      size_t bits = mpz_sizeinbase (tmp, 2);
		      mp_bitcnt_t tz = mpz_scan1 (tmp, 0);
		      this_min_exp = e + tz;
		      this_prec = bits - tz;
		      assert (this_prec > 0);
		      this_exp = this_min_exp + this_prec - 1;
		      assert (this_exp
			      == mpfr_get_exp (inputs[i].value.f) - 1);
		      this_num_ones = 1;
		      while ((size_t) this_num_ones < bits
			     && mpz_tstbit (tmp, bits - 1 - this_num_ones))
			this_num_ones++;
		      mpz_clear (tmp);
		      if (have_fp_arg)
			{
			  if (this_exp > max_exp
			      || (this_exp == max_exp
				  && this_num_ones > num_ones))
			    {
			      max_exp = this_exp;
			      num_ones = this_num_ones;
			    }
			  if (this_min_exp < min_exp)
			    min_exp = this_min_exp;
			  if (this_prec > max_prec)
			    max_prec = this_prec;
			}
		      else
			{
			  max_exp = this_exp;
			  num_ones = this_num_ones;
			  min_exp = this_min_exp;
			  max_prec = this_prec;
			}
		      have_fp_arg = true;
		    }
		  else
		    {
		      round_real (res, exc_before, exc_after,
				  inputs[i].value.f, f);
		      if (!mpfr_equal_p (res[rm_tonearest], inputs[i].value.f))
			fits = false;
		      for (rounding_mode m = rm_first_mode;
			   m < rm_num_modes;
			   m++)
			mpfr_clear (res[m]);
		      if (!fits)
			break;
		    }
		}
	    }
	  if (!fits)
	    continue;
	  /* The inputs fit this type if required to do so, so compute
	     the ideal outputs and exceptions.  */
	  mpfr_t all_res[MAX_NRET][rm_num_modes];
	  unsigned int all_exc_before[MAX_NRET][rm_num_modes];
	  unsigned int all_exc_after[MAX_NRET][rm_num_modes];
	  unsigned int merged_exc_before[rm_num_modes] = { 0 };
	  unsigned int merged_exc_after[rm_num_modes] = { 0 };
	  /* For functions not exactly determined, track whether
	     underflow is required (some result is inexact, and
	     magnitude does not exceed the greatest magnitude
	     subnormal), and permitted (not an exact zero, and
	     magnitude does not exceed the least magnitude
	     normal).  */
	  bool must_underflow = false;
	  bool may_underflow = false;
	  for (size_t i = 0; i < tf->num_ret; i++)
	    {
	      switch (generic_outputs[i].type)
		{
		case gtype_fp:
		  round_real (all_res[i], all_exc_before[i], all_exc_after[i],
			      generic_outputs[i].value.f, f);
		  for (rounding_mode m = rm_first_mode; m < rm_num_modes; m++)
		    {
		      merged_exc_before[m] |= all_exc_before[i][m];
		      merged_exc_after[m] |= all_exc_after[i][m];
		      if (!tf->exact)
			{
			  must_underflow
			    |= ((all_exc_before[i][m]
				 & (1U << exc_inexact)) != 0
				&& (mpfr_cmpabs (generic_outputs[i].value.f,
						fp_formats[f].subnorm_max)
				    <= 0));
			  may_underflow
			    |= (!mpfr_zero_p (generic_outputs[i].value.f)
				&& (mpfr_cmpabs (generic_outputs[i].value.f,
						 fp_formats[f].min_plus_half)
				    <= 0));
			}
		      /* If the result is an exact zero, the sign may
			 depend on the rounding mode, so recompute it
			 directly in that mode.  */
		      if (mpfr_zero_p (all_res[i][m])
			  && (all_exc_before[i][m] & (1U << exc_inexact)) == 0)
			{
			  generic_value outputs_rm[MAX_NRET];
			  calc_generic_results (outputs_rm, inputs,
						&tf->calc, m);
			  assert_exact (mpfr_set (all_res[i][m],
						  outputs_rm[i].value.f,
						  MPFR_RNDN));
			  for (size_t j = 0; j < tf->num_ret; j++)
			    generic_value_free (&outputs_rm[j]);
			}
		    }
		  break;

		case gtype_int:
		  if (ignore_output[i])
		    for (rounding_mode m = rm_first_mode;
			 m < rm_num_modes;
			 m++)
		      {
			merged_exc_before[m] |= 1U << exc_invalid;
			merged_exc_after[m] |= 1U << exc_invalid;
		      }
		  break;

		default:
		  abort ();
		}
	    }
	  assert (may_underflow || !must_underflow);
	  for (rounding_mode m = rm_first_mode; m < rm_num_modes; m++)
	    {
	      bool before_after_matters
		= tf->exact && merged_exc_before[m] != merged_exc_after[m];
	      if (before_after_matters)
		{
		  assert ((merged_exc_before[m] ^ merged_exc_after[m])
			  == (1U << exc_underflow));
		  assert ((merged_exc_before[m] & (1U << exc_underflow)) != 0);
		}
	      unsigned int merged_exc = merged_exc_before[m];
	      if (narrow)
		{
		  if (fprintf (fp, "= %s %s %s%s:arg_fmt(%d,%d,%d,%d)",
			       tf->name, rounding_modes[m].name,
			       fp_formats[f].name, long_cond, max_exp,
			       num_ones, min_exp, max_prec) < 0)
		    error (EXIT_FAILURE, errno, "write to '%s'", filename);
		}
	      else
		{
		  if (fprintf (fp, "= %s %s %s%s", tf->name,
			       rounding_modes[m].name, fp_formats[f].name,
			       long_cond) < 0)
		    error (EXIT_FAILURE, errno, "write to '%s'", filename);
		}
	      /* Print inputs.  */
	      for (size_t i = 0; i < tf->num_args; i++)
		output_generic_value (fp, filename, &inputs[i], false,
				      tf->arg_types[i], long_bits);
	      if (fputs (" :", fp) < 0)
		error (EXIT_FAILURE, errno, "write to '%s'", filename);
	      /* Print outputs.  */
	      bool must_erange = false;
	      bool some_underflow_zero = false;
	      for (size_t i = 0; i < tf->num_ret; i++)
		{
		  generic_value g;
		  g.type = generic_outputs[i].type;
		  switch (g.type)
		    {
		    case gtype_fp:
		      if (mpfr_inf_p (all_res[i][m])
			  && (all_exc_before[i][m]
			      & (1U << exc_overflow)) != 0)
			must_erange = true;
		      if (mpfr_zero_p (all_res[i][m])
			  && (tf->exact
			      || mpfr_zero_p (all_res[i][rm_tonearest]))
			  && (all_exc_before[i][m]
			      & (1U << exc_underflow)) != 0)
			must_erange = true;
		      if (mpfr_zero_p (all_res[i][rm_towardzero])
			  && (all_exc_before[i][m]
			      & (1U << exc_underflow)) != 0)
			some_underflow_zero = true;
		      mpfr_init2 (g.value.f, fp_formats[f].mant_dig);
		      assert_exact (mpfr_set (g.value.f, all_res[i][m],
					      MPFR_RNDN));
		      break;

		    case gtype_int:
		      mpz_init (g.value.i);
		      mpz_set (g.value.i, generic_outputs[i].value.i);
		      break;

		    default:
		      abort ();
		    }
		  output_generic_value (fp, filename, &g, ignore_output[i],
					tf->ret_types[i], long_bits);
		  generic_value_free (&g);
		}
	      if (fputs (" :", fp) < 0)
		error (EXIT_FAILURE, errno, "write to '%s'", filename);
	      /* Print miscellaneous flags (passed through from
		 input).  */
	      for (size_t i = 0; i < it->num_flags; i++)
		switch (it->flags[i].type)
		  {
		  case flag_ignore_zero_inf_sign:
		  case flag_xfail:
		    if (fprintf (fp, " %s%s",
				 input_flags[it->flags[i].type],
				 (it->flags[i].cond
				  ? it->flags[i].cond
				  : "")) < 0)
		      error (EXIT_FAILURE, errno, "write to '%s'",
			     filename);
		    break;
		  case flag_xfail_rounding:
		    if (m != rm_tonearest)
		      if (fprintf (fp, " xfail%s",
				   (it->flags[i].cond
				    ? it->flags[i].cond
				    : "")) < 0)
			error (EXIT_FAILURE, errno, "write to '%s'",
			       filename);
		    break;
		  default:
		    break;
		  }
	      /* For the ibm128 format, expect incorrect overflowing
		 results in rounding modes other than to nearest;
		 likewise incorrect results where the result may
		 underflow to 0.  */
	      if (f == fp_ldbl_128ibm
		  && m != rm_tonearest
		  && (some_underflow_zero
		      || (merged_exc_before[m] & (1U << exc_overflow)) != 0))
		if (fputs (" xfail:ibm128-libgcc", fp) < 0)
		  error (EXIT_FAILURE, errno, "write to '%s'", filename);
	      /* Print exception flags and compute errno
		 expectations where not already computed.  */
	      bool may_edom = false;
	      bool must_edom = false;
	      bool may_erange = must_erange || may_underflow;
	      for (fp_exception e = exc_first_exception;
		   e < exc_num_exceptions;
		   e++)
		{
		  bool expect_e = (merged_exc & (1U << e)) != 0;
		  bool e_optional = false;
		  switch (e)
		    {
		    case exc_divbyzero:
		      if (expect_e)
			may_erange = must_erange = true;
		      break;

		    case exc_inexact:
		      if (!tf->exact)
			e_optional = true;
		      break;

		    case exc_invalid:
		      if (expect_e)
			may_edom = must_edom = true;
		      break;

		    case exc_overflow:
		      if (expect_e)
			may_erange = true;
		      break;

		    case exc_underflow:
		      if (expect_e)
			may_erange = true;
		      if (must_underflow)
			assert (expect_e);
		      if (may_underflow && !must_underflow)
			e_optional = true;
		      break;

		    default:
		      abort ();
		    }
		  if (e_optional)
		    {
		      assert (!before_after_matters);
		      if (fprintf (fp, " %s-ok", exceptions[e]) < 0)
			error (EXIT_FAILURE, errno, "write to '%s'",
			       filename);
		    }
		  else
		    {
		      if (expect_e)
			if (fprintf (fp, " %s", exceptions[e]) < 0)
			  error (EXIT_FAILURE, errno, "write to '%s'",
				 filename);
		      if (before_after_matters && e == exc_underflow)
			if (fputs (":before-rounding", fp) < 0)
			  error (EXIT_FAILURE, errno, "write to '%s'",
				 filename);
		      for (int after = 0; after <= 1; after++)
			{
			  bool expect_e_here = expect_e;
			  if (after == 1 && (!before_after_matters
					     || e != exc_underflow))
			    continue;
			  const char *after_cond;
			  if (before_after_matters && e == exc_underflow)
			    {
			      after_cond = (after
					    ? ":after-rounding"
					    : ":before-rounding");
			      expect_e_here = !after;
			    }
			  else
			    after_cond = "";
			  input_flag_type okflag;
			  okflag = (expect_e_here
				    ? flag_missing_first
				    : flag_spurious_first) + e;
			  for (size_t i = 0; i < it->num_flags; i++)
			    if (it->flags[i].type == okflag)
			      if (fprintf (fp, " %s-ok%s%s",
					   exceptions[e],
					   (it->flags[i].cond
					    ? it->flags[i].cond
					    : ""), after_cond) < 0)
				error (EXIT_FAILURE, errno, "write to '%s'",
				       filename);
			}
		    }
		}
	      /* Print errno expectations.  */
	      if (tf->complex_fn)
		{
		  must_edom = false;
		  must_erange = false;
		}
	      if (may_edom && !must_edom)
		{
		  if (fputs (" errno-edom-ok", fp) < 0)
		    error (EXIT_FAILURE, errno, "write to '%s'",
			   filename);
		}
	      else
		{
		  if (must_edom)
		    if (fputs (" errno-edom", fp) < 0)
		      error (EXIT_FAILURE, errno, "write to '%s'",
			     filename);
		  input_flag_type okflag = (must_edom
					    ? flag_missing_errno
					    : flag_spurious_errno);
		  for (size_t i = 0; i < it->num_flags; i++)
		    if (it->flags[i].type == okflag)
		      if (fprintf (fp, " errno-edom-ok%s",
				   (it->flags[i].cond
				    ? it->flags[i].cond
				    : "")) < 0)
			error (EXIT_FAILURE, errno, "write to '%s'",
			       filename);
		}
	      if (before_after_matters)
		assert (may_erange && !must_erange);
	      if (may_erange && !must_erange)
		{
		  if (fprintf (fp, " errno-erange-ok%s",
			       (before_after_matters
				? ":before-rounding"
				: "")) < 0)
		    error (EXIT_FAILURE, errno, "write to '%s'",
			   filename);
		}
	      if (before_after_matters || !(may_erange && !must_erange))
		{
		  if (must_erange)
		    if (fputs (" errno-erange", fp) < 0)
		      error (EXIT_FAILURE, errno, "write to '%s'",
			     filename);
		  input_flag_type okflag = (must_erange
					    ? flag_missing_errno
					    : flag_spurious_errno);
		  for (size_t i = 0; i < it->num_flags; i++)
		    if (it->flags[i].type == okflag)
		      if (fprintf (fp, " errno-erange-ok%s%s",
				   (it->flags[i].cond
				    ? it->flags[i].cond
				    : ""),
				   (before_after_matters
				    ? ":after-rounding"
				    : "")) < 0)
			error (EXIT_FAILURE, errno, "write to '%s'",
			       filename);
		}
	      if (putc ('\n', fp) < 0)
		error (EXIT_FAILURE, errno, "write to '%s'", filename);
	    }
	  for (size_t i = 0; i < tf->num_ret; i++)
	    {
	      if (generic_outputs[i].type == gtype_fp)
		for (rounding_mode m = rm_first_mode; m < rm_num_modes; m++)
		  mpfr_clear (all_res[i][m]);
	    }
	}
    }
 out:
  for (size_t i = 0; i < tf->num_ret; i++)
    generic_value_free (&generic_outputs[i]);
}

/* Generate test output data for FUNCTION to FILENAME.  The function
   is interpreted as rounding its results to a narrower type if
   NARROW.  */

static void
generate_output (const char *function, bool narrow, const char *filename)
{
  FILE *fp = fopen (filename, "w");
  if (fp == NULL)
    error (EXIT_FAILURE, errno, "open '%s'", filename);
  for (size_t i = 0; i < ARRAY_SIZE (test_functions); i++)
    {
      test_function *tf = &test_functions[i];
      if (strcmp (tf->name, function) != 0)
	continue;
      for (size_t j = 0; j < tf->num_tests; j++)
	{
	  input_test *it = &tf->tests[j];
	  if (fputs (it->line, fp) < 0)
	    error (EXIT_FAILURE, errno, "write to '%s'", filename);
	  for (size_t k = 0; k < it->num_input_cases; k++)
	    output_for_one_input_case (fp, filename, tf, narrow,
				       it, it->inputs[k]);
	}
    }
  if (fclose (fp) != 0)
    error (EXIT_FAILURE, errno, "close '%s'", filename);
}

int
main (int argc, char **argv)
{
  if (argc != 4
      && !(argc == 5 && strcmp (argv[1], "--narrow") == 0))
    error (EXIT_FAILURE, 0,
	   "usage: gen-auto-libm-tests [--narrow] <input> <func> <output>");
  bool narrow;
  const char *input_filename = argv[1];
  const char *function = argv[2];
  const char *output_filename = argv[3];
  if (argc == 4)
    {
      narrow = false;
      input_filename = argv[1];
      function = argv[2];
      output_filename = argv[3];
    }
  else
    {
      narrow = true;
      input_filename = argv[2];
      function = argv[3];
      output_filename = argv[4];
    }
  init_fp_formats ();
  read_input (input_filename);
  generate_output (function, narrow, output_filename);
  exit (EXIT_SUCCESS);
}
