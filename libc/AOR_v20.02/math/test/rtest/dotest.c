/*
 * dotest.c - actually generate mathlib test cases
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <limits.h>

#include "semi.h"
#include "intern.h"
#include "random.h"

#define MPFR_PREC 96 /* good enough for float or double + a few extra bits */

extern int lib_fo, lib_no_arith, ntests;

/*
 * Prototypes.
 */
static void cases_biased(uint32 *, uint32, uint32);
static void cases_biased_positive(uint32 *, uint32, uint32);
static void cases_biased_float(uint32 *, uint32, uint32);
static void cases_uniform(uint32 *, uint32, uint32);
static void cases_uniform_positive(uint32 *, uint32, uint32);
static void cases_uniform_float(uint32 *, uint32, uint32);
static void cases_uniform_float_positive(uint32 *, uint32, uint32);
static void log_cases(uint32 *, uint32, uint32);
static void log_cases_float(uint32 *, uint32, uint32);
static void log1p_cases(uint32 *, uint32, uint32);
static void log1p_cases_float(uint32 *, uint32, uint32);
static void minmax_cases(uint32 *, uint32, uint32);
static void minmax_cases_float(uint32 *, uint32, uint32);
static void atan2_cases(uint32 *, uint32, uint32);
static void atan2_cases_float(uint32 *, uint32, uint32);
static void pow_cases(uint32 *, uint32, uint32);
static void pow_cases_float(uint32 *, uint32, uint32);
static void rred_cases(uint32 *, uint32, uint32);
static void rred_cases_float(uint32 *, uint32, uint32);
static void cases_semi1(uint32 *, uint32, uint32);
static void cases_semi1_float(uint32 *, uint32, uint32);
static void cases_semi2(uint32 *, uint32, uint32);
static void cases_semi2_float(uint32 *, uint32, uint32);
static void cases_ldexp(uint32 *, uint32, uint32);
static void cases_ldexp_float(uint32 *, uint32, uint32);

static void complex_cases_uniform(uint32 *, uint32, uint32);
static void complex_cases_uniform_float(uint32 *, uint32, uint32);
static void complex_cases_biased(uint32 *, uint32, uint32);
static void complex_cases_biased_float(uint32 *, uint32, uint32);
static void complex_log_cases(uint32 *, uint32, uint32);
static void complex_log_cases_float(uint32 *, uint32, uint32);
static void complex_pow_cases(uint32 *, uint32, uint32);
static void complex_pow_cases_float(uint32 *, uint32, uint32);
static void complex_arithmetic_cases(uint32 *, uint32, uint32);
static void complex_arithmetic_cases_float(uint32 *, uint32, uint32);

static uint32 doubletop(int x, int scale);
static uint32 floatval(int x, int scale);

/*
 * Convert back and forth between IEEE bit patterns and the
 * mpfr_t/mpc_t types.
 */
static void set_mpfr_d(mpfr_t x, uint32 h, uint32 l)
{
    uint64_t hl = ((uint64_t)h << 32) | l;
    uint32 exp = (hl >> 52) & 0x7ff;
    int64_t mantissa = hl & (((uint64_t)1 << 52) - 1);
    int sign = (hl >> 63) ? -1 : +1;
    if (exp == 0x7ff) {
        if (mantissa == 0)
            mpfr_set_inf(x, sign);
        else
            mpfr_set_nan(x);
    } else if (exp == 0 && mantissa == 0) {
        mpfr_set_ui(x, 0, GMP_RNDN);
        mpfr_setsign(x, x, sign < 0, GMP_RNDN);
    } else {
        if (exp != 0)
            mantissa |= ((uint64_t)1 << 52);
        else
            exp++;
        mpfr_set_sj_2exp(x, mantissa * sign, (int)exp - 0x3ff - 52, GMP_RNDN);
    }
}
static void set_mpfr_f(mpfr_t x, uint32 f)
{
    uint32 exp = (f >> 23) & 0xff;
    int32 mantissa = f & ((1 << 23) - 1);
    int sign = (f >> 31) ? -1 : +1;
    if (exp == 0xff) {
        if (mantissa == 0)
            mpfr_set_inf(x, sign);
        else
            mpfr_set_nan(x);
    } else if (exp == 0 && mantissa == 0) {
        mpfr_set_ui(x, 0, GMP_RNDN);
        mpfr_setsign(x, x, sign < 0, GMP_RNDN);
    } else {
        if (exp != 0)
            mantissa |= (1 << 23);
        else
            exp++;
        mpfr_set_sj_2exp(x, mantissa * sign, (int)exp - 0x7f - 23, GMP_RNDN);
    }
}
static void set_mpc_d(mpc_t z, uint32 rh, uint32 rl, uint32 ih, uint32 il)
{
    mpfr_t x, y;
    mpfr_init2(x, MPFR_PREC);
    mpfr_init2(y, MPFR_PREC);
    set_mpfr_d(x, rh, rl);
    set_mpfr_d(y, ih, il);
    mpc_set_fr_fr(z, x, y, MPC_RNDNN);
    mpfr_clear(x);
    mpfr_clear(y);
}
static void set_mpc_f(mpc_t z, uint32 r, uint32 i)
{
    mpfr_t x, y;
    mpfr_init2(x, MPFR_PREC);
    mpfr_init2(y, MPFR_PREC);
    set_mpfr_f(x, r);
    set_mpfr_f(y, i);
    mpc_set_fr_fr(z, x, y, MPC_RNDNN);
    mpfr_clear(x);
    mpfr_clear(y);
}
static void get_mpfr_d(const mpfr_t x, uint32 *h, uint32 *l, uint32 *extra)
{
    uint32_t sign, expfield, mantfield;
    mpfr_t significand;
    int exp;

    if (mpfr_nan_p(x)) {
        *h = 0x7ff80000;
        *l = 0;
        *extra = 0;
        return;
    }

    sign = mpfr_signbit(x) ? 0x80000000U : 0;

    if (mpfr_inf_p(x)) {
        *h = 0x7ff00000 | sign;
        *l = 0;
        *extra = 0;
        return;
    }

    if (mpfr_zero_p(x)) {
        *h = 0x00000000 | sign;
        *l = 0;
        *extra = 0;
        return;
    }

    mpfr_init2(significand, MPFR_PREC);
    mpfr_set(significand, x, GMP_RNDN);
    exp = mpfr_get_exp(significand);
    mpfr_set_exp(significand, 0);

    /* Now significand is in [1/2,1), and significand * 2^exp == x.
     * So the IEEE exponent corresponding to exp==0 is 0x3fe. */
    if (exp > 0x400) {
        /* overflow to infinity anyway */
        *h = 0x7ff00000 | sign;
        *l = 0;
        *extra = 0;
        mpfr_clear(significand);
        return;
    }

    if (exp <= -0x3fe || mpfr_zero_p(x))
        exp = -0x3fd;       /* denormalise */
    expfield = exp + 0x3fd; /* offset to cancel leading mantissa bit */

    mpfr_div_2si(significand, x, exp - 21, GMP_RNDN);
    mpfr_abs(significand, significand, GMP_RNDN);
    mantfield = mpfr_get_ui(significand, GMP_RNDZ);
    *h = sign + ((uint64_t)expfield << 20) + mantfield;
    mpfr_sub_ui(significand, significand, mantfield, GMP_RNDN);
    mpfr_mul_2ui(significand, significand, 32, GMP_RNDN);
    mantfield = mpfr_get_ui(significand, GMP_RNDZ);
    *l = mantfield;
    mpfr_sub_ui(significand, significand, mantfield, GMP_RNDN);
    mpfr_mul_2ui(significand, significand, 32, GMP_RNDN);
    mantfield = mpfr_get_ui(significand, GMP_RNDZ);
    *extra = mantfield;

    mpfr_clear(significand);
}
static void get_mpfr_f(const mpfr_t x, uint32 *f, uint32 *extra)
{
    uint32_t sign, expfield, mantfield;
    mpfr_t significand;
    int exp;

    if (mpfr_nan_p(x)) {
        *f = 0x7fc00000;
        *extra = 0;
        return;
    }

    sign = mpfr_signbit(x) ? 0x80000000U : 0;

    if (mpfr_inf_p(x)) {
        *f = 0x7f800000 | sign;
        *extra = 0;
        return;
    }

    if (mpfr_zero_p(x)) {
        *f = 0x00000000 | sign;
        *extra = 0;
        return;
    }

    mpfr_init2(significand, MPFR_PREC);
    mpfr_set(significand, x, GMP_RNDN);
    exp = mpfr_get_exp(significand);
    mpfr_set_exp(significand, 0);

    /* Now significand is in [1/2,1), and significand * 2^exp == x.
     * So the IEEE exponent corresponding to exp==0 is 0x7e. */
    if (exp > 0x80) {
        /* overflow to infinity anyway */
        *f = 0x7f800000 | sign;
        *extra = 0;
        mpfr_clear(significand);
        return;
    }

    if (exp <= -0x7e || mpfr_zero_p(x))
        exp = -0x7d;                   /* denormalise */
    expfield = exp + 0x7d; /* offset to cancel leading mantissa bit */

    mpfr_div_2si(significand, x, exp - 24, GMP_RNDN);
    mpfr_abs(significand, significand, GMP_RNDN);
    mantfield = mpfr_get_ui(significand, GMP_RNDZ);
    *f = sign + ((uint64_t)expfield << 23) + mantfield;
    mpfr_sub_ui(significand, significand, mantfield, GMP_RNDN);
    mpfr_mul_2ui(significand, significand, 32, GMP_RNDN);
    mantfield = mpfr_get_ui(significand, GMP_RNDZ);
    *extra = mantfield;

    mpfr_clear(significand);
}
static void get_mpc_d(const mpc_t z,
                      uint32 *rh, uint32 *rl, uint32 *rextra,
                      uint32 *ih, uint32 *il, uint32 *iextra)
{
    mpfr_t x, y;
    mpfr_init2(x, MPFR_PREC);
    mpfr_init2(y, MPFR_PREC);
    mpc_real(x, z, GMP_RNDN);
    mpc_imag(y, z, GMP_RNDN);
    get_mpfr_d(x, rh, rl, rextra);
    get_mpfr_d(y, ih, il, iextra);
    mpfr_clear(x);
    mpfr_clear(y);
}
static void get_mpc_f(const mpc_t z,
                      uint32 *r, uint32 *rextra,
                      uint32 *i, uint32 *iextra)
{
    mpfr_t x, y;
    mpfr_init2(x, MPFR_PREC);
    mpfr_init2(y, MPFR_PREC);
    mpc_real(x, z, GMP_RNDN);
    mpc_imag(y, z, GMP_RNDN);
    get_mpfr_f(x, r, rextra);
    get_mpfr_f(y, i, iextra);
    mpfr_clear(x);
    mpfr_clear(y);
}

/*
 * Implementation of mathlib functions that aren't trivially
 * implementable using an existing mpfr or mpc function.
 */
int test_rred(mpfr_t ret, const mpfr_t x, int *quadrant)
{
    mpfr_t halfpi;
    long quo;
    int status;

    /*
     * In the worst case of range reduction, we get an input of size
     * around 2^1024, and must find its remainder mod pi, which means
     * we need 1024 bits of pi at least. Plus, the remainder might
     * happen to come out very very small if we're unlucky. How
     * unlucky can we be? Well, conveniently, I once went through and
     * actually worked that out using Paxson's modular minimisation
     * algorithm, and it turns out that the smallest exponent you can
     * get out of a nontrivial[1] double precision range reduction is
     * 0x3c2, i.e. of the order of 2^-61. So we need 1024 bits of pi
     * to get us down to the units digit, another 61 or so bits (say
     * 64) to get down to the highest set bit of the output, and then
     * some bits to make the actual mantissa big enough.
     *
     *   [1] of course the output of range reduction can have an
     *   arbitrarily small exponent in the trivial case, where the
     *   input is so small that it's the identity function. That
     *   doesn't count.
     */
    mpfr_init2(halfpi, MPFR_PREC + 1024 + 64);
    mpfr_const_pi(halfpi, GMP_RNDN);
    mpfr_div_ui(halfpi, halfpi, 2, GMP_RNDN);

    status = mpfr_remquo(ret, &quo, x, halfpi, GMP_RNDN);
    *quadrant = quo & 3;

    mpfr_clear(halfpi);

    return status;
}
int test_lgamma(mpfr_t ret, const mpfr_t x, mpfr_rnd_t rnd)
{
    /*
     * mpfr_lgamma takes an extra int * parameter to hold the output
     * sign. We don't bother testing that, so this wrapper throws away
     * the sign and hence fits into the same function prototype as all
     * the other real->real mpfr functions.
     *
     * There is also mpfr_lngamma which has no sign output and hence
     * has the right prototype already, but unfortunately it returns
     * NaN in cases where gamma(x) < 0, so it's no use to us.
     */
    int sign;
    return mpfr_lgamma(ret, &sign, x, rnd);
}
int test_cpow(mpc_t ret, const mpc_t x, const mpc_t y, mpc_rnd_t rnd)
{
    /*
     * For complex pow, we must bump up the precision by a huge amount
     * if we want it to get the really difficult cases right. (Not
     * that we expect the library under test to be getting those cases
     * right itself, but we'd at least like the test suite to report
     * them as wrong for the _right reason_.)
     *
     * This works around a bug in mpc_pow(), fixed by r1455 in the MPC
     * svn repository (2014-10-14) and expected to be in any MPC
     * release after 1.0.2 (which was the latest release already made
     * at the time of the fix). So as and when we update to an MPC
     * with the fix in it, we could remove this workaround.
     *
     * For the reasons for choosing this amount of extra precision,
     * see analysis in complex/cpownotes.txt for the rationale for the
     * amount.
     */
    mpc_t xbig, ybig, retbig;
    int status;

    mpc_init2(xbig, 1034 + 53 + 60 + MPFR_PREC);
    mpc_init2(ybig, 1034 + 53 + 60 + MPFR_PREC);
    mpc_init2(retbig, 1034 + 53 + 60 + MPFR_PREC);

    mpc_set(xbig, x, MPC_RNDNN);
    mpc_set(ybig, y, MPC_RNDNN);
    status = mpc_pow(retbig, xbig, ybig, rnd);
    mpc_set(ret, retbig, rnd);

    mpc_clear(xbig);
    mpc_clear(ybig);
    mpc_clear(retbig);

    return status;
}

/*
 * Identify 'hard' values (NaN, Inf, nonzero denormal) for deciding
 * whether microlib will decline to run a test.
 */
#define is_shard(in) ( \
    (((in)[0] & 0x7F800000) == 0x7F800000 || \
     (((in)[0] & 0x7F800000) == 0 && ((in)[0]&0x7FFFFFFF) != 0)))

#define is_dhard(in) ( \
    (((in)[0] & 0x7FF00000) == 0x7FF00000 || \
     (((in)[0] & 0x7FF00000) == 0 && (((in)[0] & 0xFFFFF) | (in)[1]) != 0)))

/*
 * Identify integers.
 */
int is_dinteger(uint32 *in)
{
    uint32 out[3];
    if ((0x7FF00000 & ~in[0]) == 0)
        return 0;                      /* not finite, hence not integer */
    test_ceil(in, out);
    return in[0] == out[0] && in[1] == out[1];
}
int is_sinteger(uint32 *in)
{
    uint32 out[3];
    if ((0x7F800000 & ~in[0]) == 0)
        return 0;                      /* not finite, hence not integer */
    test_ceilf(in, out);
    return in[0] == out[0];
}

/*
 * Identify signalling NaNs.
 */
int is_dsnan(const uint32 *in)
{
    if ((in[0] & 0x7FF00000) != 0x7FF00000)
        return 0;                      /* not the inf/nan exponent */
    if ((in[0] << 12) == 0 && in[1] == 0)
        return 0;                      /* inf */
    if (in[0] & 0x00080000)
        return 0;                      /* qnan */
    return 1;
}
int is_ssnan(const uint32 *in)
{
    if ((in[0] & 0x7F800000) != 0x7F800000)
        return 0;                      /* not the inf/nan exponent */
    if ((in[0] << 9) == 0)
        return 0;                      /* inf */
    if (in[0] & 0x00400000)
        return 0;                      /* qnan */
    return 1;
}
int is_snan(const uint32 *in, int size)
{
    return size == 2 ? is_dsnan(in) : is_ssnan(in);
}

/*
 * Wrapper functions called to fix up unusual results after the main
 * test function has run.
 */
void universal_wrapper(wrapperctx *ctx)
{
    /*
     * Any SNaN input gives rise to a QNaN output.
     */
    int op;
    for (op = 0; op < wrapper_get_nops(ctx); op++) {
        int size = wrapper_get_size(ctx, op);

        if (!wrapper_is_complex(ctx, op) &&
            is_snan(wrapper_get_ieee(ctx, op), size)) {
            wrapper_set_nan(ctx);
        }
    }
}

Testable functions[] = {
    /*
     * Trig functions: sin, cos, tan. We test the core function
     * between -16 and +16: we assume that range reduction exists
     * and will be used for larger arguments, and we'll test that
     * separately. Also we only go down to 2^-27 in magnitude,
     * because below that sin(x)=tan(x)=x and cos(x)=1 as far as
     * double precision can tell, which is boring.
     */
    {"sin", (funcptr)mpfr_sin, args1, {NULL},
        cases_uniform, 0x3e400000, 0x40300000},
    {"sinf", (funcptr)mpfr_sin, args1f, {NULL},
        cases_uniform_float, 0x39800000, 0x41800000},
    {"cos", (funcptr)mpfr_cos, args1, {NULL},
        cases_uniform, 0x3e400000, 0x40300000},
    {"cosf", (funcptr)mpfr_cos, args1f, {NULL},
        cases_uniform_float, 0x39800000, 0x41800000},
    {"tan", (funcptr)mpfr_tan, args1, {NULL},
        cases_uniform, 0x3e400000, 0x40300000},
    {"tanf", (funcptr)mpfr_tan, args1f, {NULL},
        cases_uniform_float, 0x39800000, 0x41800000},
    {"sincosf_sinf", (funcptr)mpfr_sin, args1f, {NULL},
        cases_uniform_float, 0x39800000, 0x41800000},
    {"sincosf_cosf", (funcptr)mpfr_cos, args1f, {NULL},
        cases_uniform_float, 0x39800000, 0x41800000},
    /*
     * Inverse trig: asin, acos. Between 1 and -1, of course. acos
     * goes down to 2^-54, asin to 2^-27.
     */
    {"asin", (funcptr)mpfr_asin, args1, {NULL},
        cases_uniform, 0x3e400000, 0x3fefffff},
    {"asinf", (funcptr)mpfr_asin, args1f, {NULL},
        cases_uniform_float, 0x39800000, 0x3f7fffff},
    {"acos", (funcptr)mpfr_acos, args1, {NULL},
        cases_uniform, 0x3c900000, 0x3fefffff},
    {"acosf", (funcptr)mpfr_acos, args1f, {NULL},
        cases_uniform_float, 0x33800000, 0x3f7fffff},
    /*
     * Inverse trig: atan. atan is stable (in double prec) with
     * argument magnitude past 2^53, so we'll test up to there.
     * atan(x) is boringly just x below 2^-27.
     */
    {"atan", (funcptr)mpfr_atan, args1, {NULL},
        cases_uniform, 0x3e400000, 0x43400000},
    {"atanf", (funcptr)mpfr_atan, args1f, {NULL},
        cases_uniform_float, 0x39800000, 0x4b800000},
    /*
     * atan2. Interesting cases arise when the exponents of the
     * arguments differ by at most about 50.
     */
    {"atan2", (funcptr)mpfr_atan2, args2, {NULL},
        atan2_cases, 0},
    {"atan2f", (funcptr)mpfr_atan2, args2f, {NULL},
        atan2_cases_float, 0},
    /*
     * The exponentials: exp, sinh, cosh. They overflow at around
     * 710. exp and sinh are boring below 2^-54, cosh below 2^-27.
     */
    {"exp", (funcptr)mpfr_exp, args1, {NULL},
        cases_uniform, 0x3c900000, 0x40878000},
    {"expf", (funcptr)mpfr_exp, args1f, {NULL},
        cases_uniform_float, 0x33800000, 0x42dc0000},
    {"sinh", (funcptr)mpfr_sinh, args1, {NULL},
        cases_uniform, 0x3c900000, 0x40878000},
    {"sinhf", (funcptr)mpfr_sinh, args1f, {NULL},
        cases_uniform_float, 0x33800000, 0x42dc0000},
    {"cosh", (funcptr)mpfr_cosh, args1, {NULL},
        cases_uniform, 0x3e400000, 0x40878000},
    {"coshf", (funcptr)mpfr_cosh, args1f, {NULL},
        cases_uniform_float, 0x39800000, 0x42dc0000},
    /*
     * tanh is stable past around 20. It's boring below 2^-27.
     */
    {"tanh", (funcptr)mpfr_tanh, args1, {NULL},
        cases_uniform, 0x3e400000, 0x40340000},
    {"tanhf", (funcptr)mpfr_tanh, args1f, {NULL},
        cases_uniform, 0x39800000, 0x41100000},
    /*
     * log must be tested only on positive numbers, but can cover
     * the whole range of positive nonzero finite numbers. It never
     * gets boring.
     */
    {"log", (funcptr)mpfr_log, args1, {NULL}, log_cases, 0},
    {"logf", (funcptr)mpfr_log, args1f, {NULL}, log_cases_float, 0},
    {"log10", (funcptr)mpfr_log10, args1, {NULL}, log_cases, 0},
    {"log10f", (funcptr)mpfr_log10, args1f, {NULL}, log_cases_float, 0},
    /*
     * pow.
     */
    {"pow", (funcptr)mpfr_pow, args2, {NULL}, pow_cases, 0},
    {"powf", (funcptr)mpfr_pow, args2f, {NULL}, pow_cases_float, 0},
    /*
     * Trig range reduction. We are able to test this for all
     * finite values, but will only bother for things between 2^-3
     * and 2^+52.
     */
    {"rred", (funcptr)test_rred, rred, {NULL}, rred_cases, 0},
    {"rredf", (funcptr)test_rred, rredf, {NULL}, rred_cases_float, 0},
    /*
     * Square and cube root.
     */
    {"sqrt", (funcptr)mpfr_sqrt, args1, {NULL}, log_cases, 0},
    {"sqrtf", (funcptr)mpfr_sqrt, args1f, {NULL}, log_cases_float, 0},
    {"cbrt", (funcptr)mpfr_cbrt, args1, {NULL}, log_cases, 0},
    {"cbrtf", (funcptr)mpfr_cbrt, args1f, {NULL}, log_cases_float, 0},
    {"hypot", (funcptr)mpfr_hypot, args2, {NULL}, atan2_cases, 0},
    {"hypotf", (funcptr)mpfr_hypot, args2f, {NULL}, atan2_cases_float, 0},
    /*
     * Seminumerical functions.
     */
    {"ceil", (funcptr)test_ceil, semi1, {NULL}, cases_semi1},
    {"ceilf", (funcptr)test_ceilf, semi1f, {NULL}, cases_semi1_float},
    {"floor", (funcptr)test_floor, semi1, {NULL}, cases_semi1},
    {"floorf", (funcptr)test_floorf, semi1f, {NULL}, cases_semi1_float},
    {"fmod", (funcptr)test_fmod, semi2, {NULL}, cases_semi2},
    {"fmodf", (funcptr)test_fmodf, semi2f, {NULL}, cases_semi2_float},
    {"ldexp", (funcptr)test_ldexp, t_ldexp, {NULL}, cases_ldexp},
    {"ldexpf", (funcptr)test_ldexpf, t_ldexpf, {NULL}, cases_ldexp_float},
    {"frexp", (funcptr)test_frexp, t_frexp, {NULL}, cases_semi1},
    {"frexpf", (funcptr)test_frexpf, t_frexpf, {NULL}, cases_semi1_float},
    {"modf", (funcptr)test_modf, t_modf, {NULL}, cases_semi1},
    {"modff", (funcptr)test_modff, t_modff, {NULL}, cases_semi1_float},

    /*
     * Classification and more semi-numericals
     */
    {"copysign", (funcptr)test_copysign, semi2, {NULL}, cases_semi2},
    {"copysignf", (funcptr)test_copysignf, semi2f, {NULL}, cases_semi2_float},
    {"isfinite", (funcptr)test_isfinite, classify, {NULL}, cases_uniform, 0, 0x7fffffff},
    {"isfinitef", (funcptr)test_isfinitef, classifyf, {NULL}, cases_uniform_float, 0, 0x7fffffff},
    {"isinf", (funcptr)test_isinf, classify, {NULL}, cases_uniform, 0, 0x7fffffff},
    {"isinff", (funcptr)test_isinff, classifyf, {NULL}, cases_uniform_float, 0, 0x7fffffff},
    {"isnan", (funcptr)test_isnan, classify, {NULL}, cases_uniform, 0, 0x7fffffff},
    {"isnanf", (funcptr)test_isnanf, classifyf, {NULL}, cases_uniform_float, 0, 0x7fffffff},
    {"isnormal", (funcptr)test_isnormal, classify, {NULL}, cases_uniform, 0, 0x7fffffff},
    {"isnormalf", (funcptr)test_isnormalf, classifyf, {NULL}, cases_uniform_float, 0, 0x7fffffff},
    {"signbit", (funcptr)test_signbit, classify, {NULL}, cases_uniform, 0, 0x7fffffff},
    {"signbitf", (funcptr)test_signbitf, classifyf, {NULL}, cases_uniform_float, 0, 0x7fffffff},
    {"fpclassify", (funcptr)test_fpclassify, classify, {NULL}, cases_uniform, 0, 0x7fffffff},
    {"fpclassifyf", (funcptr)test_fpclassifyf, classifyf, {NULL}, cases_uniform_float, 0, 0x7fffffff},
    /*
     * Comparisons
     */
    {"isgreater", (funcptr)test_isgreater, compare, {NULL}, cases_uniform, 0, 0x7fffffff},
    {"isgreaterequal", (funcptr)test_isgreaterequal, compare, {NULL}, cases_uniform, 0, 0x7fffffff},
    {"isless", (funcptr)test_isless, compare, {NULL}, cases_uniform, 0, 0x7fffffff},
    {"islessequal", (funcptr)test_islessequal, compare, {NULL}, cases_uniform, 0, 0x7fffffff},
    {"islessgreater", (funcptr)test_islessgreater, compare, {NULL}, cases_uniform, 0, 0x7fffffff},
    {"isunordered", (funcptr)test_isunordered, compare, {NULL}, cases_uniform, 0, 0x7fffffff},

    {"isgreaterf", (funcptr)test_isgreaterf, comparef, {NULL}, cases_uniform_float, 0, 0x7fffffff},
    {"isgreaterequalf", (funcptr)test_isgreaterequalf, comparef, {NULL}, cases_uniform_float, 0, 0x7fffffff},
    {"islessf", (funcptr)test_islessf, comparef, {NULL}, cases_uniform_float, 0, 0x7fffffff},
    {"islessequalf", (funcptr)test_islessequalf, comparef, {NULL}, cases_uniform_float, 0, 0x7fffffff},
    {"islessgreaterf", (funcptr)test_islessgreaterf, comparef, {NULL}, cases_uniform_float, 0, 0x7fffffff},
    {"isunorderedf", (funcptr)test_isunorderedf, comparef, {NULL}, cases_uniform_float, 0, 0x7fffffff},

    /*
     * Inverse Hyperbolic functions
     */
    {"atanh", (funcptr)mpfr_atanh, args1, {NULL}, cases_uniform, 0x3e400000, 0x3fefffff},
    {"asinh", (funcptr)mpfr_asinh, args1, {NULL}, cases_uniform, 0x3e400000, 0x3fefffff},
    {"acosh", (funcptr)mpfr_acosh, args1, {NULL}, cases_uniform_positive, 0x3ff00000, 0x7fefffff},

    {"atanhf", (funcptr)mpfr_atanh, args1f, {NULL}, cases_uniform_float, 0x32000000, 0x3f7fffff},
    {"asinhf", (funcptr)mpfr_asinh, args1f, {NULL}, cases_uniform_float, 0x32000000, 0x3f7fffff},
    {"acoshf", (funcptr)mpfr_acosh, args1f, {NULL}, cases_uniform_float_positive, 0x3f800000, 0x7f800000},

    /*
     * Everything else (sitting in a section down here at the bottom
     * because historically they were not tested because we didn't
     * have reference implementations for them)
     */
    {"csin", (funcptr)mpc_sin, args1c, {NULL}, complex_cases_uniform, 0x3f000000, 0x40300000},
    {"csinf", (funcptr)mpc_sin, args1fc, {NULL}, complex_cases_uniform_float, 0x38000000, 0x41800000},
    {"ccos", (funcptr)mpc_cos, args1c, {NULL}, complex_cases_uniform, 0x3f000000, 0x40300000},
    {"ccosf", (funcptr)mpc_cos, args1fc, {NULL}, complex_cases_uniform_float, 0x38000000, 0x41800000},
    {"ctan", (funcptr)mpc_tan, args1c, {NULL}, complex_cases_uniform, 0x3f000000, 0x40300000},
    {"ctanf", (funcptr)mpc_tan, args1fc, {NULL}, complex_cases_uniform_float, 0x38000000, 0x41800000},

    {"casin", (funcptr)mpc_asin, args1c, {NULL}, complex_cases_uniform, 0x3f000000, 0x40300000},
    {"casinf", (funcptr)mpc_asin, args1fc, {NULL}, complex_cases_uniform_float, 0x38000000, 0x41800000},
    {"cacos", (funcptr)mpc_acos, args1c, {NULL}, complex_cases_uniform, 0x3f000000, 0x40300000},
    {"cacosf", (funcptr)mpc_acos, args1fc, {NULL}, complex_cases_uniform_float, 0x38000000, 0x41800000},
    {"catan", (funcptr)mpc_atan, args1c, {NULL}, complex_cases_uniform, 0x3f000000, 0x40300000},
    {"catanf", (funcptr)mpc_atan, args1fc, {NULL}, complex_cases_uniform_float, 0x38000000, 0x41800000},

    {"csinh", (funcptr)mpc_sinh, args1c, {NULL}, complex_cases_uniform, 0x3f000000, 0x40300000},
    {"csinhf", (funcptr)mpc_sinh, args1fc, {NULL}, complex_cases_uniform_float, 0x38000000, 0x41800000},
    {"ccosh", (funcptr)mpc_cosh, args1c, {NULL}, complex_cases_uniform, 0x3f000000, 0x40300000},
    {"ccoshf", (funcptr)mpc_cosh, args1fc, {NULL}, complex_cases_uniform_float, 0x38000000, 0x41800000},
    {"ctanh", (funcptr)mpc_tanh, args1c, {NULL}, complex_cases_uniform, 0x3f000000, 0x40300000},
    {"ctanhf", (funcptr)mpc_tanh, args1fc, {NULL}, complex_cases_uniform_float, 0x38000000, 0x41800000},

    {"casinh", (funcptr)mpc_asinh, args1c, {NULL}, complex_cases_uniform, 0x3f000000, 0x40300000},
    {"casinhf", (funcptr)mpc_asinh, args1fc, {NULL}, complex_cases_uniform_float, 0x38000000, 0x41800000},
    {"cacosh", (funcptr)mpc_acosh, args1c, {NULL}, complex_cases_uniform, 0x3f000000, 0x40300000},
    {"cacoshf", (funcptr)mpc_acosh, args1fc, {NULL}, complex_cases_uniform_float, 0x38000000, 0x41800000},
    {"catanh", (funcptr)mpc_atanh, args1c, {NULL}, complex_cases_uniform, 0x3f000000, 0x40300000},
    {"catanhf", (funcptr)mpc_atanh, args1fc, {NULL}, complex_cases_uniform_float, 0x38000000, 0x41800000},

    {"cexp", (funcptr)mpc_exp, args1c, {NULL}, complex_cases_uniform, 0x3c900000, 0x40862000},
    {"cpow", (funcptr)test_cpow, args2c, {NULL}, complex_pow_cases, 0x3fc00000, 0x40000000},
    {"clog", (funcptr)mpc_log, args1c, {NULL}, complex_log_cases, 0, 0},
    {"csqrt", (funcptr)mpc_sqrt, args1c, {NULL}, complex_log_cases, 0, 0},

    {"cexpf", (funcptr)mpc_exp, args1fc, {NULL}, complex_cases_uniform_float, 0x24800000, 0x42b00000},
    {"cpowf", (funcptr)test_cpow, args2fc, {NULL}, complex_pow_cases_float, 0x3e000000, 0x41000000},
    {"clogf", (funcptr)mpc_log, args1fc, {NULL}, complex_log_cases_float, 0, 0},
    {"csqrtf", (funcptr)mpc_sqrt, args1fc, {NULL}, complex_log_cases_float, 0, 0},

    {"cdiv", (funcptr)mpc_div, args2c, {NULL}, complex_arithmetic_cases, 0, 0},
    {"cmul", (funcptr)mpc_mul, args2c, {NULL}, complex_arithmetic_cases, 0, 0},
    {"cadd", (funcptr)mpc_add, args2c, {NULL}, complex_arithmetic_cases, 0, 0},
    {"csub", (funcptr)mpc_sub, args2c, {NULL}, complex_arithmetic_cases, 0, 0},

    {"cdivf", (funcptr)mpc_div, args2fc, {NULL}, complex_arithmetic_cases_float, 0, 0},
    {"cmulf", (funcptr)mpc_mul, args2fc, {NULL}, complex_arithmetic_cases_float, 0, 0},
    {"caddf", (funcptr)mpc_add, args2fc, {NULL}, complex_arithmetic_cases_float, 0, 0},
    {"csubf", (funcptr)mpc_sub, args2fc, {NULL}, complex_arithmetic_cases_float, 0, 0},

    {"cabsf", (funcptr)mpc_abs, args1fcr, {NULL}, complex_arithmetic_cases_float, 0, 0},
    {"cabs", (funcptr)mpc_abs, args1cr, {NULL}, complex_arithmetic_cases, 0, 0},
    {"cargf", (funcptr)mpc_arg, args1fcr, {NULL}, complex_arithmetic_cases_float, 0, 0},
    {"carg", (funcptr)mpc_arg, args1cr, {NULL}, complex_arithmetic_cases, 0, 0},
    {"cimagf", (funcptr)mpc_imag, args1fcr, {NULL}, complex_arithmetic_cases_float, 0, 0},
    {"cimag", (funcptr)mpc_imag, args1cr, {NULL}, complex_arithmetic_cases, 0, 0},
    {"conjf", (funcptr)mpc_conj, args1fc, {NULL}, complex_arithmetic_cases_float, 0, 0},
    {"conj", (funcptr)mpc_conj, args1c, {NULL}, complex_arithmetic_cases, 0, 0},
    {"cprojf", (funcptr)mpc_proj, args1fc, {NULL}, complex_arithmetic_cases_float, 0, 0},
    {"cproj", (funcptr)mpc_proj, args1c, {NULL}, complex_arithmetic_cases, 0, 0},
    {"crealf", (funcptr)mpc_real, args1fcr, {NULL}, complex_arithmetic_cases_float, 0, 0},
    {"creal", (funcptr)mpc_real, args1cr, {NULL}, complex_arithmetic_cases, 0, 0},
    {"erfcf", (funcptr)mpfr_erfc, args1f, {NULL}, cases_biased_float, 0x1e800000, 0x41000000},
    {"erfc", (funcptr)mpfr_erfc, args1, {NULL}, cases_biased, 0x3bd00000, 0x403c0000},
    {"erff", (funcptr)mpfr_erf, args1f, {NULL}, cases_biased_float, 0x03800000, 0x40700000},
    {"erf", (funcptr)mpfr_erf, args1, {NULL}, cases_biased, 0x00800000, 0x40200000},
    {"exp2f", (funcptr)mpfr_exp2, args1f, {NULL}, cases_uniform_float, 0x33800000, 0x43c00000},
    {"exp2", (funcptr)mpfr_exp2, args1, {NULL}, cases_uniform, 0x3ca00000, 0x40a00000},
    {"expm1f", (funcptr)mpfr_expm1, args1f, {NULL}, cases_uniform_float, 0x33000000, 0x43800000},
    {"expm1", (funcptr)mpfr_expm1, args1, {NULL}, cases_uniform, 0x3c900000, 0x409c0000},
    {"fmaxf", (funcptr)mpfr_max, args2f, {NULL}, minmax_cases_float, 0, 0x7f7fffff},
    {"fmax", (funcptr)mpfr_max, args2, {NULL}, minmax_cases, 0, 0x7fefffff},
    {"fminf", (funcptr)mpfr_min, args2f, {NULL}, minmax_cases_float, 0, 0x7f7fffff},
    {"fmin", (funcptr)mpfr_min, args2, {NULL}, minmax_cases, 0, 0x7fefffff},
    {"lgammaf", (funcptr)test_lgamma, args1f, {NULL}, cases_uniform_float, 0x01800000, 0x7f800000},
    {"lgamma", (funcptr)test_lgamma, args1, {NULL}, cases_uniform, 0x00100000, 0x7ff00000},
    {"log1pf", (funcptr)mpfr_log1p, args1f, {NULL}, log1p_cases_float, 0, 0},
    {"log1p", (funcptr)mpfr_log1p, args1, {NULL}, log1p_cases, 0, 0},
    {"log2f", (funcptr)mpfr_log2, args1f, {NULL}, log_cases_float, 0, 0},
    {"log2", (funcptr)mpfr_log2, args1, {NULL}, log_cases, 0, 0},
    {"tgammaf", (funcptr)mpfr_gamma, args1f, {NULL}, cases_uniform_float, 0x2f800000, 0x43000000},
    {"tgamma", (funcptr)mpfr_gamma, args1, {NULL}, cases_uniform, 0x3c000000, 0x40800000},
};

const int nfunctions = ( sizeof(functions)/sizeof(*functions) );

#define random_sign ( random_upto(1) ? 0x80000000 : 0 )

static int iszero(uint32 *x) {
    return !((x[0] & 0x7FFFFFFF) || x[1]);
}


static void complex_log_cases(uint32 *out, uint32 param1,
                              uint32 param2) {
    cases_uniform(out,0x00100000,0x7fefffff);
    cases_uniform(out+2,0x00100000,0x7fefffff);
}


static void complex_log_cases_float(uint32 *out, uint32 param1,
                                    uint32 param2) {
    cases_uniform_float(out,0x00800000,0x7f7fffff);
    cases_uniform_float(out+2,0x00800000,0x7f7fffff);
}

static void complex_cases_biased(uint32 *out, uint32 lowbound,
                                 uint32 highbound) {
    cases_biased(out,lowbound,highbound);
    cases_biased(out+2,lowbound,highbound);
}

static void complex_cases_biased_float(uint32 *out, uint32 lowbound,
                                       uint32 highbound) {
    cases_biased_float(out,lowbound,highbound);
    cases_biased_float(out+2,lowbound,highbound);
}

static void complex_cases_uniform(uint32 *out, uint32 lowbound,
                                 uint32 highbound) {
    cases_uniform(out,lowbound,highbound);
    cases_uniform(out+2,lowbound,highbound);
}

static void complex_cases_uniform_float(uint32 *out, uint32 lowbound,
                                       uint32 highbound) {
    cases_uniform_float(out,lowbound,highbound);
    cases_uniform(out+2,lowbound,highbound);
}

static void complex_pow_cases(uint32 *out, uint32 lowbound,
                              uint32 highbound) {
    /*
     * Generating non-overflowing cases for complex pow:
     *
     * Our base has both parts within the range [1/2,2], and hence
     * its magnitude is within [1/2,2*sqrt(2)]. The magnitude of its
     * logarithm in base 2 is therefore at most the magnitude of
     * (log2(2*sqrt(2)) + i*pi/log(2)), or in other words
     * hypot(3/2,pi/log(2)) = 4.77. So the magnitude of the exponent
     * input must be at most our output magnitude limit (as a power
     * of two) divided by that.
     *
     * I also set the output magnitude limit a bit low, because we
     * don't guarantee (and neither does glibc) to prevent internal
     * overflow in cases where the output _magnitude_ overflows but
     * scaling it back down by cos and sin of the argument brings it
     * back in range.
     */
    cases_uniform(out,0x3fe00000, 0x40000000);
    cases_uniform(out+2,0x3fe00000, 0x40000000);
    cases_uniform(out+4,0x3f800000, 0x40600000);
    cases_uniform(out+6,0x3f800000, 0x40600000);
}

static void complex_pow_cases_float(uint32 *out, uint32 lowbound,
                                    uint32 highbound) {
    /*
     * Reasoning as above, though of course the detailed numbers are
     * all different.
     */
    cases_uniform_float(out,0x3f000000, 0x40000000);
    cases_uniform_float(out+2,0x3f000000, 0x40000000);
    cases_uniform_float(out+4,0x3d600000, 0x41900000);
    cases_uniform_float(out+6,0x3d600000, 0x41900000);
}

static void complex_arithmetic_cases(uint32 *out, uint32 lowbound,
                                     uint32 highbound) {
    cases_uniform(out,0,0x7fefffff);
    cases_uniform(out+2,0,0x7fefffff);
    cases_uniform(out+4,0,0x7fefffff);
    cases_uniform(out+6,0,0x7fefffff);
}

static void complex_arithmetic_cases_float(uint32 *out, uint32 lowbound,
                                           uint32 highbound) {
    cases_uniform_float(out,0,0x7f7fffff);
    cases_uniform_float(out+2,0,0x7f7fffff);
    cases_uniform_float(out+4,0,0x7f7fffff);
    cases_uniform_float(out+6,0,0x7f7fffff);
}

/*
 * Included from fplib test suite, in a compact self-contained
 * form.
 */

void float32_case(uint32 *ret) {
    int n, bits;
    uint32 f;
    static int premax, preptr;
    static uint32 *specifics = NULL;

    if (!ret) {
        if (specifics)
            free(specifics);
        specifics = NULL;
        premax = preptr = 0;
        return;
    }

    if (!specifics) {
        int exps[] = {
            -127, -126, -125, -24, -4, -3, -2, -1, 0, 1, 2, 3, 4,
                24, 29, 30, 31, 32, 61, 62, 63, 64, 126, 127, 128
        };
        int sign, eptr;
        uint32 se, j;
        /*
         * We want a cross product of:
         *  - each of two sign bits (2)
         *  - each of the above (unbiased) exponents (25)
         *  - the following list of fraction parts:
         *    * zero (1)
         *    * all bits (1)
         *    * one-bit-set (23)
         *    * one-bit-clear (23)
         *    * one-bit-and-above (20: 3 are duplicates)
         *    * one-bit-and-below (20: 3 are duplicates)
         *    (total 88)
         *  (total 4400)
         */
        specifics = malloc(4400 * sizeof(*specifics));
        preptr = 0;
        for (sign = 0; sign <= 1; sign++) {
            for (eptr = 0; eptr < sizeof(exps)/sizeof(*exps); eptr++) {
                se = (sign ? 0x80000000 : 0) | ((exps[eptr]+127) << 23);
                /*
                 * Zero.
                 */
                specifics[preptr++] = se | 0;
                /*
                 * All bits.
                 */
                specifics[preptr++] = se | 0x7FFFFF;
                /*
                 * One-bit-set.
                 */
                for (j = 1; j && j <= 0x400000; j <<= 1)
                    specifics[preptr++] = se | j;
                /*
                 * One-bit-clear.
                 */
                for (j = 1; j && j <= 0x400000; j <<= 1)
                    specifics[preptr++] = se | (0x7FFFFF ^ j);
                /*
                 * One-bit-and-everything-below.
                 */
                for (j = 2; j && j <= 0x100000; j <<= 1)
                    specifics[preptr++] = se | (2*j-1);
                /*
                 * One-bit-and-everything-above.
                 */
                for (j = 4; j && j <= 0x200000; j <<= 1)
                    specifics[preptr++] = se | (0x7FFFFF ^ (j-1));
                /*
                 * Done.
                 */
            }
        }
        assert(preptr == 4400);
        premax = preptr;
    }

    /*
     * Decide whether to return a pre or a random case.
     */
    n = random32() % (premax+1);
    if (n < preptr) {
        /*
         * Return pre[n].
         */
        uint32 t;
        t = specifics[n];
        specifics[n] = specifics[preptr-1];
        specifics[preptr-1] = t;        /* (not really needed) */
        preptr--;
        *ret = t;
    } else {
        /*
         * Random case.
         * Sign and exponent:
         *  - FIXME
         * Significand:
         *  - with prob 1/5, a totally random bit pattern
         *  - with prob 1/5, all 1s down to some point and then random
         *  - with prob 1/5, all 1s up to some point and then random
         *  - with prob 1/5, all 0s down to some point and then random
         *  - with prob 1/5, all 0s up to some point and then random
         */
        n = random32() % 5;
        f = random32();                /* some random bits */
        bits = random32() % 22 + 1;    /* 1-22 */
        switch (n) {
          case 0:
            break;                     /* leave f alone */
          case 1:
            f |= (1<<bits)-1;
            break;
          case 2:
            f &= ~((1<<bits)-1);
            break;
          case 3:
            f |= ~((1<<bits)-1);
            break;
          case 4:
            f &= (1<<bits)-1;
            break;
        }
        f &= 0x7FFFFF;
        f |= (random32() & 0xFF800000);/* FIXME - do better */
        *ret = f;
    }
}
static void float64_case(uint32 *ret) {
    int n, bits;
    uint32 f, g;
    static int premax, preptr;
    static uint32 (*specifics)[2] = NULL;

    if (!ret) {
        if (specifics)
            free(specifics);
        specifics = NULL;
        premax = preptr = 0;
        return;
    }

    if (!specifics) {
        int exps[] = {
            -1023, -1022, -1021, -129, -128, -127, -126, -53, -4, -3, -2,
            -1, 0, 1, 2, 3, 4, 29, 30, 31, 32, 53, 61, 62, 63, 64, 127,
            128, 129, 1022, 1023, 1024
        };
        int sign, eptr;
        uint32 se, j;
        /*
         * We want a cross product of:
         *  - each of two sign bits (2)
         *  - each of the above (unbiased) exponents (32)
         *  - the following list of fraction parts:
         *    * zero (1)
         *    * all bits (1)
         *    * one-bit-set (52)
         *    * one-bit-clear (52)
         *    * one-bit-and-above (49: 3 are duplicates)
         *    * one-bit-and-below (49: 3 are duplicates)
         *    (total 204)
         *  (total 13056)
         */
        specifics = malloc(13056 * sizeof(*specifics));
        preptr = 0;
        for (sign = 0; sign <= 1; sign++) {
            for (eptr = 0; eptr < sizeof(exps)/sizeof(*exps); eptr++) {
                se = (sign ? 0x80000000 : 0) | ((exps[eptr]+1023) << 20);
                /*
                 * Zero.
                 */
                specifics[preptr][0] = 0;
                specifics[preptr][1] = 0;
                specifics[preptr++][0] |= se;
                /*
                 * All bits.
                 */
                specifics[preptr][0] = 0xFFFFF;
                specifics[preptr][1] = ~0;
                specifics[preptr++][0] |= se;
                /*
                 * One-bit-set.
                 */
                for (j = 1; j && j <= 0x80000000; j <<= 1) {
                    specifics[preptr][0] = 0;
                    specifics[preptr][1] = j;
                    specifics[preptr++][0] |= se;
                    if (j & 0xFFFFF) {
                        specifics[preptr][0] = j;
                        specifics[preptr][1] = 0;
                        specifics[preptr++][0] |= se;
                    }
                }
                /*
                 * One-bit-clear.
                 */
                for (j = 1; j && j <= 0x80000000; j <<= 1) {
                    specifics[preptr][0] = 0xFFFFF;
                    specifics[preptr][1] = ~j;
                    specifics[preptr++][0] |= se;
                    if (j & 0xFFFFF) {
                        specifics[preptr][0] = 0xFFFFF ^ j;
                        specifics[preptr][1] = ~0;
                        specifics[preptr++][0] |= se;
                    }
                }
                /*
                 * One-bit-and-everything-below.
                 */
                for (j = 2; j && j <= 0x80000000; j <<= 1) {
                    specifics[preptr][0] = 0;
                    specifics[preptr][1] = 2*j-1;
                    specifics[preptr++][0] |= se;
                }
                for (j = 1; j && j <= 0x20000; j <<= 1) {
                    specifics[preptr][0] = 2*j-1;
                    specifics[preptr][1] = ~0;
                    specifics[preptr++][0] |= se;
                }
                /*
                 * One-bit-and-everything-above.
                 */
                for (j = 4; j && j <= 0x80000000; j <<= 1) {
                    specifics[preptr][0] = 0xFFFFF;
                    specifics[preptr][1] = ~(j-1);
                    specifics[preptr++][0] |= se;
                }
                for (j = 1; j && j <= 0x40000; j <<= 1) {
                    specifics[preptr][0] = 0xFFFFF ^ (j-1);
                    specifics[preptr][1] = 0;
                    specifics[preptr++][0] |= se;
                }
                /*
                 * Done.
                 */
            }
        }
        assert(preptr == 13056);
        premax = preptr;
    }

    /*
     * Decide whether to return a pre or a random case.
     */
    n = (uint32) random32() % (uint32) (premax+1);
    if (n < preptr) {
        /*
         * Return pre[n].
         */
        uint32 t;
        t = specifics[n][0];
        specifics[n][0] = specifics[preptr-1][0];
        specifics[preptr-1][0] = t;     /* (not really needed) */
        ret[0] = t;
        t = specifics[n][1];
        specifics[n][1] = specifics[preptr-1][1];
        specifics[preptr-1][1] = t;     /* (not really needed) */
        ret[1] = t;
        preptr--;
    } else {
        /*
         * Random case.
         * Sign and exponent:
         *  - FIXME
         * Significand:
         *  - with prob 1/5, a totally random bit pattern
         *  - with prob 1/5, all 1s down to some point and then random
         *  - with prob 1/5, all 1s up to some point and then random
         *  - with prob 1/5, all 0s down to some point and then random
         *  - with prob 1/5, all 0s up to some point and then random
         */
        n = random32() % 5;
        f = random32();                /* some random bits */
        g = random32();                /* some random bits */
        bits = random32() % 51 + 1;    /* 1-51 */
        switch (n) {
          case 0:
            break;                     /* leave f alone */
          case 1:
            if (bits <= 32)
                f |= (1<<bits)-1;
            else {
                bits -= 32;
                g |= (1<<bits)-1;
                f = ~0;
            }
            break;
          case 2:
            if (bits <= 32)
                f &= ~((1<<bits)-1);
            else {
                bits -= 32;
                g &= ~((1<<bits)-1);
                f = 0;
            }
            break;
          case 3:
            if (bits <= 32)
                g &= (1<<bits)-1;
            else {
                bits -= 32;
                f &= (1<<bits)-1;
                g = 0;
            }
            break;
          case 4:
            if (bits <= 32)
                g |= ~((1<<bits)-1);
            else {
                bits -= 32;
                f |= ~((1<<bits)-1);
                g = ~0;
            }
            break;
        }
        g &= 0xFFFFF;
        g |= (random32() & 0xFFF00000);/* FIXME - do better */
        ret[0] = g;
        ret[1] = f;
    }
}

static void cases_biased(uint32 *out, uint32 lowbound,
                          uint32 highbound) {
    do {
        out[0] = highbound - random_upto_biased(highbound-lowbound, 8);
        out[1] = random_upto(0xFFFFFFFF);
        out[0] |= random_sign;
    } while (iszero(out));             /* rule out zero */
}

static void cases_biased_positive(uint32 *out, uint32 lowbound,
                                  uint32 highbound) {
    do {
        out[0] = highbound - random_upto_biased(highbound-lowbound, 8);
        out[1] = random_upto(0xFFFFFFFF);
    } while (iszero(out));             /* rule out zero */
}

static void cases_biased_float(uint32 *out, uint32 lowbound,
                               uint32 highbound) {
    do {
        out[0] = highbound - random_upto_biased(highbound-lowbound, 8);
        out[1] = 0;
        out[0] |= random_sign;
    } while (iszero(out));             /* rule out zero */
}

static void cases_semi1(uint32 *out, uint32 param1,
                        uint32 param2) {
    float64_case(out);
}

static void cases_semi1_float(uint32 *out, uint32 param1,
                              uint32 param2) {
    float32_case(out);
}

static void cases_semi2(uint32 *out, uint32 param1,
                        uint32 param2) {
    float64_case(out);
    float64_case(out+2);
}

static void cases_semi2_float(uint32 *out, uint32 param1,
                        uint32 param2) {
    float32_case(out);
    float32_case(out+2);
}

static void cases_ldexp(uint32 *out, uint32 param1,
                        uint32 param2) {
    float64_case(out);
    out[2] = random_upto(2048)-1024;
}

static void cases_ldexp_float(uint32 *out, uint32 param1,
                              uint32 param2) {
    float32_case(out);
    out[2] = random_upto(256)-128;
}

static void cases_uniform(uint32 *out, uint32 lowbound,
                          uint32 highbound) {
    do {
        out[0] = highbound - random_upto(highbound-lowbound);
        out[1] = random_upto(0xFFFFFFFF);
        out[0] |= random_sign;
    } while (iszero(out));             /* rule out zero */
}
static void cases_uniform_float(uint32 *out, uint32 lowbound,
                                uint32 highbound) {
    do {
        out[0] = highbound - random_upto(highbound-lowbound);
        out[1] = 0;
        out[0] |= random_sign;
    } while (iszero(out));             /* rule out zero */
}

static void cases_uniform_positive(uint32 *out, uint32 lowbound,
                                   uint32 highbound) {
    do {
        out[0] = highbound - random_upto(highbound-lowbound);
        out[1] = random_upto(0xFFFFFFFF);
    } while (iszero(out));             /* rule out zero */
}
static void cases_uniform_float_positive(uint32 *out, uint32 lowbound,
                                         uint32 highbound) {
    do {
        out[0] = highbound - random_upto(highbound-lowbound);
        out[1] = 0;
    } while (iszero(out));             /* rule out zero */
}


static void log_cases(uint32 *out, uint32 param1,
                      uint32 param2) {
    do {
        out[0] = random_upto(0x7FEFFFFF);
        out[1] = random_upto(0xFFFFFFFF);
    } while (iszero(out));             /* rule out zero */
}

static void log_cases_float(uint32 *out, uint32 param1,
                            uint32 param2) {
    do {
        out[0] = random_upto(0x7F7FFFFF);
        out[1] = 0;
    } while (iszero(out));             /* rule out zero */
}

static void log1p_cases(uint32 *out, uint32 param1, uint32 param2)
{
    uint32 sign = random_sign;
    if (sign == 0) {
        cases_uniform_positive(out, 0x3c700000, 0x43400000);
    } else {
        cases_uniform_positive(out, 0x3c000000, 0x3ff00000);
    }
    out[0] |= sign;
}

static void log1p_cases_float(uint32 *out, uint32 param1, uint32 param2)
{
    uint32 sign = random_sign;
    if (sign == 0) {
        cases_uniform_float_positive(out, 0x32000000, 0x4c000000);
    } else {
        cases_uniform_float_positive(out, 0x30000000, 0x3f800000);
    }
    out[0] |= sign;
}

static void minmax_cases(uint32 *out, uint32 param1, uint32 param2)
{
    do {
        out[0] = random_upto(0x7FEFFFFF);
        out[1] = random_upto(0xFFFFFFFF);
        out[0] |= random_sign;
        out[2] = random_upto(0x7FEFFFFF);
        out[3] = random_upto(0xFFFFFFFF);
        out[2] |= random_sign;
    } while (iszero(out));             /* rule out zero */
}

static void minmax_cases_float(uint32 *out, uint32 param1, uint32 param2)
{
    do {
        out[0] = random_upto(0x7F7FFFFF);
        out[1] = 0;
        out[0] |= random_sign;
        out[2] = random_upto(0x7F7FFFFF);
        out[3] = 0;
        out[2] |= random_sign;
    } while (iszero(out));             /* rule out zero */
}

static void rred_cases(uint32 *out, uint32 param1,
                       uint32 param2) {
    do {
        out[0] = ((0x3fc00000 + random_upto(0x036fffff)) |
                  (random_upto(1) << 31));
        out[1] = random_upto(0xFFFFFFFF);
    } while (iszero(out));             /* rule out zero */
}

static void rred_cases_float(uint32 *out, uint32 param1,
                             uint32 param2) {
    do {
        out[0] = ((0x3e000000 + random_upto(0x0cffffff)) |
                  (random_upto(1) << 31));
        out[1] = 0;                    /* for iszero */
    } while (iszero(out));             /* rule out zero */
}

static void atan2_cases(uint32 *out, uint32 param1,
                        uint32 param2) {
    do {
        int expdiff = random_upto(101)-51;
        int swap;
        if (expdiff < 0) {
            expdiff = -expdiff;
            swap = 2;
        } else
            swap = 0;
        out[swap ^ 0] = random_upto(0x7FEFFFFF-((expdiff+1)<<20));
        out[swap ^ 2] = random_upto(((expdiff+1)<<20)-1) + out[swap ^ 0];
        out[1] = random_upto(0xFFFFFFFF);
        out[3] = random_upto(0xFFFFFFFF);
        out[0] |= random_sign;
        out[2] |= random_sign;
    } while (iszero(out) || iszero(out+2));/* rule out zero */
}

static void atan2_cases_float(uint32 *out, uint32 param1,
                              uint32 param2) {
    do {
        int expdiff = random_upto(44)-22;
        int swap;
        if (expdiff < 0) {
            expdiff = -expdiff;
            swap = 2;
        } else
            swap = 0;
        out[swap ^ 0] = random_upto(0x7F7FFFFF-((expdiff+1)<<23));
        out[swap ^ 2] = random_upto(((expdiff+1)<<23)-1) + out[swap ^ 0];
        out[0] |= random_sign;
        out[2] |= random_sign;
        out[1] = out[3] = 0;           /* for iszero */
    } while (iszero(out) || iszero(out+2));/* rule out zero */
}

static void pow_cases(uint32 *out, uint32 param1,
                      uint32 param2) {
    /*
     * Pick an exponent e (-0x33 to +0x7FE) for x, and here's the
     * range of numbers we can use as y:
     *
     * For e < 0x3FE, the range is [-0x400/(0x3FE-e),+0x432/(0x3FE-e)]
     * For e > 0x3FF, the range is [-0x432/(e-0x3FF),+0x400/(e-0x3FF)]
     *
     * For e == 0x3FE or e == 0x3FF, the range gets infinite at one
     * end or the other, so we have to be cleverer: pick a number n
     * of useful bits in the mantissa (1 thru 52, so 1 must imply
     * 0x3ff00000.00000001 whereas 52 is anything at least as big
     * as 0x3ff80000.00000000; for e == 0x3fe, 1 necessarily means
     * 0x3fefffff.ffffffff and 52 is anything at most as big as
     * 0x3fe80000.00000000). Then, as it happens, a sensible
     * maximum power is 2^(63-n) for e == 0x3fe, and 2^(62-n) for
     * e == 0x3ff.
     *
     * We inevitably get some overflows in approximating the log
     * curves by these nasty step functions, but that's all right -
     * we do want _some_ overflows to be tested.
     *
     * Having got that, then, it's just a matter of inventing a
     * probability distribution for all of this.
     */
    int e, n;
    uint32 dmin, dmax;
    const uint32 pmin = 0x3e100000;

    /*
     * Generate exponents in a slightly biased fashion.
     */
    e = (random_upto(1) ?              /* is exponent small or big? */
         0x3FE - random_upto_biased(0x431,2) :   /* small */
         0x3FF + random_upto_biased(0x3FF,2));   /* big */

    /*
     * Now split into cases.
     */
    if (e < 0x3FE || e > 0x3FF) {
        uint32 imin, imax;
        if (e < 0x3FE)
            imin = 0x40000 / (0x3FE - e), imax = 0x43200 / (0x3FE - e);
        else
            imin = 0x43200 / (e - 0x3FF), imax = 0x40000 / (e - 0x3FF);
        /* Power range runs from -imin to imax. Now convert to doubles */
        dmin = doubletop(imin, -8);
        dmax = doubletop(imax, -8);
        /* Compute the number of mantissa bits. */
        n = (e > 0 ? 53 : 52+e);
    } else {
        /* Critical exponents. Generate a top bit index. */
        n = 52 - random_upto_biased(51, 4);
        if (e == 0x3FE)
            dmax = 63 - n;
        else
            dmax = 62 - n;
        dmax = (dmax << 20) + 0x3FF00000;
        dmin = dmax;
    }
    /* Generate a mantissa. */
    if (n <= 32) {
        out[0] = 0;
        out[1] = random_upto((1 << (n-1)) - 1) + (1 << (n-1));
    } else if (n == 33) {
        out[0] = 1;
        out[1] = random_upto(0xFFFFFFFF);
    } else if (n > 33) {
        out[0] = random_upto((1 << (n-33)) - 1) + (1 << (n-33));
        out[1] = random_upto(0xFFFFFFFF);
    }
    /* Negate the mantissa if e == 0x3FE. */
    if (e == 0x3FE) {
        out[1] = -out[1];
        out[0] = -out[0];
        if (out[1]) out[0]--;
    }
    /* Put the exponent on. */
    out[0] &= 0xFFFFF;
    out[0] |= ((e > 0 ? e : 0) << 20);
    /* Generate a power. Powers don't go below 2^-30. */
    if (random_upto(1)) {
        /* Positive power */
        out[2] = dmax - random_upto_biased(dmax-pmin, 10);
    } else {
        /* Negative power */
        out[2] = (dmin - random_upto_biased(dmin-pmin, 10)) | 0x80000000;
    }
    out[3] = random_upto(0xFFFFFFFF);
}
static void pow_cases_float(uint32 *out, uint32 param1,
                            uint32 param2) {
    /*
     * Pick an exponent e (-0x16 to +0xFE) for x, and here's the
     * range of numbers we can use as y:
     *
     * For e < 0x7E, the range is [-0x80/(0x7E-e),+0x95/(0x7E-e)]
     * For e > 0x7F, the range is [-0x95/(e-0x7F),+0x80/(e-0x7F)]
     *
     * For e == 0x7E or e == 0x7F, the range gets infinite at one
     * end or the other, so we have to be cleverer: pick a number n
     * of useful bits in the mantissa (1 thru 23, so 1 must imply
     * 0x3f800001 whereas 23 is anything at least as big as
     * 0x3fc00000; for e == 0x7e, 1 necessarily means 0x3f7fffff
     * and 23 is anything at most as big as 0x3f400000). Then, as
     * it happens, a sensible maximum power is 2^(31-n) for e ==
     * 0x7e, and 2^(30-n) for e == 0x7f.
     *
     * We inevitably get some overflows in approximating the log
     * curves by these nasty step functions, but that's all right -
     * we do want _some_ overflows to be tested.
     *
     * Having got that, then, it's just a matter of inventing a
     * probability distribution for all of this.
     */
    int e, n;
    uint32 dmin, dmax;
    const uint32 pmin = 0x38000000;

    /*
     * Generate exponents in a slightly biased fashion.
     */
    e = (random_upto(1) ?              /* is exponent small or big? */
         0x7E - random_upto_biased(0x94,2) :   /* small */
         0x7F + random_upto_biased(0x7f,2));   /* big */

    /*
     * Now split into cases.
     */
    if (e < 0x7E || e > 0x7F) {
        uint32 imin, imax;
        if (e < 0x7E)
            imin = 0x8000 / (0x7e - e), imax = 0x9500 / (0x7e - e);
        else
            imin = 0x9500 / (e - 0x7f), imax = 0x8000 / (e - 0x7f);
        /* Power range runs from -imin to imax. Now convert to doubles */
        dmin = floatval(imin, -8);
        dmax = floatval(imax, -8);
        /* Compute the number of mantissa bits. */
        n = (e > 0 ? 24 : 23+e);
    } else {
        /* Critical exponents. Generate a top bit index. */
        n = 23 - random_upto_biased(22, 4);
        if (e == 0x7E)
            dmax = 31 - n;
        else
            dmax = 30 - n;
        dmax = (dmax << 23) + 0x3F800000;
        dmin = dmax;
    }
    /* Generate a mantissa. */
    out[0] = random_upto((1 << (n-1)) - 1) + (1 << (n-1));
    out[1] = 0;
    /* Negate the mantissa if e == 0x7E. */
    if (e == 0x7E) {
        out[0] = -out[0];
    }
    /* Put the exponent on. */
    out[0] &= 0x7FFFFF;
    out[0] |= ((e > 0 ? e : 0) << 23);
    /* Generate a power. Powers don't go below 2^-15. */
    if (random_upto(1)) {
        /* Positive power */
        out[2] = dmax - random_upto_biased(dmax-pmin, 10);
    } else {
        /* Negative power */
        out[2] = (dmin - random_upto_biased(dmin-pmin, 10)) | 0x80000000;
    }
    out[3] = 0;
}

void vet_for_decline(Testable *fn, uint32 *args, uint32 *result, int got_errno_in) {
    int declined = 0;

    switch (fn->type) {
      case args1:
      case rred:
      case semi1:
      case t_frexp:
      case t_modf:
      case classify:
      case t_ldexp:
        declined |= lib_fo && is_dhard(args+0);
        break;
      case args1f:
      case rredf:
      case semi1f:
      case t_frexpf:
      case t_modff:
      case classifyf:
        declined |= lib_fo && is_shard(args+0);
        break;
      case args2:
      case semi2:
      case args1c:
      case args1cr:
      case compare:
        declined |= lib_fo && is_dhard(args+0);
        declined |= lib_fo && is_dhard(args+2);
        break;
      case args2f:
      case semi2f:
      case t_ldexpf:
      case comparef:
      case args1fc:
      case args1fcr:
        declined |= lib_fo && is_shard(args+0);
        declined |= lib_fo && is_shard(args+2);
        break;
      case args2c:
        declined |= lib_fo && is_dhard(args+0);
        declined |= lib_fo && is_dhard(args+2);
        declined |= lib_fo && is_dhard(args+4);
        declined |= lib_fo && is_dhard(args+6);
        break;
      case args2fc:
        declined |= lib_fo && is_shard(args+0);
        declined |= lib_fo && is_shard(args+2);
        declined |= lib_fo && is_shard(args+4);
        declined |= lib_fo && is_shard(args+6);
        break;
    }

    switch (fn->type) {
      case args1:              /* return an extra-precise result */
      case args2:
      case rred:
      case semi1:              /* return a double result */
      case semi2:
      case t_ldexp:
      case t_frexp:            /* return double * int */
      case args1cr:
        declined |= lib_fo && is_dhard(result);
        break;
      case args1f:
      case args2f:
      case rredf:
      case semi1f:
      case semi2f:
      case t_ldexpf:
      case args1fcr:
        declined |= lib_fo && is_shard(result);
        break;
      case t_modf:             /* return double * double */
        declined |= lib_fo && is_dhard(result+0);
        declined |= lib_fo && is_dhard(result+2);
        break;
      case t_modff:                    /* return float * float */
        declined |= lib_fo && is_shard(result+2);
        /* fall through */
      case t_frexpf:                   /* return float * int */
        declined |= lib_fo && is_shard(result+0);
        break;
      case args1c:
      case args2c:
        declined |= lib_fo && is_dhard(result+0);
        declined |= lib_fo && is_dhard(result+4);
        break;
      case args1fc:
      case args2fc:
        declined |= lib_fo && is_shard(result+0);
        declined |= lib_fo && is_shard(result+4);
        break;
    }

    /* Expect basic arithmetic tests to be declined if the command
     * line said that would happen */
    declined |= (lib_no_arith && (fn->func == (funcptr)mpc_add ||
                                  fn->func == (funcptr)mpc_sub ||
                                  fn->func == (funcptr)mpc_mul ||
                                  fn->func == (funcptr)mpc_div));

    if (!declined) {
        if (got_errno_in)
            ntests++;
        else
            ntests += 3;
    }
}

void docase(Testable *fn, uint32 *args) {
    uint32 result[8];  /* real part in first 4, imaginary part in last 4 */
    char *errstr = NULL;
    mpfr_t a, b, r;
    mpc_t ac, bc, rc;
    int rejected, printextra;
    wrapperctx ctx;

    mpfr_init2(a, MPFR_PREC);
    mpfr_init2(b, MPFR_PREC);
    mpfr_init2(r, MPFR_PREC);
    mpc_init2(ac, MPFR_PREC);
    mpc_init2(bc, MPFR_PREC);
    mpc_init2(rc, MPFR_PREC);

    printf("func=%s", fn->name);

    rejected = 0; /* FIXME */

    switch (fn->type) {
      case args1:
      case rred:
      case semi1:
      case t_frexp:
      case t_modf:
      case classify:
        printf(" op1=%08x.%08x", args[0], args[1]);
        break;
      case args1f:
      case rredf:
      case semi1f:
      case t_frexpf:
      case t_modff:
      case classifyf:
        printf(" op1=%08x", args[0]);
        break;
      case args2:
      case semi2:
      case compare:
        printf(" op1=%08x.%08x", args[0], args[1]);
        printf(" op2=%08x.%08x", args[2], args[3]);
        break;
      case args2f:
      case semi2f:
      case t_ldexpf:
      case comparef:
        printf(" op1=%08x", args[0]);
        printf(" op2=%08x", args[2]);
        break;
      case t_ldexp:
        printf(" op1=%08x.%08x", args[0], args[1]);
        printf(" op2=%08x", args[2]);
        break;
      case args1c:
      case args1cr:
        printf(" op1r=%08x.%08x", args[0], args[1]);
        printf(" op1i=%08x.%08x", args[2], args[3]);
        break;
      case args2c:
        printf(" op1r=%08x.%08x", args[0], args[1]);
        printf(" op1i=%08x.%08x", args[2], args[3]);
        printf(" op2r=%08x.%08x", args[4], args[5]);
        printf(" op2i=%08x.%08x", args[6], args[7]);
        break;
      case args1fc:
      case args1fcr:
        printf(" op1r=%08x", args[0]);
        printf(" op1i=%08x", args[2]);
        break;
      case args2fc:
        printf(" op1r=%08x", args[0]);
        printf(" op1i=%08x", args[2]);
        printf(" op2r=%08x", args[4]);
        printf(" op2i=%08x", args[6]);
        break;
      default:
        fprintf(stderr, "internal inconsistency?!\n");
        abort();
    }

    if (rejected == 2) {
        printf(" - test case rejected\n");
        goto cleanup;
    }

    wrapper_init(&ctx);

    if (rejected == 0) {
        switch (fn->type) {
          case args1:
            set_mpfr_d(a, args[0], args[1]);
            wrapper_op_real(&ctx, a, 2, args);
            ((testfunc1)(fn->func))(r, a, GMP_RNDN);
            get_mpfr_d(r, &result[0], &result[1], &result[2]);
            wrapper_result_real(&ctx, r, 2, result);
            if (wrapper_run(&ctx, fn->wrappers))
                get_mpfr_d(r, &result[0], &result[1], &result[2]);
            break;
          case args1cr:
            set_mpc_d(ac, args[0], args[1], args[2], args[3]);
            wrapper_op_complex(&ctx, ac, 2, args);
            ((testfunc1cr)(fn->func))(r, ac, GMP_RNDN);
            get_mpfr_d(r, &result[0], &result[1], &result[2]);
            wrapper_result_real(&ctx, r, 2, result);
            if (wrapper_run(&ctx, fn->wrappers))
                get_mpfr_d(r, &result[0], &result[1], &result[2]);
            break;
          case args1f:
            set_mpfr_f(a, args[0]);
            wrapper_op_real(&ctx, a, 1, args);
            ((testfunc1)(fn->func))(r, a, GMP_RNDN);
            get_mpfr_f(r, &result[0], &result[1]);
            wrapper_result_real(&ctx, r, 1, result);
            if (wrapper_run(&ctx, fn->wrappers))
                get_mpfr_f(r, &result[0], &result[1]);
            break;
          case args1fcr:
            set_mpc_f(ac, args[0], args[2]);
            wrapper_op_complex(&ctx, ac, 1, args);
            ((testfunc1cr)(fn->func))(r, ac, GMP_RNDN);
            get_mpfr_f(r, &result[0], &result[1]);
            wrapper_result_real(&ctx, r, 1, result);
            if (wrapper_run(&ctx, fn->wrappers))
                get_mpfr_f(r, &result[0], &result[1]);
            break;
          case args2:
            set_mpfr_d(a, args[0], args[1]);
            wrapper_op_real(&ctx, a, 2, args);
            set_mpfr_d(b, args[2], args[3]);
            wrapper_op_real(&ctx, b, 2, args+2);
            ((testfunc2)(fn->func))(r, a, b, GMP_RNDN);
            get_mpfr_d(r, &result[0], &result[1], &result[2]);
            wrapper_result_real(&ctx, r, 2, result);
            if (wrapper_run(&ctx, fn->wrappers))
                get_mpfr_d(r, &result[0], &result[1], &result[2]);
            break;
          case args2f:
            set_mpfr_f(a, args[0]);
            wrapper_op_real(&ctx, a, 1, args);
            set_mpfr_f(b, args[2]);
            wrapper_op_real(&ctx, b, 1, args+2);
            ((testfunc2)(fn->func))(r, a, b, GMP_RNDN);
            get_mpfr_f(r, &result[0], &result[1]);
            wrapper_result_real(&ctx, r, 1, result);
            if (wrapper_run(&ctx, fn->wrappers))
                get_mpfr_f(r, &result[0], &result[1]);
            break;
          case rred:
            set_mpfr_d(a, args[0], args[1]);
            wrapper_op_real(&ctx, a, 2, args);
            ((testrred)(fn->func))(r, a, (int *)&result[3]);
            get_mpfr_d(r, &result[0], &result[1], &result[2]);
            wrapper_result_real(&ctx, r, 2, result);
            /* We never need to mess about with the integer auxiliary
             * output. */
            if (wrapper_run(&ctx, fn->wrappers))
                get_mpfr_d(r, &result[0], &result[1], &result[2]);
            break;
          case rredf:
            set_mpfr_f(a, args[0]);
            wrapper_op_real(&ctx, a, 1, args);
            ((testrred)(fn->func))(r, a, (int *)&result[3]);
            get_mpfr_f(r, &result[0], &result[1]);
            wrapper_result_real(&ctx, r, 1, result);
            /* We never need to mess about with the integer auxiliary
             * output. */
            if (wrapper_run(&ctx, fn->wrappers))
                get_mpfr_f(r, &result[0], &result[1]);
            break;
          case semi1:
          case semi1f:
            errstr = ((testsemi1)(fn->func))(args, result);
            break;
          case semi2:
          case compare:
            errstr = ((testsemi2)(fn->func))(args, args+2, result);
            break;
          case semi2f:
          case comparef:
          case t_ldexpf:
            errstr = ((testsemi2f)(fn->func))(args, args+2, result);
            break;
          case t_ldexp:
            errstr = ((testldexp)(fn->func))(args, args+2, result);
            break;
          case t_frexp:
            errstr = ((testfrexp)(fn->func))(args, result, result+2);
            break;
          case t_frexpf:
            errstr = ((testfrexp)(fn->func))(args, result, result+2);
            break;
          case t_modf:
            errstr = ((testmodf)(fn->func))(args, result, result+2);
            break;
          case t_modff:
            errstr = ((testmodf)(fn->func))(args, result, result+2);
            break;
          case classify:
            errstr = ((testclassify)(fn->func))(args, &result[0]);
            break;
          case classifyf:
            errstr = ((testclassifyf)(fn->func))(args, &result[0]);
            break;
          case args1c:
            set_mpc_d(ac, args[0], args[1], args[2], args[3]);
            wrapper_op_complex(&ctx, ac, 2, args);
            ((testfunc1c)(fn->func))(rc, ac, MPC_RNDNN);
            get_mpc_d(rc, &result[0], &result[1], &result[2], &result[4], &result[5], &result[6]);
            wrapper_result_complex(&ctx, rc, 2, result);
            if (wrapper_run(&ctx, fn->wrappers))
                get_mpc_d(rc, &result[0], &result[1], &result[2], &result[4], &result[5], &result[6]);
            break;
          case args2c:
            set_mpc_d(ac, args[0], args[1], args[2], args[3]);
            wrapper_op_complex(&ctx, ac, 2, args);
            set_mpc_d(bc, args[4], args[5], args[6], args[7]);
            wrapper_op_complex(&ctx, bc, 2, args+4);
            ((testfunc2c)(fn->func))(rc, ac, bc, MPC_RNDNN);
            get_mpc_d(rc, &result[0], &result[1], &result[2], &result[4], &result[5], &result[6]);
            wrapper_result_complex(&ctx, rc, 2, result);
            if (wrapper_run(&ctx, fn->wrappers))
                get_mpc_d(rc, &result[0], &result[1], &result[2], &result[4], &result[5], &result[6]);
            break;
          case args1fc:
            set_mpc_f(ac, args[0], args[2]);
            wrapper_op_complex(&ctx, ac, 1, args);
            ((testfunc1c)(fn->func))(rc, ac, MPC_RNDNN);
            get_mpc_f(rc, &result[0], &result[1], &result[4], &result[5]);
            wrapper_result_complex(&ctx, rc, 1, result);
            if (wrapper_run(&ctx, fn->wrappers))
                get_mpc_f(rc, &result[0], &result[1], &result[4], &result[5]);
            break;
          case args2fc:
            set_mpc_f(ac, args[0], args[2]);
            wrapper_op_complex(&ctx, ac, 1, args);
            set_mpc_f(bc, args[4], args[6]);
            wrapper_op_complex(&ctx, bc, 1, args+4);
            ((testfunc2c)(fn->func))(rc, ac, bc, MPC_RNDNN);
            get_mpc_f(rc, &result[0], &result[1], &result[4], &result[5]);
            wrapper_result_complex(&ctx, rc, 1, result);
            if (wrapper_run(&ctx, fn->wrappers))
                get_mpc_f(rc, &result[0], &result[1], &result[4], &result[5]);
            break;
          default:
            fprintf(stderr, "internal inconsistency?!\n");
            abort();
        }
    }

    switch (fn->type) {
      case args1:              /* return an extra-precise result */
      case args2:
      case args1cr:
      case rred:
        printextra = 1;
        if (rejected == 0) {
            errstr = NULL;
            if (!mpfr_zero_p(a)) {
                if ((result[0] & 0x7FFFFFFF) == 0 && result[1] == 0) {
                    /*
                     * If the output is +0 or -0 apart from the extra
                     * precision in result[2], then there's a tricky
                     * judgment call about what we require in the
                     * output. If we output the extra bits and set
                     * errstr="?underflow" then mathtest will tolerate
                     * the function under test rounding down to zero
                     * _or_ up to the minimum denormal; whereas if we
                     * suppress the extra bits and set
                     * errstr="underflow", then mathtest will enforce
                     * that the function really does underflow to zero.
                     *
                     * But where to draw the line? It seems clear to
                     * me that numbers along the lines of
                     * 00000000.00000000.7ff should be treated
                     * similarly to 00000000.00000000.801, but on the
                     * other hand, we must surely be prepared to
                     * enforce a genuine underflow-to-zero in _some_
                     * case where the true mathematical output is
                     * nonzero but absurdly tiny.
                     *
                     * I think a reasonable place to draw the
                     * distinction is at 00000000.00000000.400, i.e.
                     * one quarter of the minimum positive denormal.
                     * If a value less than that rounds up to the
                     * minimum denormal, that must mean the function
                     * under test has managed to make an error of an
                     * entire factor of two, and that's something we
                     * should fix. Above that, you can misround within
                     * the limits of your accuracy bound if you have
                     * to.
                     */
                    if (result[2] < 0x40000000) {
                        /* Total underflow (ERANGE + UFL) is required,
                         * and we suppress the extra bits to make
                         * mathtest enforce that the output is really
                         * zero. */
                        errstr = "underflow";
                        printextra = 0;
                    } else {
                        /* Total underflow is not required, but if the
                         * function rounds down to zero anyway, then
                         * we should be prepared to tolerate it. */
                        errstr = "?underflow";
                    }
                } else if (!(result[0] & 0x7ff00000)) {
                    /*
                     * If the output is denormal, we usually expect a
                     * UFL exception, warning the user of partial
                     * underflow. The exception is if the denormal
                     * being returned is just one of the input values,
                     * unchanged even in principle. I bodgily handle
                     * this by just special-casing the functions in
                     * question below.
                     */
                    if (!strcmp(fn->name, "fmax") ||
                        !strcmp(fn->name, "fmin") ||
                        !strcmp(fn->name, "creal") ||
                        !strcmp(fn->name, "cimag")) {
                        /* no error expected */
                    } else {
                        errstr = "u";
                    }
                } else if ((result[0] & 0x7FFFFFFF) > 0x7FEFFFFF) {
                    /*
                     * Infinite results are usually due to overflow,
                     * but one exception is lgamma of a negative
                     * integer.
                     */
                    if (!strcmp(fn->name, "lgamma") &&
                        (args[0] & 0x80000000) != 0 && /* negative */
                        is_dinteger(args)) {
                        errstr = "ERANGE status=z";
                    } else {
                        errstr = "overflow";
                    }
                    printextra = 0;
                }
            } else {
                /* lgamma(0) is also a pole. */
                if (!strcmp(fn->name, "lgamma")) {
                    errstr = "ERANGE status=z";
                    printextra = 0;
                }
            }
        }

        if (!printextra || (rejected && !(rejected==1 && result[2]!=0))) {
            printf(" result=%08x.%08x",
                   result[0], result[1]);
        } else {
            printf(" result=%08x.%08x.%03x",
                   result[0], result[1], (result[2] >> 20) & 0xFFF);
        }
        if (fn->type == rred) {
            printf(" res2=%08x", result[3]);
        }
        break;
      case args1f:
      case args2f:
      case args1fcr:
      case rredf:
        printextra = 1;
        if (rejected == 0) {
            errstr = NULL;
            if (!mpfr_zero_p(a)) {
                if ((result[0] & 0x7FFFFFFF) == 0) {
                    /*
                     * Decide whether to print the extra bits based on
                     * just how close to zero the number is. See the
                     * big comment in the double-precision case for
                     * discussion.
                     */
                    if (result[1] < 0x40000000) {
                        errstr = "underflow";
                        printextra = 0;
                    } else {
                        errstr = "?underflow";
                    }
                } else if (!(result[0] & 0x7f800000)) {
                    /*
                     * Functions which do not report partial overflow
                     * are listed here as special cases. (See the
                     * corresponding double case above for a fuller
                     * comment.)
                     */
                    if (!strcmp(fn->name, "fmaxf") ||
                        !strcmp(fn->name, "fminf") ||
                        !strcmp(fn->name, "crealf") ||
                        !strcmp(fn->name, "cimagf")) {
                        /* no error expected */
                    } else {
                        errstr = "u";
                    }
                } else if ((result[0] & 0x7FFFFFFF) > 0x7F7FFFFF) {
                    /*
                     * Infinite results are usually due to overflow,
                     * but one exception is lgamma of a negative
                     * integer.
                     */
                    if (!strcmp(fn->name, "lgammaf") &&
                        (args[0] & 0x80000000) != 0 && /* negative */
                        is_sinteger(args)) {
                        errstr = "ERANGE status=z";
                    } else {
                        errstr = "overflow";
                    }
                    printextra = 0;
                }
            } else {
                /* lgamma(0) is also a pole. */
                if (!strcmp(fn->name, "lgammaf")) {
                    errstr = "ERANGE status=z";
                    printextra = 0;
                }
            }
        }

        if (!printextra || (rejected && !(rejected==1 && result[1]!=0))) {
            printf(" result=%08x",
                   result[0]);
        } else {
            printf(" result=%08x.%03x",
                   result[0], (result[1] >> 20) & 0xFFF);
        }
        if (fn->type == rredf) {
            printf(" res2=%08x", result[3]);
        }
        break;
      case semi1:              /* return a double result */
      case semi2:
      case t_ldexp:
        printf(" result=%08x.%08x", result[0], result[1]);
        break;
      case semi1f:
      case semi2f:
      case t_ldexpf:
        printf(" result=%08x", result[0]);
        break;
      case t_frexp:            /* return double * int */
        printf(" result=%08x.%08x res2=%08x", result[0], result[1],
               result[2]);
        break;
      case t_modf:             /* return double * double */
        printf(" result=%08x.%08x res2=%08x.%08x",
               result[0], result[1], result[2], result[3]);
        break;
      case t_modff:                    /* return float * float */
        /* fall through */
      case t_frexpf:                   /* return float * int */
        printf(" result=%08x res2=%08x", result[0], result[2]);
        break;
      case classify:
      case classifyf:
      case compare:
      case comparef:
        printf(" result=%x", result[0]);
        break;
      case args1c:
      case args2c:
        if (0/* errstr */) {
            printf(" resultr=%08x.%08x", result[0], result[1]);
            printf(" resulti=%08x.%08x", result[4], result[5]);
        } else {
            printf(" resultr=%08x.%08x.%03x",
                   result[0], result[1], (result[2] >> 20) & 0xFFF);
            printf(" resulti=%08x.%08x.%03x",
                   result[4], result[5], (result[6] >> 20) & 0xFFF);
        }
        /* Underflow behaviour doesn't seem to be specified for complex arithmetic */
        errstr = "?underflow";
        break;
      case args1fc:
      case args2fc:
        if (0/* errstr */) {
            printf(" resultr=%08x", result[0]);
            printf(" resulti=%08x", result[4]);
        } else {
            printf(" resultr=%08x.%03x",
                   result[0], (result[1] >> 20) & 0xFFF);
            printf(" resulti=%08x.%03x",
                   result[4], (result[5] >> 20) & 0xFFF);
        }
        /* Underflow behaviour doesn't seem to be specified for complex arithmetic */
        errstr = "?underflow";
        break;
    }

    if (errstr && *(errstr+1) == '\0') {
        printf(" errno=0 status=%c",*errstr);
    } else if (errstr && *errstr == '?') {
        printf(" maybeerror=%s", errstr+1);
    } else if (errstr && errstr[0] == 'E') {
        printf(" errno=%s", errstr);
    } else {
        printf(" error=%s", errstr && *errstr ? errstr : "0");
    }

    printf("\n");

    vet_for_decline(fn, args, result, 0);

  cleanup:
    mpfr_clear(a);
    mpfr_clear(b);
    mpfr_clear(r);
    mpc_clear(ac);
    mpc_clear(bc);
    mpc_clear(rc);
}

void gencases(Testable *fn, int number) {
    int i;
    uint32 args[8];

    float32_case(NULL);
    float64_case(NULL);

    printf("random=on\n"); /* signal to runtests.pl that the following tests are randomly generated */
    for (i = 0; i < number; i++) {
        /* generate test point */
        fn->cases(args, fn->caseparam1, fn->caseparam2);
        docase(fn, args);
    }
    printf("random=off\n");
}

static uint32 doubletop(int x, int scale) {
    int e = 0x412 + scale;
    while (!(x & 0x100000))
        x <<= 1, e--;
    return (e << 20) + x;
}

static uint32 floatval(int x, int scale) {
    int e = 0x95 + scale;
    while (!(x & 0x800000))
        x <<= 1, e--;
    return (e << 23) + x;
}
