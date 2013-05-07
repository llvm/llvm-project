//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <math.h>

#include <math.h>
#include <type_traits>
#include <cassert>

#include "hexfloat.h"

void test_acos()
{
    static_assert((std::is_same<decltype(acos((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(acosf(0)), float>::value), "");
    static_assert((std::is_same<decltype(acosl(0)), long double>::value), "");
    assert(acos(1) == 0);
}

void test_asin()
{
    static_assert((std::is_same<decltype(asin((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(asinf(0)), float>::value), "");
    static_assert((std::is_same<decltype(asinl(0)), long double>::value), "");
    assert(asin(0) == 0);
}

void test_atan()
{
    static_assert((std::is_same<decltype(atan((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(atanf(0)), float>::value), "");
    static_assert((std::is_same<decltype(atanl(0)), long double>::value), "");
    assert(atan(0) == 0);
}

void test_atan2()
{
    static_assert((std::is_same<decltype(atan2((double)0, (double)0)), double>::value), "");
    static_assert((std::is_same<decltype(atan2f(0,0)), float>::value), "");
    static_assert((std::is_same<decltype(atan2l(0,0)), long double>::value), "");
    assert(atan2(0,1) == 0);
}

void test_ceil()
{
    static_assert((std::is_same<decltype(ceil((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(ceilf(0)), float>::value), "");
    static_assert((std::is_same<decltype(ceill(0)), long double>::value), "");
    assert(ceil(0) == 0);
}

void test_cos()
{
    static_assert((std::is_same<decltype(cos((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(cosf(0)), float>::value), "");
    static_assert((std::is_same<decltype(cosl(0)), long double>::value), "");
    assert(cos(0) == 1);
}

void test_cosh()
{
    static_assert((std::is_same<decltype(cosh((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(coshf(0)), float>::value), "");
    static_assert((std::is_same<decltype(coshl(0)), long double>::value), "");
    assert(cosh(0) == 1);
}

void test_exp()
{
    static_assert((std::is_same<decltype(exp((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(expf(0)), float>::value), "");
    static_assert((std::is_same<decltype(expl(0)), long double>::value), "");
    assert(exp(0) == 1);
}

void test_fabs()
{
    static_assert((std::is_same<decltype(fabs((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(fabsf(0)), float>::value), "");
    static_assert((std::is_same<decltype(fabsl(0)), long double>::value), "");
    assert(fabs(-1) == 1);
}

void test_floor()
{
    static_assert((std::is_same<decltype(floor((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(floorf(0)), float>::value), "");
    static_assert((std::is_same<decltype(floorl(0)), long double>::value), "");
    assert(floor(1) == 1);
}

void test_fmod()
{
    static_assert((std::is_same<decltype(fmod((double)0, (double)0)), double>::value), "");
    static_assert((std::is_same<decltype(fmodf(0,0)), float>::value), "");
    static_assert((std::is_same<decltype(fmodl(0,0)), long double>::value), "");
    assert(fmod(1.5,1) == .5);
}

void test_frexp()
{
    int ip;
    static_assert((std::is_same<decltype(frexp((double)0, &ip)), double>::value), "");
    static_assert((std::is_same<decltype(frexpf(0, &ip)), float>::value), "");
    static_assert((std::is_same<decltype(frexpl(0, &ip)), long double>::value), "");
    assert(frexp(0, &ip) == 0);
}

void test_ldexp()
{
    int ip = 1;
    static_assert((std::is_same<decltype(ldexp((double)0, ip)), double>::value), "");
    static_assert((std::is_same<decltype(ldexpf(0, ip)), float>::value), "");
    static_assert((std::is_same<decltype(ldexpl(0, ip)), long double>::value), "");
    assert(ldexp(1, ip) == 2);
}

void test_log()
{
    static_assert((std::is_same<decltype(log((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(logf(0)), float>::value), "");
    static_assert((std::is_same<decltype(logl(0)), long double>::value), "");
    assert(log(1) == 0);
}

void test_log10()
{
    static_assert((std::is_same<decltype(log10((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(log10f(0)), float>::value), "");
    static_assert((std::is_same<decltype(log10l(0)), long double>::value), "");
    assert(log10(1) == 0);
}

void test_modf()
{
    static_assert((std::is_same<decltype(modf((double)0, (double*)0)), double>::value), "");
    static_assert((std::is_same<decltype(modff(0, (float*)0)), float>::value), "");
    static_assert((std::is_same<decltype(modfl(0, (long double*)0)), long double>::value), "");
    double i;
    assert(modf(1., &i) == 0);
}

void test_pow()
{
    static_assert((std::is_same<decltype(pow((double)0, (double)0)), double>::value), "");
    static_assert((std::is_same<decltype(powf(0,0)), float>::value), "");
    static_assert((std::is_same<decltype(powl(0,0)), long double>::value), "");
    assert(pow(1,1) == 1);
}

void test_sin()
{
    static_assert((std::is_same<decltype(sin((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(sinf(0)), float>::value), "");
    static_assert((std::is_same<decltype(sinl(0)), long double>::value), "");
    assert(sin(0) == 0);
}

void test_sinh()
{
    static_assert((std::is_same<decltype(sinh((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(sinhf(0)), float>::value), "");
    static_assert((std::is_same<decltype(sinhl(0)), long double>::value), "");
    assert(sinh(0) == 0);
}

void test_sqrt()
{
    static_assert((std::is_same<decltype(sqrt((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(sqrtf(0)), float>::value), "");
    static_assert((std::is_same<decltype(sqrtl(0)), long double>::value), "");
    assert(sqrt(4) == 2);
}

void test_tan()
{
    static_assert((std::is_same<decltype(tan((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(tanf(0)), float>::value), "");
    static_assert((std::is_same<decltype(tanl(0)), long double>::value), "");
    assert(tan(0) == 0);
}

void test_tanh()
{
    static_assert((std::is_same<decltype(tanh((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(tanhf(0)), float>::value), "");
    static_assert((std::is_same<decltype(tanhl(0)), long double>::value), "");
    assert(tanh(0) == 0);
}

void test_signbit()
{
    static_assert((std::is_same<decltype(signbit((float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(signbit((double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(signbit((long double)0)), bool>::value), "");
    assert(signbit(-1.0) == true);
}

void test_fpclassify()
{
    static_assert((std::is_same<decltype(fpclassify((float)0)), int>::value), "");
    static_assert((std::is_same<decltype(fpclassify((double)0)), int>::value), "");
    static_assert((std::is_same<decltype(fpclassify((long double)0)), int>::value), "");
    assert(fpclassify(-1.0) == FP_NORMAL);
}

void test_isfinite()
{
    static_assert((std::is_same<decltype(isfinite((float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isfinite((double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isfinite((long double)0)), bool>::value), "");
    assert(isfinite(-1.0) == true);
}

void test_isinf()
{
    static_assert((std::is_same<decltype(isinf((float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isinf((double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isinf((long double)0)), bool>::value), "");
    assert(isinf(-1.0) == false);
}

void test_isnan()
{
    static_assert((std::is_same<decltype(isnan((float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isnan((double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isnan((long double)0)), bool>::value), "");
    assert(isnan(-1.0) == false);
}

void test_isnormal()
{
    static_assert((std::is_same<decltype(isnormal((float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isnormal((double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isnormal((long double)0)), bool>::value), "");
    assert(isnormal(-1.0) == true);
}

void test_isgreater()
{
    static_assert((std::is_same<decltype(isgreater((float)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isgreater((float)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isgreater((float)0, (long double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isgreater((double)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isgreater((double)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isgreater((double)0, (long double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isgreater((long double)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isgreater((long double)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isgreater((long double)0, (long double)0)), bool>::value), "");
    assert(isgreater(-1.0, 0.F) == false);
}

void test_isgreaterequal()
{
    static_assert((std::is_same<decltype(isgreaterequal((float)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isgreaterequal((float)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isgreaterequal((float)0, (long double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isgreaterequal((double)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isgreaterequal((double)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isgreaterequal((double)0, (long double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isgreaterequal((long double)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isgreaterequal((long double)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isgreaterequal((long double)0, (long double)0)), bool>::value), "");
    assert(isgreaterequal(-1.0, 0.F) == false);
}

void test_isless()
{
    static_assert((std::is_same<decltype(isless((float)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isless((float)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isless((float)0, (long double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isless((double)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isless((double)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isless((double)0, (long double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isless((long double)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isless((long double)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isless((long double)0, (long double)0)), bool>::value), "");
    assert(isless(-1.0, 0.F) == true);
}

void test_islessequal()
{
    static_assert((std::is_same<decltype(islessequal((float)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(islessequal((float)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(islessequal((float)0, (long double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(islessequal((double)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(islessequal((double)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(islessequal((double)0, (long double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(islessequal((long double)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(islessequal((long double)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(islessequal((long double)0, (long double)0)), bool>::value), "");
    assert(islessequal(-1.0, 0.F) == true);
}

void test_islessgreater()
{
    static_assert((std::is_same<decltype(islessgreater((float)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(islessgreater((float)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(islessgreater((float)0, (long double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(islessgreater((double)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(islessgreater((double)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(islessgreater((double)0, (long double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(islessgreater((long double)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(islessgreater((long double)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(islessgreater((long double)0, (long double)0)), bool>::value), "");
    assert(islessgreater(-1.0, 0.F) == true);
}

void test_isunordered()
{
    static_assert((std::is_same<decltype(isunordered((float)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isunordered((float)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isunordered((float)0, (long double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isunordered((double)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isunordered((double)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isunordered((double)0, (long double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isunordered((long double)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isunordered((long double)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isunordered((long double)0, (long double)0)), bool>::value), "");
    assert(isunordered(-1.0, 0.F) == false);
}

void test_acosh()
{
    static_assert((std::is_same<decltype(acosh((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(acoshf(0)), float>::value), "");
    static_assert((std::is_same<decltype(acoshl(0)), long double>::value), "");
    assert(acosh(1) == 0);
}

void test_asinh()
{
    static_assert((std::is_same<decltype(asinh((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(asinhf(0)), float>::value), "");
    static_assert((std::is_same<decltype(asinhl(0)), long double>::value), "");
    assert(asinh(0) == 0);
}

void test_atanh()
{
    static_assert((std::is_same<decltype(atanh((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(atanhf(0)), float>::value), "");
    static_assert((std::is_same<decltype(atanhl(0)), long double>::value), "");
    assert(atanh(0) == 0);
}

void test_cbrt()
{
    static_assert((std::is_same<decltype(cbrt((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(cbrtf(0)), float>::value), "");
    static_assert((std::is_same<decltype(cbrtl(0)), long double>::value), "");
    assert(cbrt(1) == 1);
}

void test_copysign()
{
    static_assert((std::is_same<decltype(copysign((double)0, (double)0)), double>::value), "");
    static_assert((std::is_same<decltype(copysignf(0,0)), float>::value), "");
    static_assert((std::is_same<decltype(copysignl(0,0)), long double>::value), "");
    assert(copysign(1,1) == 1);
}

void test_erf()
{
    static_assert((std::is_same<decltype(erf((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(erff(0)), float>::value), "");
    static_assert((std::is_same<decltype(erfl(0)), long double>::value), "");
    assert(erf(0) == 0);
}

void test_erfc()
{
    static_assert((std::is_same<decltype(erfc((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(erfcf(0)), float>::value), "");
    static_assert((std::is_same<decltype(erfcl(0)), long double>::value), "");
    assert(erfc(0) == 1);
}

void test_exp2()
{
    static_assert((std::is_same<decltype(exp2((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(exp2f(0)), float>::value), "");
    static_assert((std::is_same<decltype(exp2l(0)), long double>::value), "");
    assert(exp2(1) == 2);
}

void test_expm1()
{
    static_assert((std::is_same<decltype(expm1((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(expm1f(0)), float>::value), "");
    static_assert((std::is_same<decltype(expm1l(0)), long double>::value), "");
    assert(expm1(0) == 0);
}

void test_fdim()
{
    static_assert((std::is_same<decltype(fdim((double)0, (double)0)), double>::value), "");
    static_assert((std::is_same<decltype(fdimf(0,0)), float>::value), "");
    static_assert((std::is_same<decltype(fdiml(0,0)), long double>::value), "");
    assert(fdim(1,0) == 1);
}

void test_fma()
{
    static_assert((std::is_same<decltype(fma((double)0, (double)0,  (double)0)), double>::value), "");
    static_assert((std::is_same<decltype(fmaf(0,0,0)), float>::value), "");
    static_assert((std::is_same<decltype(fmal(0,0,0)), long double>::value), "");
    assert(fma(1,1,1) == 2);
}

void test_fmax()
{
    static_assert((std::is_same<decltype(fmax((double)0, (double)0)), double>::value), "");
    static_assert((std::is_same<decltype(fmaxf(0,0)), float>::value), "");
    static_assert((std::is_same<decltype(fmaxl(0,0)), long double>::value), "");
    assert(fmax(1,0) == 1);
}

void test_fmin()
{
    static_assert((std::is_same<decltype(fmin((double)0, (double)0)), double>::value), "");
    static_assert((std::is_same<decltype(fminf(0,0)), float>::value), "");
    static_assert((std::is_same<decltype(fminl(0,0)), long double>::value), "");
    assert(fmin(1,0) == 0);
}

void test_hypot()
{
    static_assert((std::is_same<decltype(hypot((double)0, (double)0)), double>::value), "");
    static_assert((std::is_same<decltype(hypotf(0,0)), float>::value), "");
    static_assert((std::is_same<decltype(hypotl(0,0)), long double>::value), "");
    assert(hypot(3,4) == 5);
}

void test_ilogb()
{
    static_assert((std::is_same<decltype(ilogb((double)0)), int>::value), "");
    static_assert((std::is_same<decltype(ilogbf(0)), int>::value), "");
    static_assert((std::is_same<decltype(ilogbl(0)), int>::value), "");
    assert(ilogb(1) == 0);
}

void test_lgamma()
{
    static_assert((std::is_same<decltype(lgamma((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(lgammaf(0)), float>::value), "");
    static_assert((std::is_same<decltype(lgammal(0)), long double>::value), "");
    assert(lgamma(1) == 0);
}

void test_llrint()
{
    static_assert((std::is_same<decltype(llrint((double)0)), long long>::value), "");
    static_assert((std::is_same<decltype(llrintf(0)), long long>::value), "");
    static_assert((std::is_same<decltype(llrintl(0)), long long>::value), "");
    assert(llrint(1) == 1LL);
}

void test_llround()
{
    static_assert((std::is_same<decltype(llround((double)0)), long long>::value), "");
    static_assert((std::is_same<decltype(llroundf(0)), long long>::value), "");
    static_assert((std::is_same<decltype(llroundl(0)), long long>::value), "");
    assert(llround(1) == 1LL);
}

void test_log1p()
{
    static_assert((std::is_same<decltype(log1p((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(log1pf(0)), float>::value), "");
    static_assert((std::is_same<decltype(log1pl(0)), long double>::value), "");
    assert(log1p(0) == 0);
}

void test_log2()
{
    static_assert((std::is_same<decltype(log2((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(log2f(0)), float>::value), "");
    static_assert((std::is_same<decltype(log2l(0)), long double>::value), "");
    assert(log2(1) == 0);
}

void test_logb()
{
    static_assert((std::is_same<decltype(logb((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(logbf(0)), float>::value), "");
    static_assert((std::is_same<decltype(logbl(0)), long double>::value), "");
    assert(logb(1) == 0);
}

void test_lrint()
{
    static_assert((std::is_same<decltype(lrint((double)0)), long>::value), "");
    static_assert((std::is_same<decltype(lrintf(0)), long>::value), "");
    static_assert((std::is_same<decltype(lrintl(0)), long>::value), "");
    assert(lrint(1) == 1L);
}

void test_lround()
{
    static_assert((std::is_same<decltype(lround((double)0)), long>::value), "");
    static_assert((std::is_same<decltype(lroundf(0)), long>::value), "");
    static_assert((std::is_same<decltype(lroundl(0)), long>::value), "");
    assert(lround(1) == 1L);
}

void test_nan()
{
    static_assert((std::is_same<decltype(nan("")), double>::value), "");
    static_assert((std::is_same<decltype(nanf("")), float>::value), "");
    static_assert((std::is_same<decltype(nanl("")), long double>::value), "");
}

void test_nearbyint()
{
    static_assert((std::is_same<decltype(nearbyint((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(nearbyintf(0)), float>::value), "");
    static_assert((std::is_same<decltype(nearbyintl(0)), long double>::value), "");
    assert(nearbyint(1) == 1);
}

void test_nextafter()
{
    static_assert((std::is_same<decltype(nextafter((double)0, (double)0)), double>::value), "");
    static_assert((std::is_same<decltype(nextafterf(0,0)), float>::value), "");
    static_assert((std::is_same<decltype(nextafterl(0,0)), long double>::value), "");
    assert(nextafter(0,1) == hexfloat<double>(0x1, 0, -1074));
}

void test_nexttoward()
{
    static_assert((std::is_same<decltype(nexttoward((double)0, (long double)0)), double>::value), "");
    static_assert((std::is_same<decltype(nexttowardf(0, (long double)0)), float>::value), "");
    static_assert((std::is_same<decltype(nexttowardl(0, (long double)0)), long double>::value), "");
    assert(nexttoward(0, 1) == hexfloat<double>(0x1, 0, -1074));
}

void test_remainder()
{
    static_assert((std::is_same<decltype(remainder((double)0, (double)0)), double>::value), "");
    static_assert((std::is_same<decltype(remainderf(0,0)), float>::value), "");
    static_assert((std::is_same<decltype(remainderl(0,0)), long double>::value), "");
    static_assert((std::is_same<decltype(remainder((int)0, (int)0)), double>::value), "");
    assert(remainder(0.5,1) == 0.5);
}

void test_remquo()
{
    int ip;
    static_assert((std::is_same<decltype(remquo((double)0, (double)0, &ip)), double>::value), "");
    static_assert((std::is_same<decltype(remquof(0,0, &ip)), float>::value), "");
    static_assert((std::is_same<decltype(remquol(0,0, &ip)), long double>::value), "");
    assert(remquo(0.5,1, &ip) == 0.5);
}

void test_rint()
{
    static_assert((std::is_same<decltype(rint((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(rintf(0)), float>::value), "");
    static_assert((std::is_same<decltype(rintl(0)), long double>::value), "");
    assert(rint(1) == 1);
}

void test_round()
{
    static_assert((std::is_same<decltype(round((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(roundf(0)), float>::value), "");
    static_assert((std::is_same<decltype(roundl(0)), long double>::value), "");
    assert(round(1) == 1);
}

void test_scalbln()
{
    static_assert((std::is_same<decltype(scalbln((double)0, (long)0)), double>::value), "");
    static_assert((std::is_same<decltype(scalblnf(0, (long)0)), float>::value), "");
    static_assert((std::is_same<decltype(scalblnl(0, (long)0)), long double>::value), "");
    assert(scalbln(1, 1) == 2);
}

void test_scalbn()
{
    static_assert((std::is_same<decltype(scalbn((double)0, (int)0)), double>::value), "");
    static_assert((std::is_same<decltype(scalbnf(0, (int)0)), float>::value), "");
    static_assert((std::is_same<decltype(scalbnl(0, (int)0)), long double>::value), "");
    assert(scalbn(1, 1) == 2);
}

void test_tgamma()
{
    static_assert((std::is_same<decltype(tgamma((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(tgammaf(0)), float>::value), "");
    static_assert((std::is_same<decltype(tgammal(0)), long double>::value), "");
    assert(tgamma(1) == 1);
}

void test_trunc()
{
    static_assert((std::is_same<decltype(trunc((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(truncf(0)), float>::value), "");
    static_assert((std::is_same<decltype(truncl(0)), long double>::value), "");
    assert(trunc(1) == 1);
}

int main()
{
    test_acos();
    test_asin();
    test_atan();
    test_atan2();
    test_ceil();
    test_cos();
    test_cosh();
    test_exp();
    test_fabs();
    test_floor();
    test_fmod();
    test_frexp();
    test_ldexp();
    test_log();
    test_log10();
    test_modf();
    test_pow();
    test_sin();
    test_sinh();
    test_sqrt();
    test_tan();
    test_tanh();
    test_signbit();
    test_fpclassify();
    test_isfinite();
    test_isinf();
    test_isnan();
    test_isnormal();
    test_isgreater();
    test_isgreaterequal();
    test_isless();
    test_islessequal();
    test_islessgreater();
    test_isunordered();
    test_acosh();
    test_asinh();
    test_atanh();
    test_cbrt();
    test_copysign();
    test_erf();
    test_erfc();
    test_exp2();
    test_expm1();
    test_fdim();
    test_fma();
    test_fmax();
    test_fmin();
    test_hypot();
    test_ilogb();
    test_lgamma();
    test_llrint();
    test_llround();
    test_log1p();
    test_log2();
    test_logb();
    test_lrint();
    test_lround();
    test_nan();
    test_nearbyint();
    test_nextafter();
    test_nexttoward();
    test_remainder();
    test_remquo();
    test_rint();
    test_round();
    test_scalbln();
    test_scalbn();
    test_tgamma();
    test_trunc();
}
