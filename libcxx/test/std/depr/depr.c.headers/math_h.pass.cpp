//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <math.h>

#include <math.h>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "hexfloat.h"
#include "truncate_fp.h"
#include "type_algorithms.h"

// convertible to int/float/double/etc
template <class T, int N=0>
struct Value {
    operator T () { return T(N); }
};

// See PR21083
// Ambiguous is a user-defined type that defines its own overloads of cmath
// functions. When the std overloads are candidates too (by using or adl),
// they should not interfere.
struct Ambiguous : std::true_type { // ADL
    operator float () { return 0.f; }
    operator double () { return 0.; }
};
Ambiguous abs(Ambiguous){ return Ambiguous(); }
Ambiguous acos(Ambiguous){ return Ambiguous(); }
Ambiguous asin(Ambiguous){ return Ambiguous(); }
Ambiguous atan(Ambiguous){ return Ambiguous(); }
Ambiguous atan2(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous ceil(Ambiguous){ return Ambiguous(); }
Ambiguous cos(Ambiguous){ return Ambiguous(); }
Ambiguous cosh(Ambiguous){ return Ambiguous(); }
Ambiguous exp(Ambiguous){ return Ambiguous(); }
Ambiguous fabs(Ambiguous){ return Ambiguous(); }
Ambiguous floor(Ambiguous){ return Ambiguous(); }
Ambiguous fmod(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous frexp(Ambiguous, int*){ return Ambiguous(); }
Ambiguous ldexp(Ambiguous, int){ return Ambiguous(); }
Ambiguous log(Ambiguous){ return Ambiguous(); }
Ambiguous log10(Ambiguous){ return Ambiguous(); }
Ambiguous modf(Ambiguous, Ambiguous*){ return Ambiguous(); }
Ambiguous pow(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous sin(Ambiguous){ return Ambiguous(); }
Ambiguous sinh(Ambiguous){ return Ambiguous(); }
Ambiguous sqrt(Ambiguous){ return Ambiguous(); }
Ambiguous tan(Ambiguous){ return Ambiguous(); }
Ambiguous tanh(Ambiguous){ return Ambiguous(); }
Ambiguous signbit(Ambiguous){ return Ambiguous(); }
Ambiguous fpclassify(Ambiguous){ return Ambiguous(); }
Ambiguous isfinite(Ambiguous){ return Ambiguous(); }
Ambiguous isnormal(Ambiguous){ return Ambiguous(); }
Ambiguous isgreater(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous isgreaterequal(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous isless(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous islessequal(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous islessgreater(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous isunordered(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous acosh(Ambiguous){ return Ambiguous(); }
Ambiguous asinh(Ambiguous){ return Ambiguous(); }
Ambiguous atanh(Ambiguous){ return Ambiguous(); }
Ambiguous cbrt(Ambiguous){ return Ambiguous(); }
Ambiguous copysign(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous erf(Ambiguous){ return Ambiguous(); }
Ambiguous erfc(Ambiguous){ return Ambiguous(); }
Ambiguous exp2(Ambiguous){ return Ambiguous(); }
Ambiguous expm1(Ambiguous){ return Ambiguous(); }
Ambiguous fdim(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous fma(Ambiguous, Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous fmax(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous fmin(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous hypot(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous ilogb(Ambiguous){ return Ambiguous(); }
Ambiguous lgamma(Ambiguous){ return Ambiguous(); }
Ambiguous llrint(Ambiguous){ return Ambiguous(); }
Ambiguous llround(Ambiguous){ return Ambiguous(); }
Ambiguous log1p(Ambiguous){ return Ambiguous(); }
Ambiguous log2(Ambiguous){ return Ambiguous(); }
Ambiguous logb(Ambiguous){ return Ambiguous(); }
Ambiguous lrint(Ambiguous){ return Ambiguous(); }
Ambiguous lround(Ambiguous){ return Ambiguous(); }
Ambiguous nearbyint(Ambiguous){ return Ambiguous(); }
Ambiguous nextafter(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous nexttoward(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous remainder(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous remquo(Ambiguous, Ambiguous, int*){ return Ambiguous(); }
Ambiguous rint(Ambiguous){ return Ambiguous(); }
Ambiguous round(Ambiguous){ return Ambiguous(); }
Ambiguous scalbln(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous scalbn(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous tgamma(Ambiguous){ return Ambiguous(); }
Ambiguous trunc(Ambiguous){ return Ambiguous(); }

template <class T, class = decltype(::abs(std::declval<T>()))>
std::true_type has_abs_imp(int);
template <class T>
std::false_type has_abs_imp(...);

template <class T>
struct has_abs : decltype(has_abs_imp<T>(0)) {};

void test_abs()
{
  TEST_DIAGNOSTIC_PUSH
  TEST_CLANG_DIAGNOSTIC_IGNORED("-Wabsolute-value")

  static_assert((std::is_same<decltype(abs((float)0)), float>::value), "");
  static_assert((std::is_same<decltype(abs((double)0)), double>::value), "");
  static_assert(
      (std::is_same<decltype(abs((long double)0)), long double>::value), "");
  static_assert((std::is_same<decltype(abs((int)0)), int>::value), "");
  static_assert((std::is_same<decltype(abs((long)0)), long>::value), "");
  static_assert((std::is_same<decltype(abs((long long)0)), long long>::value),
                "");
  static_assert((std::is_same<decltype(abs((unsigned char)0)), int>::value),
                "");
  static_assert((std::is_same<decltype(abs((unsigned short)0)), int>::value),
                "");

  static_assert((std::is_same<decltype(abs(Ambiguous())), Ambiguous>::value),
                "");

  static_assert(!has_abs<unsigned>::value, "");
  static_assert(!has_abs<unsigned long>::value, "");
  static_assert(!has_abs<unsigned long long>::value, "");

  TEST_DIAGNOSTIC_POP

  assert(abs(-1.) == 1);
}

void test_acos()
{
    static_assert((std::is_same<decltype(acosf(0)), float>::value), "");
    static_assert((std::is_same<decltype(acosl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(acos(Ambiguous())), Ambiguous>::value), "");
    assert(acos(1) == 0);
}

void test_asin()
{
    static_assert((std::is_same<decltype(asinf(0)), float>::value), "");
    static_assert((std::is_same<decltype(asinl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(asin(Ambiguous())), Ambiguous>::value), "");
    assert(asin(0) == 0);
}

void test_atan()
{
    static_assert((std::is_same<decltype(atanf(0)), float>::value), "");
    static_assert((std::is_same<decltype(atanl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(atan(Ambiguous())), Ambiguous>::value), "");
    assert(atan(0) == 0);
}

void test_atan2()
{
    static_assert((std::is_same<decltype(atan2f(0,0)), float>::value), "");
    static_assert((std::is_same<decltype(atan2l(0,0)), long double>::value), "");
    static_assert((std::is_same<decltype(atan2(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(atan2(0,1) == 0);
}

void test_ceil()
{
    static_assert((std::is_same<decltype(ceilf(0)), float>::value), "");
    static_assert((std::is_same<decltype(ceill(0)), long double>::value), "");
    static_assert((std::is_same<decltype(ceil(Ambiguous())), Ambiguous>::value), "");
    assert(ceil(0) == 0);
}

void test_cos()
{
    static_assert((std::is_same<decltype(cosf(0)), float>::value), "");
    static_assert((std::is_same<decltype(cosl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(cos(Ambiguous())), Ambiguous>::value), "");
    assert(cos(0) == 1);
}

void test_cosh()
{
    static_assert((std::is_same<decltype(coshf(0)), float>::value), "");
    static_assert((std::is_same<decltype(coshl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(cosh(Ambiguous())), Ambiguous>::value), "");
    assert(cosh(0) == 1);
}

void test_exp()
{
    static_assert((std::is_same<decltype(expf(0)), float>::value), "");
    static_assert((std::is_same<decltype(expl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(exp(Ambiguous())), Ambiguous>::value), "");
    assert(exp(0) == 1);
}

void test_fabs()
{
    static_assert((std::is_same<decltype(fabsf(0.0f)), float>::value), "");
    static_assert((std::is_same<decltype(fabsl(0.0L)), long double>::value), "");
    static_assert((std::is_same<decltype(fabs(Ambiguous())), Ambiguous>::value), "");
    assert(fabs(-1) == 1);
}

void test_floor()
{
    static_assert((std::is_same<decltype(floorf(0)), float>::value), "");
    static_assert((std::is_same<decltype(floorl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(floor(Ambiguous())), Ambiguous>::value), "");
    assert(floor(1) == 1);
}

void test_fmod()
{
    static_assert((std::is_same<decltype(fmodf(0,0)), float>::value), "");
    static_assert((std::is_same<decltype(fmodl(0,0)), long double>::value), "");
    static_assert((std::is_same<decltype(fmod(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(fmod(1.5,1) == .5);
}

void test_frexp()
{
    int ip;
    static_assert((std::is_same<decltype(frexpf(0, &ip)), float>::value), "");
    static_assert((std::is_same<decltype(frexpl(0, &ip)), long double>::value), "");
    static_assert((std::is_same<decltype(frexp(Ambiguous(), &ip)), Ambiguous>::value), "");
    assert(frexp(0, &ip) == 0);
}

void test_ldexp()
{
    int ip = 1;
    static_assert((std::is_same<decltype(ldexpf(0, ip)), float>::value), "");
    static_assert((std::is_same<decltype(ldexpl(0, ip)), long double>::value), "");
    static_assert((std::is_same<decltype(ldexp(Ambiguous(), ip)), Ambiguous>::value), "");
    assert(ldexp(1, ip) == 2);
}

void test_log()
{
    static_assert((std::is_same<decltype(logf(0)), float>::value), "");
    static_assert((std::is_same<decltype(logl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(log(Ambiguous())), Ambiguous>::value), "");
    assert(log(1) == 0);
}

void test_log10()
{
    static_assert((std::is_same<decltype(log10f(0)), float>::value), "");
    static_assert((std::is_same<decltype(log10l(0)), long double>::value), "");
    static_assert((std::is_same<decltype(log10(Ambiguous())), Ambiguous>::value), "");
    assert(log10(1) == 0);
}

void test_modf()
{
    static_assert((std::is_same<decltype(modf((float)0, (float*)0)), float>::value), "");
    static_assert((std::is_same<decltype(modf((double)0, (double*)0)), double>::value), "");
    static_assert((std::is_same<decltype(modf((long double)0, (long double*)0)), long double>::value), "");
    static_assert((std::is_same<decltype(modff(0, (float*)0)), float>::value), "");
    static_assert((std::is_same<decltype(modfl(0, (long double*)0)), long double>::value), "");
    static_assert((std::is_same<decltype(modf(Ambiguous(), (Ambiguous*)0)), Ambiguous>::value), "");
    double i;
    assert(modf(1., &i) == 0);
}

void test_pow()
{
    static_assert((std::is_same<decltype(powf(0,0)), float>::value), "");
    static_assert((std::is_same<decltype(powl(0,0)), long double>::value), "");
    static_assert((std::is_same<decltype(pow((int)0, (int)0)), double>::value), "");
//     static_assert((std::is_same<decltype(pow(Value<int>(), (int)0)), double>::value), "");
//     static_assert((std::is_same<decltype(pow(Value<long double>(), (float)0)), long double>::value), "");
//     static_assert((std::is_same<decltype(pow((float) 0, Value<float>())), float>::value), "");
    static_assert((std::is_same<decltype(pow(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(pow(1,1) == 1);
//     assert(pow(Value<int,1>(), Value<float,1>())  == 1);
//     assert(pow(1.0f, Value<double,1>()) == 1);
//     assert(pow(1.0, Value<int,1>()) == 1);
//     assert(pow(Value<long double,1>(), 1LL) == 1);
}

void test_sin()
{
    static_assert((std::is_same<decltype(sinf(0)), float>::value), "");
    static_assert((std::is_same<decltype(sinl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(sin(Ambiguous())), Ambiguous>::value), "");
    assert(sin(0) == 0);
}

void test_sinh()
{
    static_assert((std::is_same<decltype(sinhf(0)), float>::value), "");
    static_assert((std::is_same<decltype(sinhl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(sinh(Ambiguous())), Ambiguous>::value), "");
    assert(sinh(0) == 0);
}

void test_sqrt()
{
    static_assert((std::is_same<decltype(sqrtf(0)), float>::value), "");
    static_assert((std::is_same<decltype(sqrtl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(sqrt(Ambiguous())), Ambiguous>::value), "");
    assert(sqrt(4) == 2);
}

void test_tan()
{
    static_assert((std::is_same<decltype(tanf(0)), float>::value), "");
    static_assert((std::is_same<decltype(tanl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(tan(Ambiguous())), Ambiguous>::value), "");
    assert(tan(0) == 0);
}

void test_tanh()
{
    static_assert((std::is_same<decltype(tanhf(0)), float>::value), "");
    static_assert((std::is_same<decltype(tanhl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(tanh(Ambiguous())), Ambiguous>::value), "");
    assert(tanh(0) == 0);
}

void test_signbit()
{
#ifdef signbit
#error signbit defined
#endif
    static_assert((std::is_same<decltype(signbit(Ambiguous())), Ambiguous>::value), "");
    assert(signbit(-1.0) == true);
}

void test_fpclassify()
{
#ifdef fpclassify
#error fpclassify defined
#endif
    static_assert((std::is_same<decltype(fpclassify(Ambiguous())), Ambiguous>::value), "");
    assert(fpclassify(-1.0) == FP_NORMAL);
}

void test_isfinite()
{
#ifdef isfinite
#error isfinite defined
#endif
    static_assert((std::is_same<decltype(isfinite(Ambiguous())), Ambiguous>::value), "");
    assert(isfinite(-1.0) == true);
}

void test_isnormal()
{
#ifdef isnormal
#error isnormal defined
#endif
    static_assert((std::is_same<decltype(isnormal(Ambiguous())), Ambiguous>::value), "");
    assert(isnormal(-1.0) == true);
}

void test_isgreater()
{
#ifdef isgreater
#error isgreater defined
#endif
    static_assert((std::is_same<decltype(isgreater(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(isgreater(-1.0, 0.F) == false);
}

void test_isgreaterequal()
{
#ifdef isgreaterequal
#error isgreaterequal defined
#endif
    static_assert((std::is_same<decltype(isgreaterequal(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(isgreaterequal(-1.0, 0.F) == false);
}

void test_isinf()
{
#ifdef isinf
#error isinf defined
#endif
    static_assert((std::is_same<decltype(isinf((float)0)), bool>::value), "");

    typedef decltype(isinf((double)0)) DoubleRetType;
#ifndef __linux__
    static_assert((std::is_same<DoubleRetType, bool>::value), "");
#else
    // GLIBC < 2.26 defines 'isinf(double)' with a return type of 'int' in
    // all C++ dialects. The test should tolerate this.
    // See: https://sourceware.org/bugzilla/show_bug.cgi?id=19439
    static_assert((std::is_same<DoubleRetType, bool>::value
                || std::is_same<DoubleRetType, int>::value), "");
#endif

    static_assert((std::is_same<decltype(isinf(0)), bool>::value), "");
    static_assert((std::is_same<decltype(isinf((long double)0)), bool>::value), "");
    assert(isinf(-1.0) == false);
}

void test_isless()
{
#ifdef isless
#error isless defined
#endif
    static_assert((std::is_same<decltype(isless(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(isless(-1.0, 0.F) == true);
}

void test_islessequal()
{
#ifdef islessequal
#error islessequal defined
#endif
    static_assert((std::is_same<decltype(islessequal(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(islessequal(-1.0, 0.F) == true);
}

void test_islessgreater()
{
#ifdef islessgreater
#error islessgreater defined
#endif
    static_assert((std::is_same<decltype(islessgreater(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(islessgreater(-1.0, 0.F) == true);
}

void test_isnan()
{
#ifdef isnan
#error isnan defined
#endif
    static_assert((std::is_same<decltype(isnan((float)0)), bool>::value), "");

    typedef decltype(isnan((double)0)) DoubleRetType;
#ifndef __linux__
    static_assert((std::is_same<DoubleRetType, bool>::value), "");
#else
    // GLIBC < 2.26 defines 'isnan(double)' with a return type of 'int' in
    // all C++ dialects. The test should tolerate this.
    // See: https://sourceware.org/bugzilla/show_bug.cgi?id=19439
    static_assert((std::is_same<DoubleRetType, bool>::value
                || std::is_same<DoubleRetType, int>::value), "");
#endif

    static_assert((std::is_same<decltype(isnan(0)), bool>::value), "");
    static_assert((std::is_same<decltype(isnan((long double)0)), bool>::value), "");
    assert(isnan(-1.0) == false);
}

void test_isunordered()
{
#ifdef isunordered
#error isunordered defined
#endif
    static_assert((std::is_same<decltype(isunordered(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(isunordered(-1.0, 0.F) == false);
}

void test_acosh()
{
    static_assert((std::is_same<decltype(acoshf(0)), float>::value), "");
    static_assert((std::is_same<decltype(acoshl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(acosh(Ambiguous())), Ambiguous>::value), "");
    assert(acosh(1) == 0);
}

void test_asinh()
{
    static_assert((std::is_same<decltype(asinhf(0)), float>::value), "");
    static_assert((std::is_same<decltype(asinhl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(asinh(Ambiguous())), Ambiguous>::value), "");
    assert(asinh(0) == 0);
}

void test_atanh()
{
    static_assert((std::is_same<decltype(atanhf(0)), float>::value), "");
    static_assert((std::is_same<decltype(atanhl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(atanh(Ambiguous())), Ambiguous>::value), "");
    assert(atanh(0) == 0);
}

void test_cbrt() {
    static_assert((std::is_same<decltype(cbrtf(0)), float>::value), "");
    static_assert((std::is_same<decltype(cbrtl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(cbrt(Ambiguous())), Ambiguous>::value),
                  "");
    assert(truncate_fp(cbrt(1)) == 1);

}

void test_copysign()
{
    static_assert((std::is_same<decltype(copysignf(0,0)), float>::value), "");
    static_assert((std::is_same<decltype(copysignl(0,0)), long double>::value), "");
    static_assert((std::is_same<decltype(copysign((int)0, (int)0)), double>::value), "");
    static_assert((std::is_same<decltype(copysign(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(copysign(1,1) == 1);
}

void test_erf()
{
    static_assert((std::is_same<decltype(erff(0)), float>::value), "");
    static_assert((std::is_same<decltype(erfl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(erf(Ambiguous())), Ambiguous>::value), "");
    assert(erf(0) == 0);
}

void test_erfc()
{
    static_assert((std::is_same<decltype(erfcf(0)), float>::value), "");
    static_assert((std::is_same<decltype(erfcl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(erfc(Ambiguous())), Ambiguous>::value), "");
    assert(erfc(0) == 1);
}

void test_exp2()
{
    static_assert((std::is_same<decltype(exp2f(0)), float>::value), "");
    static_assert((std::is_same<decltype(exp2l(0)), long double>::value), "");
    static_assert((std::is_same<decltype(exp2(Ambiguous())), Ambiguous>::value), "");
    assert(exp2(1) == 2);
}

void test_expm1()
{
    static_assert((std::is_same<decltype(expm1f(0)), float>::value), "");
    static_assert((std::is_same<decltype(expm1l(0)), long double>::value), "");
    static_assert((std::is_same<decltype(expm1(Ambiguous())), Ambiguous>::value), "");
    assert(expm1(0) == 0);
}

void test_fdim()
{
    static_assert((std::is_same<decltype(fdimf(0,0)), float>::value), "");
    static_assert((std::is_same<decltype(fdiml(0,0)), long double>::value), "");
    static_assert((std::is_same<decltype(fdim((int)0, (int)0)), double>::value), "");
    static_assert((std::is_same<decltype(fdim(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(fdim(1,0) == 1);
}

void test_fma()
{
    static_assert((std::is_same<decltype(fma((bool)0, (float)0, (float)0)), double>::value), "");
    static_assert((std::is_same<decltype(fma((float)0, (float)0, (double)0)), double>::value), "");
    static_assert((std::is_same<decltype(fma((float)0, (float)0, (long double)0)), long double>::value), "");
    static_assert((std::is_same<decltype(fma((float)0, (float)0, (float)0)), float>::value), "");

    static_assert((std::is_same<decltype(fma((bool)0, (double)0, (double)0)), double>::value), "");
    static_assert((std::is_same<decltype(fma((double)0, (double)0, (float)0)), double>::value), "");
    static_assert((std::is_same<decltype(fma((double)0, (double)0, (long double)0)), long double>::value), "");
    static_assert((std::is_same<decltype(fma((double)0, (double)0,  (double)0)), double>::value), "");

    static_assert((std::is_same<decltype(fma((long double)0, (long double)0, (float)0)), long double>::value), "");
    static_assert((std::is_same<decltype(fma((double)0, (long double)0, (long double)0)), long double>::value), "");
    static_assert((std::is_same<decltype(fma((long double)0, (long double)0, (long double)0)), long double>::value), "");

    static_assert((std::is_same<decltype(fmaf(0,0,0)), float>::value), "");
    static_assert((std::is_same<decltype(fmal(0,0,0)), long double>::value), "");
    static_assert((std::is_same<decltype(fma(Ambiguous(), Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(fma(1,1,1) == 2);
}

void test_fmax()
{
    static_assert((std::is_same<decltype(fmaxf(0,0)), float>::value), "");
    static_assert((std::is_same<decltype(fmaxl(0,0)), long double>::value), "");
    static_assert((std::is_same<decltype(fmax((int)0, (int)0)), double>::value), "");
    static_assert((std::is_same<decltype(fmax(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(fmax(1,0) == 1);
}

void test_fmin()
{
    static_assert((std::is_same<decltype(fminf(0,0)), float>::value), "");
    static_assert((std::is_same<decltype(fminl(0,0)), long double>::value), "");
    static_assert((std::is_same<decltype(fmin((int)0, (int)0)), double>::value), "");
    static_assert((std::is_same<decltype(fmin(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(fmin(1,0) == 0);
}

void test_hypot()
{
    static_assert((std::is_same<decltype(hypotf(0,0)), float>::value), "");
    static_assert((std::is_same<decltype(hypotl(0,0)), long double>::value), "");
    static_assert((std::is_same<decltype(hypot((int)0, (int)0)), double>::value), "");
    static_assert((std::is_same<decltype(hypot(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(hypot(3,4) == 5);
}

void test_ilogb()
{
    static_assert((std::is_same<decltype(ilogbf(0)), int>::value), "");
    static_assert((std::is_same<decltype(ilogbl(0)), int>::value), "");
    static_assert((std::is_same<decltype(ilogb(Ambiguous())), Ambiguous>::value), "");
    assert(ilogb(1) == 0);
}

void test_lgamma()
{
    static_assert((std::is_same<decltype(lgammaf(0)), float>::value), "");
    static_assert((std::is_same<decltype(lgammal(0)), long double>::value), "");
    static_assert((std::is_same<decltype(lgamma(Ambiguous())), Ambiguous>::value), "");
    assert(lgamma(1) == 0);
}

void test_llrint()
{
    static_assert((std::is_same<decltype(llrintf(0)), long long>::value), "");
    static_assert((std::is_same<decltype(llrintl(0)), long long>::value), "");
    static_assert((std::is_same<decltype(llrint(Ambiguous())), Ambiguous>::value), "");
    assert(llrint(1) == 1LL);
}

void test_llround()
{
    static_assert((std::is_same<decltype(llroundf(0)), long long>::value), "");
    static_assert((std::is_same<decltype(llroundl(0)), long long>::value), "");
    static_assert((std::is_same<decltype(llround(Ambiguous())), Ambiguous>::value), "");
    assert(llround(1) == 1LL);
}

void test_log1p()
{
    static_assert((std::is_same<decltype(log1pf(0)), float>::value), "");
    static_assert((std::is_same<decltype(log1pl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(log1p(Ambiguous())), Ambiguous>::value), "");
    assert(log1p(0) == 0);
}

void test_log2()
{
    static_assert((std::is_same<decltype(log2f(0)), float>::value), "");
    static_assert((std::is_same<decltype(log2l(0)), long double>::value), "");
    static_assert((std::is_same<decltype(log2(Ambiguous())), Ambiguous>::value), "");
    assert(log2(1) == 0);
}

void test_logb()
{
    static_assert((std::is_same<decltype(logbf(0)), float>::value), "");
    static_assert((std::is_same<decltype(logbl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(logb(Ambiguous())), Ambiguous>::value), "");
    assert(logb(1) == 0);
}

void test_lrint()
{
    static_assert((std::is_same<decltype(lrintf(0)), long>::value), "");
    static_assert((std::is_same<decltype(lrintl(0)), long>::value), "");
    static_assert((std::is_same<decltype(lrint(Ambiguous())), Ambiguous>::value), "");
    assert(lrint(1) == 1L);
}

void test_lround()
{
    static_assert((std::is_same<decltype(lroundf(0)), long>::value), "");
    static_assert((std::is_same<decltype(lroundl(0)), long>::value), "");
    static_assert((std::is_same<decltype(lround(Ambiguous())), Ambiguous>::value), "");
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
    static_assert((std::is_same<decltype(nearbyintf(0)), float>::value), "");
    static_assert((std::is_same<decltype(nearbyintl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(nearbyint(Ambiguous())), Ambiguous>::value), "");
    assert(nearbyint(1) == 1);
}

void test_nextafter()
{
    static_assert((std::is_same<decltype(nextafterf(0,0)), float>::value), "");
    static_assert((std::is_same<decltype(nextafterl(0,0)), long double>::value), "");
    static_assert((std::is_same<decltype(nextafter((int)0, (int)0)), double>::value), "");
    static_assert((std::is_same<decltype(nextafter(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(nextafter(0,1) == hexfloat<double>(0x1, 0, -1074));
}

void test_nexttoward()
{
    static_assert((std::is_same<decltype(nexttoward((float)0, (long double)0)), float>::value), "");
    static_assert((std::is_same<decltype(nexttoward((bool)0, (long double)0)), double>::value), "");
    static_assert((std::is_same<decltype(nexttoward((unsigned short)0, (long double)0)), double>::value), "");
    static_assert((std::is_same<decltype(nexttoward((int)0, (long double)0)), double>::value), "");
    static_assert((std::is_same<decltype(nexttoward((unsigned int)0, (long double)0)), double>::value), "");
    static_assert((std::is_same<decltype(nexttoward((long)0, (long double)0)), double>::value), "");
    static_assert((std::is_same<decltype(nexttoward((unsigned long)0, (long double)0)), double>::value), "");
    static_assert((std::is_same<decltype(nexttoward((long long)0, (long double)0)), double>::value), "");
    static_assert((std::is_same<decltype(nexttoward((unsigned long long)0, (long double)0)), double>::value), "");
    static_assert((std::is_same<decltype(nexttoward((double)0, (long double)0)), double>::value), "");
    static_assert((std::is_same<decltype(nexttoward((long double)0, (long double)0)), long double>::value), "");
    static_assert((std::is_same<decltype(nexttowardf(0, (long double)0)), float>::value), "");
    static_assert((std::is_same<decltype(nexttowardl(0, (long double)0)), long double>::value), "");
    static_assert((std::is_same<decltype(nexttoward(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(nexttoward(0, 1) == hexfloat<double>(0x1, 0, -1074));
}

void test_remainder()
{
    static_assert((std::is_same<decltype(remainderf(0,0)), float>::value), "");
    static_assert((std::is_same<decltype(remainderl(0,0)), long double>::value), "");
    static_assert((std::is_same<decltype(remainder((int)0, (int)0)), double>::value), "");
    static_assert((std::is_same<decltype(remainder(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(remainder(0.5,1) == 0.5);
}

void test_remquo()
{
    int ip;
    static_assert((std::is_same<decltype(remquof(0,0, &ip)), float>::value), "");
    static_assert((std::is_same<decltype(remquol(0,0, &ip)), long double>::value), "");
    static_assert((std::is_same<decltype(remquo((int)0, (int)0, &ip)), double>::value), "");
    static_assert((std::is_same<decltype(remquo(Ambiguous(), Ambiguous(), &ip)), Ambiguous>::value), "");
    assert(remquo(0.5,1, &ip) == 0.5);
}

void test_rint()
{
    static_assert((std::is_same<decltype(rintf(0)), float>::value), "");
    static_assert((std::is_same<decltype(rintl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(rint(Ambiguous())), Ambiguous>::value), "");
    assert(rint(1) == 1);
}

void test_round()
{
    static_assert((std::is_same<decltype(roundf(0)), float>::value), "");
    static_assert((std::is_same<decltype(roundl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(round(Ambiguous())), Ambiguous>::value), "");
    assert(round(1) == 1);
}

void test_scalbln()
{
    static_assert((std::is_same<decltype(scalbln((float)0, (long)0)), float>::value), "");
    static_assert((std::is_same<decltype(scalbln((bool)0, (long)0)), double>::value), "");
    static_assert((std::is_same<decltype(scalbln((unsigned short)0, (long)0)), double>::value), "");
    static_assert((std::is_same<decltype(scalbln((int)0, (long)0)), double>::value), "");
    static_assert((std::is_same<decltype(scalbln((unsigned int)0, (long)0)), double>::value), "");
    static_assert((std::is_same<decltype(scalbln((long)0, (long)0)), double>::value), "");
    static_assert((std::is_same<decltype(scalbln((unsigned long)0, (long)0)), double>::value), "");
    static_assert((std::is_same<decltype(scalbln((long long)0, (long)0)), double>::value), "");
    static_assert((std::is_same<decltype(scalbln((unsigned long long)0, (long)0)), double>::value), "");
    static_assert((std::is_same<decltype(scalbln((double)0, (long)0)), double>::value), "");
    static_assert((std::is_same<decltype(scalbln((long double)0, (long)0)), long double>::value), "");
    static_assert((std::is_same<decltype(scalblnf(0, (long)0)), float>::value), "");
    static_assert((std::is_same<decltype(scalblnl(0, (long)0)), long double>::value), "");
    static_assert((std::is_same<decltype(scalbln(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(scalbln(1, 1) == 2);
}

void test_scalbn()
{
    static_assert((std::is_same<decltype(scalbn((float)0, (int)0)), float>::value), "");
    static_assert((std::is_same<decltype(scalbn((bool)0, (int)0)), double>::value), "");
    static_assert((std::is_same<decltype(scalbn((unsigned short)0, (int)0)), double>::value), "");
    static_assert((std::is_same<decltype(scalbn((int)0, (int)0)), double>::value), "");
    static_assert((std::is_same<decltype(scalbn((unsigned int)0, (int)0)), double>::value), "");
    static_assert((std::is_same<decltype(scalbn((long)0, (int)0)), double>::value), "");
    static_assert((std::is_same<decltype(scalbn((unsigned long)0, (int)0)), double>::value), "");
    static_assert((std::is_same<decltype(scalbn((long long)0, (int)0)), double>::value), "");
    static_assert((std::is_same<decltype(scalbn((unsigned long long)0, (int)0)), double>::value), "");
    static_assert((std::is_same<decltype(scalbn((double)0, (int)0)), double>::value), "");
    static_assert((std::is_same<decltype(scalbn((long double)0, (int)0)), long double>::value), "");
    static_assert((std::is_same<decltype(scalbnf(0, (int)0)), float>::value), "");
    static_assert((std::is_same<decltype(scalbnl(0, (int)0)), long double>::value), "");
    static_assert((std::is_same<decltype(scalbn(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(scalbn(1, 1) == 2);
}

void test_tgamma()
{
    static_assert((std::is_same<decltype(tgammaf(0)), float>::value), "");
    static_assert((std::is_same<decltype(tgammal(0)), long double>::value), "");
    static_assert((std::is_same<decltype(tgamma(Ambiguous())), Ambiguous>::value), "");
    assert(tgamma(1) == 1);
}

void test_trunc()
{
    static_assert((std::is_same<decltype(truncf(0)), float>::value), "");
    static_assert((std::is_same<decltype(truncl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(trunc(Ambiguous())), Ambiguous>::value), "");
    assert(trunc(1) == 1);
}

template <class PromoteResult, class Arg = void>
struct test_single_arg {
  template <class T = Arg>
  void operator()() {
    static_assert((std::is_same<decltype(::acos(T())), PromoteResult>::value), "");
    (void)::acos(T());
    static_assert((std::is_same<decltype(::asin(T())), PromoteResult>::value), "");
    (void)::asin(T());
    static_assert((std::is_same<decltype(::atan(T())), PromoteResult>::value), "");
    (void)::atan(T());
    static_assert((std::is_same<decltype(::ceil(T())), PromoteResult>::value), "");
    (void)::ceil(T());
    static_assert((std::is_same<decltype(::cos(T())), PromoteResult>::value), "");
    (void)::cos(T());
    static_assert((std::is_same<decltype(::cosh(T())), PromoteResult>::value), "");
    (void)::cosh(T());
    static_assert((std::is_same<decltype(::exp(T())), PromoteResult>::value), "");
    (void)::exp(T());
    static_assert((std::is_same<decltype(::fabs(T())), PromoteResult>::value), "");
    (void)::fabs(T());
    static_assert((std::is_same<decltype(::floor(T())), PromoteResult>::value), "");
    (void)::floor(T());
    int ip;
    static_assert((std::is_same<decltype(::frexp(T(), &ip)), PromoteResult>::value), "");
    (void)::frexp(T(), &ip);
    static_assert((std::is_same<decltype(::ldexp(T(), ip)), PromoteResult>::value), "");
    (void)::ldexp(T(), ip);
    static_assert((std::is_same<decltype(::log(T())), PromoteResult>::value), "");
    (void)::log(T());
    static_assert((std::is_same<decltype(::log10(T())), PromoteResult>::value), "");
    (void)::log10(T());
    static_assert((std::is_same<decltype(::sin(T())), PromoteResult>::value), "");
    (void)::sin(T());
    static_assert((std::is_same<decltype(::sinh(T())), PromoteResult>::value), "");
    (void)::sinh(T());
    static_assert((std::is_same<decltype(::sqrt(T())), PromoteResult>::value), "");
    (void)::sqrt(T());
    static_assert((std::is_same<decltype(::tan(T())), PromoteResult>::value), "");
    (void)::tan(T());
    static_assert((std::is_same<decltype(::tanh(T())), PromoteResult>::value), "");
    (void)::tanh(T());
    static_assert((std::is_same<decltype(::signbit(T())), bool>::value), "");
    (void)::signbit(T());
    static_assert((std::is_same<decltype(::fpclassify(T())), int>::value), "");
    (void)::fpclassify(T());
    static_assert((std::is_same<decltype(::isfinite(T())), bool>::value), "");
    (void)::isfinite(T());
    static_assert((std::is_same<decltype(::isnormal(T())), bool>::value), "");
    (void)::isnormal(T());
    static_assert((std::is_same<decltype(::acosh(T())), PromoteResult>::value), "");
    (void)::acosh(T());
    static_assert((std::is_same<decltype(::asinh(T())), PromoteResult>::value), "");
    (void)::asinh(T());
    static_assert((std::is_same<decltype(::atanh(T())), PromoteResult>::value), "");
    (void)::atanh(T());
    static_assert((std::is_same<decltype(::cbrt(T())), PromoteResult>::value), "");
    (void)::cbrt(T());
    static_assert((std::is_same<decltype(::erf(T())), PromoteResult>::value), "");
    (void)::erf(T());
    static_assert((std::is_same<decltype(::erfc(T())), PromoteResult>::value), "");
    (void)::erfc(T());
    static_assert((std::is_same<decltype(::exp2(T())), PromoteResult>::value), "");
    (void)::exp2(T());
    static_assert((std::is_same<decltype(::expm1(T())), PromoteResult>::value), "");
    (void)::expm1(T());
    static_assert((std::is_same<decltype(::ilogb(T())), int>::value), "");
    (void)::ilogb(T());
    static_assert((std::is_same<decltype(::lgamma(T())), PromoteResult>::value), "");
    (void)::lgamma(T());
    static_assert((std::is_same<decltype(::llrint(T())), long long>::value), "");
    (void)::llrint(T());
    static_assert((std::is_same<decltype(::llround(T())), long long>::value), "");
    (void)::llround(T());
    static_assert((std::is_same<decltype(::log1p(T())), PromoteResult>::value), "");
    (void)::log1p(T());
    static_assert((std::is_same<decltype(::log2(T())), PromoteResult>::value), "");
    (void)::log2(T());
    static_assert((std::is_same<decltype(::logb(T())), PromoteResult>::value), "");
    (void)::logb(T());
    static_assert((std::is_same<decltype(::lrint(T())), long>::value), "");
    (void)::lrint(T());
    static_assert((std::is_same<decltype(::lround(T())), long>::value), "");
    (void)::lround(T());
    static_assert((std::is_same<decltype(::nearbyint(T())), PromoteResult>::value), "");
    (void)::nearbyint(T());
    static_assert((std::is_same<decltype(::rint(T())), PromoteResult>::value), "");
    (void)::rint(T());
    static_assert((std::is_same<decltype(::round(T())), PromoteResult>::value), "");
    (void)::round(T());
    static_assert((std::is_same<decltype(::trunc(T())), PromoteResult>::value), "");
    (void)::trunc(T());
    static_assert((std::is_same<decltype(::tgamma(T())), PromoteResult>::value), "");
    (void)::tgamma(T());
  }
};

template <class PromoteResult, class Arg1 = void, class Arg2 = void>
struct test_two_args {
  template <class T = Arg1, class U = Arg2>
  void operator()() {
    static_assert((std::is_same<decltype(::atan2(T(), U())), PromoteResult>::value), "");
    (void)::atan2(T(), U());
    static_assert((std::is_same<decltype(::fmod(T(), U())), PromoteResult>::value), "");
    (void)::fmod(T(), U());
    static_assert((std::is_same<decltype(::pow(T(), U())), PromoteResult>::value), "");
    (void)::pow(T(), U());
    static_assert((std::is_same<decltype(::isgreater(T(), U())), bool>::value), "");
    (void)::isgreater(T(), U());
    static_assert((std::is_same<decltype(::isgreaterequal(T(), U())), bool>::value), "");
    (void)::isgreaterequal(T(), U());
    static_assert((std::is_same<decltype(::isless(T(), U())), bool>::value), "");
    (void)::isless(T(), U());
    static_assert((std::is_same<decltype(::islessequal(T(), U())), bool>::value), "");
    (void)::islessequal(T(), U());
    static_assert((std::is_same<decltype(::islessgreater(T(), U())), bool>::value), "");
    (void)::islessgreater(T(), U());
    static_assert((std::is_same<decltype(::isunordered(T(), U())), bool>::value), "");
    (void)::isunordered(T(), U());
    static_assert((std::is_same<decltype(::copysign(T(), U())), PromoteResult>::value), "");
    (void)::copysign(T(), U());
    static_assert((std::is_same<decltype(::fdim(T(), U())), PromoteResult>::value), "");
    (void)::fdim(T(), U());
    static_assert((std::is_same<decltype(::fmax(T(), U())), PromoteResult>::value), "");
    (void)::fmax(T(), U());
    static_assert((std::is_same<decltype(::fmin(T(), U())), PromoteResult>::value), "");
    (void)::fmin(T(), U());
    static_assert((std::is_same<decltype(::hypot(T(), U())), PromoteResult>::value), "");
    (void)::hypot(T(), U());
    static_assert((std::is_same<decltype(::nextafter(T(), U())), PromoteResult>::value), "");
    (void)::nextafter(T(), U());
    static_assert((std::is_same<decltype(::remainder(T(), U())), PromoteResult>::value), "");
    (void)::remainder(T(), U());
    int ip;
    static_assert((std::is_same<decltype(::remquo(T(), U(), &ip)), PromoteResult>::value), "");
    ::remquo(T(), U(), &ip);
  }
};

template <class PromoteResult, class Arg1 = void, class Arg2 = void, class Arg3 = void>
struct test_three_args {
  template <class T = Arg1, class U = Arg2, class V = Arg3>
  void operator()() {
    static_assert((std::is_same<decltype(::fma(T(), U(), V())), PromoteResult>::value), "");
    (void)::fma(T(), U(), V());
  }
};

struct CallTwoArgs {
  using integral_float_double = meta::concatenate_t<meta::integral_types, meta::type_list<float, double> >;

  template <class Arg2>
  void operator()() {
    meta::for_each(integral_float_double(), test_two_args</*PromoteResult=*/double, /*Iterate*/void, Arg2>());
  }
};

template <class T>
struct CallThreeArgs {
  using integral_float_double = meta::concatenate_t<meta::integral_types, meta::type_list<float, double> >;

  template <class Arg3>
  struct Helper {

    template <class Arg2>
    void operator()() {
      meta::for_each(integral_float_double(), test_three_args</*PromoteResult=*/double, /*Iterate*/void, Arg2, Arg3>());
    }
  };

  template <class Arg3>
  void operator()() {
    meta::for_each(integral_float_double(), Helper<Arg3>());
  }
};

int main(int, char**)
{
  meta::for_each(meta::integral_types(), test_single_arg</*PromoteResult=*/double>());
  test_single_arg</*PromoteResult=*/float, /*Arg=*/float>();
  test_single_arg</*PromoteResult=*/double, /*Arg=*/double>();
  test_single_arg</*PromoteResult=*/long double, /*Arg=*/long double>();

  meta::for_each(meta::integral_types(), CallTwoArgs());

  meta::for_each(
      meta::integral_types(), test_two_args</*PromoteResult=*/long double, /*Arg1=*/void, /*Arg2=*/long double>());

  test_two_args</*PromoteResult=*/float, /*Args=*/float, float>();
  test_two_args</*PromoteResult=*/float, /*Args=*/double, double>();
  test_two_args</*PromoteResult=*/double, /*Args=*/float, double>();
  test_two_args</*PromoteResult=*/double, /*Args=*/double, double>();

  meta::for_each(meta::integral_types(), CallThreeArgs<double>());
  meta::for_each(
      meta::integral_types(), test_three_args</*PromoteResult=*/long double, /*Iterate*/ void, long double, double>());

  test_three_args</*PromoteResult=*/float, /*Args=*/float, float, float>();
  test_three_args</*PromoteResult=*/double, /*Args=*/double, double, double>();
  test_three_args</*PromoteResult=*/long double, /*Args=*/long double, long double, long double>();

  test_abs();
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
  test_isnormal();
  test_isgreater();
  test_isgreaterequal();
  test_isinf();
  test_isless();
  test_islessequal();
  test_islessgreater();
  test_isnan();
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

  return 0;
}
