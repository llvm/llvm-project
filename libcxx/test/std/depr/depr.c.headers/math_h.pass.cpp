//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test fails because Clang no longer enables -fdelayed-template-parsing
// by default on Windows with C++20 (#69431).
// XFAIL: msvc && (clang-18 || clang-19)

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

void test_abs() {
  TEST_DIAGNOSTIC_PUSH
  TEST_CLANG_DIAGNOSTIC_IGNORED("-Wabsolute-value")

  ASSERT_SAME_TYPE(decltype(abs((float)0)), float);
  ASSERT_SAME_TYPE(decltype(abs((double)0)), double);
  ASSERT_SAME_TYPE(decltype(abs((long double)0)), long double);
  ASSERT_SAME_TYPE(decltype(abs((int)0)), int);
  ASSERT_SAME_TYPE(decltype(abs((long)0)), long);
  ASSERT_SAME_TYPE(decltype(abs((long long)0)), long long);
  ASSERT_SAME_TYPE(decltype(abs((unsigned char)0)), int);
  ASSERT_SAME_TYPE(decltype(abs((unsigned short)0)), int);
  ASSERT_SAME_TYPE(decltype(abs(Ambiguous())), Ambiguous);

  static_assert(!has_abs<unsigned>::value, "");
  static_assert(!has_abs<unsigned long>::value, "");
  static_assert(!has_abs<unsigned long long>::value, "");

  TEST_DIAGNOSTIC_POP

  assert(abs(-1.) == 1);
}

void test_acos() {
    ASSERT_SAME_TYPE(decltype(acosf(0)), float);
    ASSERT_SAME_TYPE(decltype(acosl(0)), long double);
    ASSERT_SAME_TYPE(decltype(acos(Ambiguous())), Ambiguous);
    assert(acos(1) == 0);
}

void test_asin() {
    ASSERT_SAME_TYPE(decltype(asinf(0)), float);
    ASSERT_SAME_TYPE(decltype(asinl(0)), long double);
    ASSERT_SAME_TYPE(decltype(asin(Ambiguous())), Ambiguous);
    assert(asin(0) == 0);
}

void test_atan() {
    ASSERT_SAME_TYPE(decltype(atanf(0)), float);
    ASSERT_SAME_TYPE(decltype(atanl(0)), long double);
    ASSERT_SAME_TYPE(decltype(atan(Ambiguous())), Ambiguous);
    assert(atan(0) == 0);
}

void test_atan2() {
    ASSERT_SAME_TYPE(decltype(atan2f(0,0)), float);
    ASSERT_SAME_TYPE(decltype(atan2l(0,0)), long double);
    ASSERT_SAME_TYPE(decltype(atan2(Ambiguous(), Ambiguous())), Ambiguous);
    assert(atan2(0,1) == 0);
}

void test_ceil() {
    ASSERT_SAME_TYPE(decltype(ceilf(0)), float);
    ASSERT_SAME_TYPE(decltype(ceill(0)), long double);
    ASSERT_SAME_TYPE(decltype(ceil(Ambiguous())), Ambiguous);
    assert(ceil(0) == 0);
}

void test_cos() {
    ASSERT_SAME_TYPE(decltype(cosf(0)), float);
    ASSERT_SAME_TYPE(decltype(cosl(0)), long double);
    ASSERT_SAME_TYPE(decltype(cos(Ambiguous())), Ambiguous);
    assert(cos(0) == 1);
}

void test_cosh() {
    ASSERT_SAME_TYPE(decltype(coshf(0)), float);
    ASSERT_SAME_TYPE(decltype(coshl(0)), long double);
    ASSERT_SAME_TYPE(decltype(cosh(Ambiguous())), Ambiguous);
    assert(cosh(0) == 1);
}

void test_exp() {
    ASSERT_SAME_TYPE(decltype(expf(0)), float);
    ASSERT_SAME_TYPE(decltype(expl(0)), long double);
    ASSERT_SAME_TYPE(decltype(exp(Ambiguous())), Ambiguous);
    assert(exp(0) == 1);
}

void test_fabs() {
    ASSERT_SAME_TYPE(decltype(fabsf(0.0f)), float);
    ASSERT_SAME_TYPE(decltype(fabsl(0.0L)), long double);
    ASSERT_SAME_TYPE(decltype(fabs(Ambiguous())), Ambiguous);
    assert(fabs(-1) == 1);
}

void test_floor() {
    ASSERT_SAME_TYPE(decltype(floorf(0)), float);
    ASSERT_SAME_TYPE(decltype(floorl(0)), long double);
    ASSERT_SAME_TYPE(decltype(floor(Ambiguous())), Ambiguous);
    assert(floor(1) == 1);
}

void test_fmod() {
    ASSERT_SAME_TYPE(decltype(fmodf(0,0)), float);
    ASSERT_SAME_TYPE(decltype(fmodl(0,0)), long double);
    ASSERT_SAME_TYPE(decltype(fmod(Ambiguous(), Ambiguous())), Ambiguous);
    assert(fmod(1.5,1) == .5);
}

void test_frexp() {
    int ip;
    ASSERT_SAME_TYPE(decltype(frexpf(0, &ip)), float);
    ASSERT_SAME_TYPE(decltype(frexpl(0, &ip)), long double);
    ASSERT_SAME_TYPE(decltype(frexp(Ambiguous(), &ip)), Ambiguous);
    assert(frexp(0, &ip) == 0);
}

void test_ldexp() {
    int ip = 1;
    ASSERT_SAME_TYPE(decltype(ldexpf(0, ip)), float);
    ASSERT_SAME_TYPE(decltype(ldexpl(0, ip)), long double);
    ASSERT_SAME_TYPE(decltype(ldexp(Ambiguous(), ip)), Ambiguous);
    assert(ldexp(1, ip) == 2);
}

void test_log() {
    ASSERT_SAME_TYPE(decltype(logf(0)), float);
    ASSERT_SAME_TYPE(decltype(logl(0)), long double);
    ASSERT_SAME_TYPE(decltype(log(Ambiguous())), Ambiguous);
    assert(log(1) == 0);
}

void test_log10() {
    ASSERT_SAME_TYPE(decltype(log10f(0)), float);
    ASSERT_SAME_TYPE(decltype(log10l(0)), long double);
    ASSERT_SAME_TYPE(decltype(log10(Ambiguous())), Ambiguous);
    assert(log10(1) == 0);
}

void test_modf() {
    ASSERT_SAME_TYPE(decltype(modf((float)0, (float*)0)), float);
    ASSERT_SAME_TYPE(decltype(modf((double)0, (double*)0)), double);
    ASSERT_SAME_TYPE(decltype(modf((long double)0, (long double*)0)), long double);
    ASSERT_SAME_TYPE(decltype(modff(0, (float*)0)), float);
    ASSERT_SAME_TYPE(decltype(modfl(0, (long double*)0)), long double);
    ASSERT_SAME_TYPE(decltype(modf(Ambiguous(), (Ambiguous*)0)), Ambiguous);
    double i;
    assert(modf(1., &i) == 0);
}

void test_pow() {
    ASSERT_SAME_TYPE(decltype(powf(0,0)), float);
    ASSERT_SAME_TYPE(decltype(powl(0,0)), long double);
    ASSERT_SAME_TYPE(decltype(pow((int)0, (int)0)), double);
    // ASSERT_SAME_TYPE(decltype(pow(Value<int>(), (int)0)), double);
    // ASSERT_SAME_TYPE(decltype(pow(Value<long double>(), (float)0)), long double);
    // ASSERT_SAME_TYPE(decltype(pow((float) 0, Value<float>())), float);
    ASSERT_SAME_TYPE(decltype(pow(Ambiguous(), Ambiguous())), Ambiguous);
    assert(pow(1,1) == 1);
    // assert(pow(Value<int,1>(), Value<float,1>())  == 1);
    // assert(pow(1.0f, Value<double,1>()) == 1);
    // assert(pow(1.0, Value<int,1>()) == 1);
    // assert(pow(Value<long double,1>(), 1LL) == 1);
}

void test_sin() {
    ASSERT_SAME_TYPE(decltype(sinf(0)), float);
    ASSERT_SAME_TYPE(decltype(sinl(0)), long double);
    ASSERT_SAME_TYPE(decltype(sin(Ambiguous())), Ambiguous);
    assert(sin(0) == 0);
}

void test_sinh() {
    ASSERT_SAME_TYPE(decltype(sinhf(0)), float);
    ASSERT_SAME_TYPE(decltype(sinhl(0)), long double);
    ASSERT_SAME_TYPE(decltype(sinh(Ambiguous())), Ambiguous);
    assert(sinh(0) == 0);
}

void test_sqrt() {
    ASSERT_SAME_TYPE(decltype(sqrtf(0)), float);
    ASSERT_SAME_TYPE(decltype(sqrtl(0)), long double);
    ASSERT_SAME_TYPE(decltype(sqrt(Ambiguous())), Ambiguous);
    assert(sqrt(4) == 2);
}

void test_tan() {
    ASSERT_SAME_TYPE(decltype(tanf(0)), float);
    ASSERT_SAME_TYPE(decltype(tanl(0)), long double);
    ASSERT_SAME_TYPE(decltype(tan(Ambiguous())), Ambiguous);
    assert(tan(0) == 0);
}

void test_tanh() {
    ASSERT_SAME_TYPE(decltype(tanhf(0)), float);
    ASSERT_SAME_TYPE(decltype(tanhl(0)), long double);
    ASSERT_SAME_TYPE(decltype(tanh(Ambiguous())), Ambiguous);
    assert(tanh(0) == 0);
}

void test_signbit() {
#ifdef signbit
#error signbit defined
#endif
    ASSERT_SAME_TYPE(decltype(signbit(Ambiguous())), Ambiguous);
    assert(signbit(-1.0) == true);
}

void test_fpclassify() {
#ifdef fpclassify
#error fpclassify defined
#endif
    ASSERT_SAME_TYPE(decltype(fpclassify(Ambiguous())), Ambiguous);
    assert(fpclassify(-1.0) == FP_NORMAL);
}

void test_isfinite() {
#ifdef isfinite
#error isfinite defined
#endif
    ASSERT_SAME_TYPE(decltype(isfinite(Ambiguous())), Ambiguous);
    assert(isfinite(-1.0) == true);
}

void test_isnormal() {
#ifdef isnormal
#error isnormal defined
#endif
    ASSERT_SAME_TYPE(decltype(isnormal(Ambiguous())), Ambiguous);
    assert(isnormal(-1.0) == true);
}

void test_isgreater() {
#ifdef isgreater
#error isgreater defined
#endif
    ASSERT_SAME_TYPE(decltype(isgreater(Ambiguous(), Ambiguous())), Ambiguous);
    assert(isgreater(-1.0, 0.F) == false);
}

void test_isgreaterequal() {
#ifdef isgreaterequal
#error isgreaterequal defined
#endif
    ASSERT_SAME_TYPE(decltype(isgreaterequal(Ambiguous(), Ambiguous())), Ambiguous);
    assert(isgreaterequal(-1.0, 0.F) == false);
}

void test_isinf() {
#ifdef isinf
#error isinf defined
#endif
    ASSERT_SAME_TYPE(decltype(isinf((float)0)), bool);

    typedef decltype(isinf((double)0)) DoubleRetType;
#ifndef __linux__
    ASSERT_SAME_TYPE(DoubleRetType, bool);
#else
    // GLIBC < 2.26 defines 'isinf(double)' with a return type of 'int' in
    // all C++ dialects. The test should tolerate this.
    // See: https://sourceware.org/bugzilla/show_bug.cgi?id=19439
    static_assert((std::is_same<DoubleRetType, bool>::value
                || std::is_same<DoubleRetType, int>::value), "");
#endif

    ASSERT_SAME_TYPE(decltype(isinf(0)), bool);
    ASSERT_SAME_TYPE(decltype(isinf((long double)0)), bool);
    assert(isinf(-1.0) == false);
}

void test_isless() {
#ifdef isless
#error isless defined
#endif
    ASSERT_SAME_TYPE(decltype(isless(Ambiguous(), Ambiguous())), Ambiguous);
    assert(isless(-1.0, 0.F) == true);
}

void test_islessequal() {
#ifdef islessequal
#error islessequal defined
#endif
    ASSERT_SAME_TYPE(decltype(islessequal(Ambiguous(), Ambiguous())), Ambiguous);
    assert(islessequal(-1.0, 0.F) == true);
}

void test_islessgreater() {
#ifdef islessgreater
#error islessgreater defined
#endif
    ASSERT_SAME_TYPE(decltype(islessgreater(Ambiguous(), Ambiguous())), Ambiguous);
    assert(islessgreater(-1.0, 0.F) == true);
}

void test_isnan() {
#ifdef isnan
#error isnan defined
#endif
    ASSERT_SAME_TYPE(decltype(isnan((float)0)), bool);

    typedef decltype(isnan((double)0)) DoubleRetType;
#ifndef __linux__
    ASSERT_SAME_TYPE(DoubleRetType, bool);
#else
    // GLIBC < 2.26 defines 'isnan(double)' with a return type of 'int' in
    // all C++ dialects. The test should tolerate this.
    // See: https://sourceware.org/bugzilla/show_bug.cgi?id=19439
    static_assert((std::is_same<DoubleRetType, bool>::value
                || std::is_same<DoubleRetType, int>::value), "");
#endif

    ASSERT_SAME_TYPE(decltype(isnan(0)), bool);
    ASSERT_SAME_TYPE(decltype(isnan((long double)0)), bool);
    assert(isnan(-1.0) == false);
}

void test_isunordered() {
#ifdef isunordered
#error isunordered defined
#endif
    ASSERT_SAME_TYPE(decltype(isunordered(Ambiguous(), Ambiguous())), Ambiguous);
    assert(isunordered(-1.0, 0.F) == false);
}

void test_acosh() {
    ASSERT_SAME_TYPE(decltype(acoshf(0)), float);
    ASSERT_SAME_TYPE(decltype(acoshl(0)), long double);
    ASSERT_SAME_TYPE(decltype(acosh(Ambiguous())), Ambiguous);
    assert(acosh(1) == 0);
}

void test_asinh() {
    ASSERT_SAME_TYPE(decltype(asinhf(0)), float);
    ASSERT_SAME_TYPE(decltype(asinhl(0)), long double);
    ASSERT_SAME_TYPE(decltype(asinh(Ambiguous())), Ambiguous);
    assert(asinh(0) == 0);
}

void test_atanh() {
    ASSERT_SAME_TYPE(decltype(atanhf(0)), float);
    ASSERT_SAME_TYPE(decltype(atanhl(0)), long double);
    ASSERT_SAME_TYPE(decltype(atanh(Ambiguous())), Ambiguous);
    assert(atanh(0) == 0);
}

void test_cbrt() {
    ASSERT_SAME_TYPE(decltype(cbrtf(0)), float);
    ASSERT_SAME_TYPE(decltype(cbrtl(0)), long double);
    ASSERT_SAME_TYPE(decltype(cbrt(Ambiguous())), Ambiguous);
    assert(truncate_fp(cbrt(1)) == 1);
}

void test_copysign() {
    ASSERT_SAME_TYPE(decltype(copysignf(0,0)), float);
    ASSERT_SAME_TYPE(decltype(copysignl(0,0)), long double);
    ASSERT_SAME_TYPE(decltype(copysign((int)0, (int)0)), double);
    ASSERT_SAME_TYPE(decltype(copysign(Ambiguous(), Ambiguous())), Ambiguous);
    assert(copysign(1,1) == 1);
}

void test_erf() {
    ASSERT_SAME_TYPE(decltype(erff(0)), float);
    ASSERT_SAME_TYPE(decltype(erfl(0)), long double);
    ASSERT_SAME_TYPE(decltype(erf(Ambiguous())), Ambiguous);
    assert(erf(0) == 0);
}

void test_erfc() {
    ASSERT_SAME_TYPE(decltype(erfcf(0)), float);
    ASSERT_SAME_TYPE(decltype(erfcl(0)), long double);
    ASSERT_SAME_TYPE(decltype(erfc(Ambiguous())), Ambiguous);
    assert(erfc(0) == 1);
}

void test_exp2() {
    ASSERT_SAME_TYPE(decltype(exp2f(0)), float);
    ASSERT_SAME_TYPE(decltype(exp2l(0)), long double);
    ASSERT_SAME_TYPE(decltype(exp2(Ambiguous())), Ambiguous);
    assert(exp2(1) == 2);
}

void test_expm1() {
    ASSERT_SAME_TYPE(decltype(expm1f(0)), float);
    ASSERT_SAME_TYPE(decltype(expm1l(0)), long double);
    ASSERT_SAME_TYPE(decltype(expm1(Ambiguous())), Ambiguous);
    assert(expm1(0) == 0);
}

void test_fdim() {
    ASSERT_SAME_TYPE(decltype(fdimf(0,0)), float);
    ASSERT_SAME_TYPE(decltype(fdiml(0,0)), long double);
    ASSERT_SAME_TYPE(decltype(fdim((int)0, (int)0)), double);
    ASSERT_SAME_TYPE(decltype(fdim(Ambiguous(), Ambiguous())), Ambiguous);
    assert(fdim(1,0) == 1);
}

void test_fma() {
    ASSERT_SAME_TYPE(decltype(fma((bool)0, (float)0, (float)0)), double);
    ASSERT_SAME_TYPE(decltype(fma((float)0, (float)0, (double)0)), double);
    ASSERT_SAME_TYPE(decltype(fma((float)0, (float)0, (long double)0)), long double);
    ASSERT_SAME_TYPE(decltype(fma((float)0, (float)0, (float)0)), float);

    ASSERT_SAME_TYPE(decltype(fma((bool)0, (double)0, (double)0)), double);
    ASSERT_SAME_TYPE(decltype(fma((double)0, (double)0, (float)0)), double);
    ASSERT_SAME_TYPE(decltype(fma((double)0, (double)0, (long double)0)), long double);
    ASSERT_SAME_TYPE(decltype(fma((double)0, (double)0,  (double)0)), double);

    ASSERT_SAME_TYPE(decltype(fma((long double)0, (long double)0, (float)0)), long double);
    ASSERT_SAME_TYPE(decltype(fma((double)0, (long double)0, (long double)0)), long double);
    ASSERT_SAME_TYPE(decltype(fma((long double)0, (long double)0, (long double)0)), long double);

    ASSERT_SAME_TYPE(decltype(fmaf(0,0,0)), float);
    ASSERT_SAME_TYPE(decltype(fmal(0,0,0)), long double);
    ASSERT_SAME_TYPE(decltype(fma(Ambiguous(), Ambiguous(), Ambiguous())), Ambiguous);
    assert(fma(1,1,1) == 2);
}

void test_fmax() {
    ASSERT_SAME_TYPE(decltype(fmaxf(0,0)), float);
    ASSERT_SAME_TYPE(decltype(fmaxl(0,0)), long double);
    ASSERT_SAME_TYPE(decltype(fmax((int)0, (int)0)), double);
    ASSERT_SAME_TYPE(decltype(fmax(Ambiguous(), Ambiguous())), Ambiguous);
    assert(fmax(1,0) == 1);
}

void test_fmin() {
    ASSERT_SAME_TYPE(decltype(fminf(0,0)), float);
    ASSERT_SAME_TYPE(decltype(fminl(0,0)), long double);
    ASSERT_SAME_TYPE(decltype(fmin((int)0, (int)0)), double);
    ASSERT_SAME_TYPE(decltype(fmin(Ambiguous(), Ambiguous())), Ambiguous);
    assert(fmin(1,0) == 0);
}

void test_hypot() {
    ASSERT_SAME_TYPE(decltype(hypotf(0,0)), float);
    ASSERT_SAME_TYPE(decltype(hypotl(0,0)), long double);
    ASSERT_SAME_TYPE(decltype(hypot((int)0, (int)0)), double);
    ASSERT_SAME_TYPE(decltype(hypot(Ambiguous(), Ambiguous())), Ambiguous);
    assert(hypot(3,4) == 5);
}

void test_ilogb() {
    ASSERT_SAME_TYPE(decltype(ilogbf(0)), int);
    ASSERT_SAME_TYPE(decltype(ilogbl(0)), int);
    ASSERT_SAME_TYPE(decltype(ilogb(Ambiguous())), Ambiguous);
    assert(ilogb(1) == 0);
}

void test_lgamma() {
    ASSERT_SAME_TYPE(decltype(lgammaf(0)), float);
    ASSERT_SAME_TYPE(decltype(lgammal(0)), long double);
    ASSERT_SAME_TYPE(decltype(lgamma(Ambiguous())), Ambiguous);
    assert(lgamma(1) == 0);
}

void test_llrint() {
    ASSERT_SAME_TYPE(decltype(llrintf(0)), long long);
    ASSERT_SAME_TYPE(decltype(llrintl(0)), long long);
    ASSERT_SAME_TYPE(decltype(llrint(Ambiguous())), Ambiguous);
    assert(llrint(1) == 1LL);
}

void test_llround() {
    ASSERT_SAME_TYPE(decltype(llroundf(0)), long long);
    ASSERT_SAME_TYPE(decltype(llroundl(0)), long long);
    ASSERT_SAME_TYPE(decltype(llround(Ambiguous())), Ambiguous);
    assert(llround(1) == 1LL);
}

void test_log1p() {
    ASSERT_SAME_TYPE(decltype(log1pf(0)), float);
    ASSERT_SAME_TYPE(decltype(log1pl(0)), long double);
    ASSERT_SAME_TYPE(decltype(log1p(Ambiguous())), Ambiguous);
    assert(log1p(0) == 0);
}

void test_log2() {
    ASSERT_SAME_TYPE(decltype(log2f(0)), float);
    ASSERT_SAME_TYPE(decltype(log2l(0)), long double);
    ASSERT_SAME_TYPE(decltype(log2(Ambiguous())), Ambiguous);
    assert(log2(1) == 0);
}

void test_logb() {
    ASSERT_SAME_TYPE(decltype(logbf(0)), float);
    ASSERT_SAME_TYPE(decltype(logbl(0)), long double);
    ASSERT_SAME_TYPE(decltype(logb(Ambiguous())), Ambiguous);
    assert(logb(1) == 0);
}

void test_lrint() {
    ASSERT_SAME_TYPE(decltype(lrintf(0)), long);
    ASSERT_SAME_TYPE(decltype(lrintl(0)), long);
    ASSERT_SAME_TYPE(decltype(lrint(Ambiguous())), Ambiguous);
    assert(lrint(1) == 1L);
}

void test_lround() {
    ASSERT_SAME_TYPE(decltype(lroundf(0)), long);
    ASSERT_SAME_TYPE(decltype(lroundl(0)), long);
    ASSERT_SAME_TYPE(decltype(lround(Ambiguous())), Ambiguous);
    assert(lround(1) == 1L);
}

void test_nan() {
    ASSERT_SAME_TYPE(decltype(nan("")), double);
    ASSERT_SAME_TYPE(decltype(nanf("")), float);
    ASSERT_SAME_TYPE(decltype(nanl("")), long double);
}

void test_nearbyint() {
    ASSERT_SAME_TYPE(decltype(nearbyintf(0)), float);
    ASSERT_SAME_TYPE(decltype(nearbyintl(0)), long double);
    ASSERT_SAME_TYPE(decltype(nearbyint(Ambiguous())), Ambiguous);
    assert(nearbyint(1) == 1);
}

void test_nextafter() {
    ASSERT_SAME_TYPE(decltype(nextafterf(0,0)), float);
    ASSERT_SAME_TYPE(decltype(nextafterl(0,0)), long double);
    ASSERT_SAME_TYPE(decltype(nextafter((int)0, (int)0)), double);
    ASSERT_SAME_TYPE(decltype(nextafter(Ambiguous(), Ambiguous())), Ambiguous);
    assert(nextafter(0,1) == hexfloat<double>(0x1, 0, -1074));
}

void test_nexttoward() {
    ASSERT_SAME_TYPE(decltype(nexttoward((float)0, (long double)0)), float);
    ASSERT_SAME_TYPE(decltype(nexttoward((bool)0, (long double)0)), double);
    ASSERT_SAME_TYPE(decltype(nexttoward((unsigned short)0, (long double)0)), double);
    ASSERT_SAME_TYPE(decltype(nexttoward((int)0, (long double)0)), double);
    ASSERT_SAME_TYPE(decltype(nexttoward((unsigned int)0, (long double)0)), double);
    ASSERT_SAME_TYPE(decltype(nexttoward((long)0, (long double)0)), double);
    ASSERT_SAME_TYPE(decltype(nexttoward((unsigned long)0, (long double)0)), double);
    ASSERT_SAME_TYPE(decltype(nexttoward((long long)0, (long double)0)), double);
    ASSERT_SAME_TYPE(decltype(nexttoward((unsigned long long)0, (long double)0)), double);
    ASSERT_SAME_TYPE(decltype(nexttoward((double)0, (long double)0)), double);
    ASSERT_SAME_TYPE(decltype(nexttoward((long double)0, (long double)0)), long double);
    ASSERT_SAME_TYPE(decltype(nexttowardf(0, (long double)0)), float);
    ASSERT_SAME_TYPE(decltype(nexttowardl(0, (long double)0)), long double);
    ASSERT_SAME_TYPE(decltype(nexttoward(Ambiguous(), Ambiguous())), Ambiguous);
    assert(nexttoward(0, 1) == hexfloat<double>(0x1, 0, -1074));
}

void test_remainder() {
    ASSERT_SAME_TYPE(decltype(remainderf(0,0)), float);
    ASSERT_SAME_TYPE(decltype(remainderl(0,0)), long double);
    ASSERT_SAME_TYPE(decltype(remainder((int)0, (int)0)), double);
    ASSERT_SAME_TYPE(decltype(remainder(Ambiguous(), Ambiguous())), Ambiguous);
    assert(remainder(0.5,1) == 0.5);
}

void test_remquo() {
    int ip;
    ASSERT_SAME_TYPE(decltype(remquof(0,0, &ip)), float);
    ASSERT_SAME_TYPE(decltype(remquol(0,0, &ip)), long double);
    ASSERT_SAME_TYPE(decltype(remquo((int)0, (int)0, &ip)), double);
    ASSERT_SAME_TYPE(decltype(remquo(Ambiguous(), Ambiguous(), &ip)), Ambiguous);
    assert(remquo(0.5,1, &ip) == 0.5);
}

void test_rint() {
    ASSERT_SAME_TYPE(decltype(rintf(0)), float);
    ASSERT_SAME_TYPE(decltype(rintl(0)), long double);
    ASSERT_SAME_TYPE(decltype(rint(Ambiguous())), Ambiguous);
    assert(rint(1) == 1);
}

void test_round() {
    ASSERT_SAME_TYPE(decltype(roundf(0)), float);
    ASSERT_SAME_TYPE(decltype(roundl(0)), long double);
    ASSERT_SAME_TYPE(decltype(round(Ambiguous())), Ambiguous);
    assert(round(1) == 1);
}

void test_scalbln() {
    ASSERT_SAME_TYPE(decltype(scalbln((float)0, (long)0)), float);
    ASSERT_SAME_TYPE(decltype(scalbln((bool)0, (long)0)), double);
    ASSERT_SAME_TYPE(decltype(scalbln((unsigned short)0, (long)0)), double);
    ASSERT_SAME_TYPE(decltype(scalbln((int)0, (long)0)), double);
    ASSERT_SAME_TYPE(decltype(scalbln((unsigned int)0, (long)0)), double);
    ASSERT_SAME_TYPE(decltype(scalbln((long)0, (long)0)), double);
    ASSERT_SAME_TYPE(decltype(scalbln((unsigned long)0, (long)0)), double);
    ASSERT_SAME_TYPE(decltype(scalbln((long long)0, (long)0)), double);
    ASSERT_SAME_TYPE(decltype(scalbln((unsigned long long)0, (long)0)), double);
    ASSERT_SAME_TYPE(decltype(scalbln((double)0, (long)0)), double);
    ASSERT_SAME_TYPE(decltype(scalbln((long double)0, (long)0)), long double);
    ASSERT_SAME_TYPE(decltype(scalblnf(0, (long)0)), float);
    ASSERT_SAME_TYPE(decltype(scalblnl(0, (long)0)), long double);
    ASSERT_SAME_TYPE(decltype(scalbln(Ambiguous(), Ambiguous())), Ambiguous);
    assert(scalbln(1, 1) == 2);
}

void test_scalbn() {
    ASSERT_SAME_TYPE(decltype(scalbn((float)0, (int)0)), float);
    ASSERT_SAME_TYPE(decltype(scalbn((bool)0, (int)0)), double);
    ASSERT_SAME_TYPE(decltype(scalbn((unsigned short)0, (int)0)), double);
    ASSERT_SAME_TYPE(decltype(scalbn((int)0, (int)0)), double);
    ASSERT_SAME_TYPE(decltype(scalbn((unsigned int)0, (int)0)), double);
    ASSERT_SAME_TYPE(decltype(scalbn((long)0, (int)0)), double);
    ASSERT_SAME_TYPE(decltype(scalbn((unsigned long)0, (int)0)), double);
    ASSERT_SAME_TYPE(decltype(scalbn((long long)0, (int)0)), double);
    ASSERT_SAME_TYPE(decltype(scalbn((unsigned long long)0, (int)0)), double);
    ASSERT_SAME_TYPE(decltype(scalbn((double)0, (int)0)), double);
    ASSERT_SAME_TYPE(decltype(scalbn((long double)0, (int)0)), long double);
    ASSERT_SAME_TYPE(decltype(scalbnf(0, (int)0)), float);
    ASSERT_SAME_TYPE(decltype(scalbnl(0, (int)0)), long double);
    ASSERT_SAME_TYPE(decltype(scalbn(Ambiguous(), Ambiguous())), Ambiguous);
    assert(scalbn(1, 1) == 2);
}

void test_tgamma() {
    ASSERT_SAME_TYPE(decltype(tgammaf(0)), float);
    ASSERT_SAME_TYPE(decltype(tgammal(0)), long double);
    ASSERT_SAME_TYPE(decltype(tgamma(Ambiguous())), Ambiguous);
    assert(tgamma(1) == 1);
}

void test_trunc() {
    ASSERT_SAME_TYPE(decltype(truncf(0)), float);
    ASSERT_SAME_TYPE(decltype(truncl(0)), long double);
    ASSERT_SAME_TYPE(decltype(trunc(Ambiguous())), Ambiguous);
    assert(trunc(1) == 1);
}

template <class PromoteResult, class Arg = void>
struct test_single_arg {
  template <class T = Arg>
  void operator()() {
    ASSERT_SAME_TYPE(decltype(::acos(T())), PromoteResult);
    (void)::acos(T());
    ASSERT_SAME_TYPE(decltype(::asin(T())), PromoteResult);
    (void)::asin(T());
    ASSERT_SAME_TYPE(decltype(::atan(T())), PromoteResult);
    (void)::atan(T());
    ASSERT_SAME_TYPE(decltype(::ceil(T())), PromoteResult);
    (void)::ceil(T());
    ASSERT_SAME_TYPE(decltype(::cos(T())), PromoteResult);
    (void)::cos(T());
    ASSERT_SAME_TYPE(decltype(::cosh(T())), PromoteResult);
    (void)::cosh(T());
    ASSERT_SAME_TYPE(decltype(::exp(T())), PromoteResult);
    (void)::exp(T());
    ASSERT_SAME_TYPE(decltype(::fabs(T())), PromoteResult);
    (void)::fabs(T());
    ASSERT_SAME_TYPE(decltype(::floor(T())), PromoteResult);
    (void)::floor(T());
    int ip;
    ASSERT_SAME_TYPE(decltype(::frexp(T(), &ip)), PromoteResult);
    (void)::frexp(T(), &ip);
    ASSERT_SAME_TYPE(decltype(::ldexp(T(), ip)), PromoteResult);
    (void)::ldexp(T(), ip);
    ASSERT_SAME_TYPE(decltype(::log(T())), PromoteResult);
    (void)::log(T());
    ASSERT_SAME_TYPE(decltype(::log10(T())), PromoteResult);
    (void)::log10(T());
    ASSERT_SAME_TYPE(decltype(::sin(T())), PromoteResult);
    (void)::sin(T());
    ASSERT_SAME_TYPE(decltype(::sinh(T())), PromoteResult);
    (void)::sinh(T());
    ASSERT_SAME_TYPE(decltype(::sqrt(T())), PromoteResult);
    (void)::sqrt(T());
    ASSERT_SAME_TYPE(decltype(::tan(T())), PromoteResult);
    (void)::tan(T());
    ASSERT_SAME_TYPE(decltype(::tanh(T())), PromoteResult);
    (void)::tanh(T());
    ASSERT_SAME_TYPE(decltype(::signbit(T())), bool);
    (void)::signbit(T());
    ASSERT_SAME_TYPE(decltype(::fpclassify(T())), int);
    (void)::fpclassify(T());
    ASSERT_SAME_TYPE(decltype(::isfinite(T())), bool);
    (void)::isfinite(T());
    ASSERT_SAME_TYPE(decltype(::isnormal(T())), bool);
    (void)::isnormal(T());
    ASSERT_SAME_TYPE(decltype(::acosh(T())), PromoteResult);
    (void)::acosh(T());
    ASSERT_SAME_TYPE(decltype(::asinh(T())), PromoteResult);
    (void)::asinh(T());
    ASSERT_SAME_TYPE(decltype(::atanh(T())), PromoteResult);
    (void)::atanh(T());
    ASSERT_SAME_TYPE(decltype(::cbrt(T())), PromoteResult);
    (void)::cbrt(T());
    ASSERT_SAME_TYPE(decltype(::erf(T())), PromoteResult);
    (void)::erf(T());
    ASSERT_SAME_TYPE(decltype(::erfc(T())), PromoteResult);
    (void)::erfc(T());
    ASSERT_SAME_TYPE(decltype(::exp2(T())), PromoteResult);
    (void)::exp2(T());
    ASSERT_SAME_TYPE(decltype(::expm1(T())), PromoteResult);
    (void)::expm1(T());
    ASSERT_SAME_TYPE(decltype(::ilogb(T())), int);
    (void)::ilogb(T());
    ASSERT_SAME_TYPE(decltype(::lgamma(T())), PromoteResult);
    (void)::lgamma(T());
    ASSERT_SAME_TYPE(decltype(::llrint(T())), long long);
    (void)::llrint(T());
    ASSERT_SAME_TYPE(decltype(::llround(T())), long long);
    (void)::llround(T());
    ASSERT_SAME_TYPE(decltype(::log1p(T())), PromoteResult);
    (void)::log1p(T());
    ASSERT_SAME_TYPE(decltype(::log2(T())), PromoteResult);
    (void)::log2(T());
    ASSERT_SAME_TYPE(decltype(::logb(T())), PromoteResult);
    (void)::logb(T());
    ASSERT_SAME_TYPE(decltype(::lrint(T())), long);
    (void)::lrint(T());
    ASSERT_SAME_TYPE(decltype(::lround(T())), long);
    (void)::lround(T());
    ASSERT_SAME_TYPE(decltype(::nearbyint(T())), PromoteResult);
    (void)::nearbyint(T());
    ASSERT_SAME_TYPE(decltype(::rint(T())), PromoteResult);
    (void)::rint(T());
    ASSERT_SAME_TYPE(decltype(::round(T())), PromoteResult);
    (void)::round(T());
    ASSERT_SAME_TYPE(decltype(::trunc(T())), PromoteResult);
    (void)::trunc(T());
    ASSERT_SAME_TYPE(decltype(::tgamma(T())), PromoteResult);
    (void)::tgamma(T());
  }
};

template <class PromoteResult, class Arg1 = void, class Arg2 = void>
struct test_two_args {
  template <class T = Arg1, class U = Arg2>
  void operator()() {
    ASSERT_SAME_TYPE(decltype(::atan2(T(), U())), PromoteResult);
    (void)::atan2(T(), U());
    ASSERT_SAME_TYPE(decltype(::fmod(T(), U())), PromoteResult);
    (void)::fmod(T(), U());
    ASSERT_SAME_TYPE(decltype(::pow(T(), U())), PromoteResult);
    (void)::pow(T(), U());
    ASSERT_SAME_TYPE(decltype(::isgreater(T(), U())), bool);
    (void)::isgreater(T(), U());
    ASSERT_SAME_TYPE(decltype(::isgreaterequal(T(), U())), bool);
    (void)::isgreaterequal(T(), U());
    ASSERT_SAME_TYPE(decltype(::isless(T(), U())), bool);
    (void)::isless(T(), U());
    ASSERT_SAME_TYPE(decltype(::islessequal(T(), U())), bool);
    (void)::islessequal(T(), U());
    ASSERT_SAME_TYPE(decltype(::islessgreater(T(), U())), bool);
    (void)::islessgreater(T(), U());
    ASSERT_SAME_TYPE(decltype(::isunordered(T(), U())), bool);
    (void)::isunordered(T(), U());
    ASSERT_SAME_TYPE(decltype(::copysign(T(), U())), PromoteResult);
    (void)::copysign(T(), U());
    ASSERT_SAME_TYPE(decltype(::fdim(T(), U())), PromoteResult);
    (void)::fdim(T(), U());
    ASSERT_SAME_TYPE(decltype(::fmax(T(), U())), PromoteResult);
    (void)::fmax(T(), U());
    ASSERT_SAME_TYPE(decltype(::fmin(T(), U())), PromoteResult);
    (void)::fmin(T(), U());
    ASSERT_SAME_TYPE(decltype(::hypot(T(), U())), PromoteResult);
    (void)::hypot(T(), U());
    ASSERT_SAME_TYPE(decltype(::nextafter(T(), U())), PromoteResult);
    (void)::nextafter(T(), U());
    ASSERT_SAME_TYPE(decltype(::remainder(T(), U())), PromoteResult);
    (void)::remainder(T(), U());
    int ip;
    ASSERT_SAME_TYPE(decltype(::remquo(T(), U(), &ip)), PromoteResult);
    ::remquo(T(), U(), &ip);
  }
};

template <class PromoteResult, class Arg1 = void, class Arg2 = void, class Arg3 = void>
struct test_three_args {
  template <class T = Arg1, class U = Arg2, class V = Arg3>
  void operator()() {
    ASSERT_SAME_TYPE(decltype(::fma(T(), U(), V())), PromoteResult);
    (void)::fma(T(), U(), V());
  }
};

struct CallTwoArgs {
  using integral_float_double = types::concatenate_t<types::integral_types, types::type_list<float, double> >;

  template <class Arg2>
  void operator()() {
    types::for_each(integral_float_double(), test_two_args</*PromoteResult=*/double, /*Iterate*/void, Arg2>());
  }
};

template <class T>
struct CallThreeArgs {
  using integral_float_double = types::concatenate_t<types::integral_types, types::type_list<float, double> >;

  template <class Arg3>
  struct Helper {

    template <class Arg2>
    void operator()() {
      types::for_each(integral_float_double(), test_three_args</*PromoteResult=*/double, /*Iterate*/void, Arg2, Arg3>());
    }
  };

  template <class Arg3>
  void operator()() {
    types::for_each(integral_float_double(), Helper<Arg3>());
  }
};

int main(int, char**) {
  types::for_each(types::integral_types(), test_single_arg</*PromoteResult=*/double>());
  test_single_arg</*PromoteResult=*/float, /*Arg=*/float>();
  test_single_arg</*PromoteResult=*/double, /*Arg=*/double>();
  test_single_arg</*PromoteResult=*/long double, /*Arg=*/long double>();

  types::for_each(types::integral_types(), CallTwoArgs());

  types::for_each(
      types::integral_types(), test_two_args</*PromoteResult=*/long double, /*Arg1=*/void, /*Arg2=*/long double>());

  test_two_args</*PromoteResult=*/float, /*Args=*/float, float>();
  test_two_args</*PromoteResult=*/float, /*Args=*/double, double>();
  test_two_args</*PromoteResult=*/double, /*Args=*/float, double>();
  test_two_args</*PromoteResult=*/double, /*Args=*/double, double>();

  types::for_each(types::integral_types(), CallThreeArgs<double>());
  types::for_each(
      types::integral_types(), test_three_args</*PromoteResult=*/long double, /*Iterate*/ void, long double, double>());

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
