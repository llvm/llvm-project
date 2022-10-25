// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_MATH_H
#define _LIBCPP_MATH_H

/*
    math.h synopsis

Macros:

    HUGE_VAL
    HUGE_VALF               // C99
    HUGE_VALL               // C99
    INFINITY                // C99
    NAN                     // C99
    FP_INFINITE             // C99
    FP_NAN                  // C99
    FP_NORMAL               // C99
    FP_SUBNORMAL            // C99
    FP_ZERO                 // C99
    FP_FAST_FMA             // C99
    FP_FAST_FMAF            // C99
    FP_FAST_FMAL            // C99
    FP_ILOGB0               // C99
    FP_ILOGBNAN             // C99
    MATH_ERRNO              // C99
    MATH_ERREXCEPT          // C99
    math_errhandling        // C99

Types:

    float_t                 // C99
    double_t                // C99

// C90

floating_point abs(floating_point x);

floating_point acos (arithmetic x);
float          acosf(float x);
long double    acosl(long double x);

floating_point asin (arithmetic x);
float          asinf(float x);
long double    asinl(long double x);

floating_point atan (arithmetic x);
float          atanf(float x);
long double    atanl(long double x);

floating_point atan2 (arithmetic y, arithmetic x);
float          atan2f(float y, float x);
long double    atan2l(long double y, long double x);

floating_point ceil (arithmetic x);
float          ceilf(float x);
long double    ceill(long double x);

floating_point cos (arithmetic x);
float          cosf(float x);
long double    cosl(long double x);

floating_point cosh (arithmetic x);
float          coshf(float x);
long double    coshl(long double x);

floating_point exp (arithmetic x);
float          expf(float x);
long double    expl(long double x);

floating_point fabs (arithmetic x);
float          fabsf(float x);
long double    fabsl(long double x);

floating_point floor (arithmetic x);
float          floorf(float x);
long double    floorl(long double x);

floating_point fmod (arithmetic x, arithmetic y);
float          fmodf(float x, float y);
long double    fmodl(long double x, long double y);

floating_point frexp (arithmetic value, int* exp);
float          frexpf(float value, int* exp);
long double    frexpl(long double value, int* exp);

floating_point ldexp (arithmetic value, int exp);
float          ldexpf(float value, int exp);
long double    ldexpl(long double value, int exp);

floating_point log (arithmetic x);
float          logf(float x);
long double    logl(long double x);

floating_point log10 (arithmetic x);
float          log10f(float x);
long double    log10l(long double x);

floating_point modf (floating_point value, floating_point* iptr);
float          modff(float value, float* iptr);
long double    modfl(long double value, long double* iptr);

floating_point pow (arithmetic x, arithmetic y);
float          powf(float x, float y);
long double    powl(long double x, long double y);

floating_point sin (arithmetic x);
float          sinf(float x);
long double    sinl(long double x);

floating_point sinh (arithmetic x);
float          sinhf(float x);
long double    sinhl(long double x);

floating_point sqrt (arithmetic x);
float          sqrtf(float x);
long double    sqrtl(long double x);

floating_point tan (arithmetic x);
float          tanf(float x);
long double    tanl(long double x);

floating_point tanh (arithmetic x);
float          tanhf(float x);
long double    tanhl(long double x);

//  C99

bool signbit(arithmetic x);

int fpclassify(arithmetic x);

bool isfinite(arithmetic x);
bool isinf(arithmetic x);
bool isnan(arithmetic x);
bool isnormal(arithmetic x);

bool isgreater(arithmetic x, arithmetic y);
bool isgreaterequal(arithmetic x, arithmetic y);
bool isless(arithmetic x, arithmetic y);
bool islessequal(arithmetic x, arithmetic y);
bool islessgreater(arithmetic x, arithmetic y);
bool isunordered(arithmetic x, arithmetic y);

floating_point acosh (arithmetic x);
float          acoshf(float x);
long double    acoshl(long double x);

floating_point asinh (arithmetic x);
float          asinhf(float x);
long double    asinhl(long double x);

floating_point atanh (arithmetic x);
float          atanhf(float x);
long double    atanhl(long double x);

floating_point cbrt (arithmetic x);
float          cbrtf(float x);
long double    cbrtl(long double x);

floating_point copysign (arithmetic x, arithmetic y);
float          copysignf(float x, float y);
long double    copysignl(long double x, long double y);

floating_point erf (arithmetic x);
float          erff(float x);
long double    erfl(long double x);

floating_point erfc (arithmetic x);
float          erfcf(float x);
long double    erfcl(long double x);

floating_point exp2 (arithmetic x);
float          exp2f(float x);
long double    exp2l(long double x);

floating_point expm1 (arithmetic x);
float          expm1f(float x);
long double    expm1l(long double x);

floating_point fdim (arithmetic x, arithmetic y);
float          fdimf(float x, float y);
long double    fdiml(long double x, long double y);

floating_point fma (arithmetic x, arithmetic y, arithmetic z);
float          fmaf(float x, float y, float z);
long double    fmal(long double x, long double y, long double z);

floating_point fmax (arithmetic x, arithmetic y);
float          fmaxf(float x, float y);
long double    fmaxl(long double x, long double y);

floating_point fmin (arithmetic x, arithmetic y);
float          fminf(float x, float y);
long double    fminl(long double x, long double y);

floating_point hypot (arithmetic x, arithmetic y);
float          hypotf(float x, float y);
long double    hypotl(long double x, long double y);

int ilogb (arithmetic x);
int ilogbf(float x);
int ilogbl(long double x);

floating_point lgamma (arithmetic x);
float          lgammaf(float x);
long double    lgammal(long double x);

long long llrint (arithmetic x);
long long llrintf(float x);
long long llrintl(long double x);

long long llround (arithmetic x);
long long llroundf(float x);
long long llroundl(long double x);

floating_point log1p (arithmetic x);
float          log1pf(float x);
long double    log1pl(long double x);

floating_point log2 (arithmetic x);
float          log2f(float x);
long double    log2l(long double x);

floating_point logb (arithmetic x);
float          logbf(float x);
long double    logbl(long double x);

long lrint (arithmetic x);
long lrintf(float x);
long lrintl(long double x);

long lround (arithmetic x);
long lroundf(float x);
long lroundl(long double x);

double      nan (const char* str);
float       nanf(const char* str);
long double nanl(const char* str);

floating_point nearbyint (arithmetic x);
float          nearbyintf(float x);
long double    nearbyintl(long double x);

floating_point nextafter (arithmetic x, arithmetic y);
float          nextafterf(float x, float y);
long double    nextafterl(long double x, long double y);

floating_point nexttoward (arithmetic x, long double y);
float          nexttowardf(float x, long double y);
long double    nexttowardl(long double x, long double y);

floating_point remainder (arithmetic x, arithmetic y);
float          remainderf(float x, float y);
long double    remainderl(long double x, long double y);

floating_point remquo (arithmetic x, arithmetic y, int* pquo);
float          remquof(float x, float y, int* pquo);
long double    remquol(long double x, long double y, int* pquo);

floating_point rint (arithmetic x);
float          rintf(float x);
long double    rintl(long double x);

floating_point round (arithmetic x);
float          roundf(float x);
long double    roundl(long double x);

floating_point scalbln (arithmetic x, long ex);
float          scalblnf(float x, long ex);
long double    scalblnl(long double x, long ex);

floating_point scalbn (arithmetic x, int ex);
float          scalbnf(float x, int ex);
long double    scalbnl(long double x, int ex);

floating_point tgamma (arithmetic x);
float          tgammaf(float x);
long double    tgammal(long double x);

floating_point trunc (arithmetic x);
float          truncf(float x);
long double    truncl(long double x);

*/

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#  if __has_include_next(<math.h>)
#    include_next <math.h>
#  endif

#ifdef __cplusplus

// We support including .h headers inside 'extern "C"' contexts, so switch
// back to C++ linkage before including these C++ headers.
extern "C++" {

#include <__type_traits/promote.h>
#include <limits>
#include <stdlib.h>
#include <type_traits>

// signbit

#ifdef signbit

template <class _A1>
_LIBCPP_HIDE_FROM_ABI
bool
__libcpp_signbit(_A1 __x) _NOEXCEPT
{
#if __has_builtin(__builtin_signbit)
    return __builtin_signbit(__x);
#else
    return signbit(__x);
#endif
}

#undef signbit

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_floating_point<_A1>::value, bool>::type
signbit(_A1 __x) _NOEXCEPT
{
    return __libcpp_signbit(__x);
}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<
    std::is_integral<_A1>::value && std::is_signed<_A1>::value, bool>::type
signbit(_A1 __x) _NOEXCEPT
{ return __x < 0; }

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<
    std::is_integral<_A1>::value && !std::is_signed<_A1>::value, bool>::type
signbit(_A1) _NOEXCEPT
{ return false; }

#elif defined(_LIBCPP_MSVCRT)

template <typename _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_floating_point<_A1>::value, bool>::type
signbit(_A1 __x) _NOEXCEPT
{
  return ::signbit(__x);
}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<
    std::is_integral<_A1>::value && std::is_signed<_A1>::value, bool>::type
signbit(_A1 __x) _NOEXCEPT
{ return __x < 0; }

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<
    std::is_integral<_A1>::value && !std::is_signed<_A1>::value, bool>::type
signbit(_A1) _NOEXCEPT
{ return false; }

#endif // signbit

// fpclassify

#ifdef fpclassify

template <class _A1>
_LIBCPP_HIDE_FROM_ABI
int
__libcpp_fpclassify(_A1 __x) _NOEXCEPT
{
#if __has_builtin(__builtin_fpclassify)
  return __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL,
                                FP_ZERO, __x);
#else
    return fpclassify(__x);
#endif
}

#undef fpclassify

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_floating_point<_A1>::value, int>::type
fpclassify(_A1 __x) _NOEXCEPT
{
    return __libcpp_fpclassify(__x);
}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, int>::type
fpclassify(_A1 __x) _NOEXCEPT
{ return __x == 0 ? FP_ZERO : FP_NORMAL; }

#elif defined(_LIBCPP_MSVCRT)

template <typename _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_floating_point<_A1>::value, bool>::type
fpclassify(_A1 __x) _NOEXCEPT
{
  return ::fpclassify(__x);
}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, int>::type
fpclassify(_A1 __x) _NOEXCEPT
{ return __x == 0 ? FP_ZERO : FP_NORMAL; }

#endif // fpclassify

// isfinite

#ifdef isfinite

template <class _A1>
_LIBCPP_HIDE_FROM_ABI
bool
__libcpp_isfinite(_A1 __x) _NOEXCEPT
{
#if __has_builtin(__builtin_isfinite)
    return __builtin_isfinite(__x);
#else
    return isfinite(__x);
#endif
}

#undef isfinite

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<
    std::is_arithmetic<_A1>::value && std::numeric_limits<_A1>::has_infinity,
    bool>::type
isfinite(_A1 __x) _NOEXCEPT
{
    return __libcpp_isfinite((typename std::__promote<_A1>::type)__x);
}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<
    std::is_arithmetic<_A1>::value && !std::numeric_limits<_A1>::has_infinity,
    bool>::type
isfinite(_A1) _NOEXCEPT
{ return true; }

#endif // isfinite

// isinf

#ifdef isinf

template <class _A1>
_LIBCPP_HIDE_FROM_ABI
bool
__libcpp_isinf(_A1 __x) _NOEXCEPT
{
#if __has_builtin(__builtin_isinf)
    return __builtin_isinf(__x);
#else
    return isinf(__x);
#endif
}

#undef isinf

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<
    std::is_arithmetic<_A1>::value && std::numeric_limits<_A1>::has_infinity,
    bool>::type
isinf(_A1 __x) _NOEXCEPT
{
    return __libcpp_isinf((typename std::__promote<_A1>::type)__x);
}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<
    std::is_arithmetic<_A1>::value && !std::numeric_limits<_A1>::has_infinity,
    bool>::type
isinf(_A1) _NOEXCEPT
{ return false; }

#ifdef _LIBCPP_PREFERRED_OVERLOAD
inline _LIBCPP_HIDE_FROM_ABI
bool
isinf(float __x) _NOEXCEPT { return __libcpp_isinf(__x); }

inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_PREFERRED_OVERLOAD
bool
isinf(double __x) _NOEXCEPT { return __libcpp_isinf(__x); }

inline _LIBCPP_HIDE_FROM_ABI
bool
isinf(long double __x) _NOEXCEPT { return __libcpp_isinf(__x); }
#endif

#endif // isinf

// isnan

#ifdef isnan

template <class _A1>
_LIBCPP_HIDE_FROM_ABI
bool
__libcpp_isnan(_A1 __x) _NOEXCEPT
{
#if __has_builtin(__builtin_isnan)
    return __builtin_isnan(__x);
#else
    return isnan(__x);
#endif
}

#undef isnan

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_floating_point<_A1>::value, bool>::type
isnan(_A1 __x) _NOEXCEPT
{
    return __libcpp_isnan(__x);
}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, bool>::type
isnan(_A1) _NOEXCEPT
{ return false; }

#ifdef _LIBCPP_PREFERRED_OVERLOAD
inline _LIBCPP_HIDE_FROM_ABI
bool
isnan(float __x) _NOEXCEPT { return __libcpp_isnan(__x); }

inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_PREFERRED_OVERLOAD
bool
isnan(double __x) _NOEXCEPT { return __libcpp_isnan(__x); }

inline _LIBCPP_HIDE_FROM_ABI
bool
isnan(long double __x) _NOEXCEPT { return __libcpp_isnan(__x); }
#endif

#endif // isnan

// isnormal

#ifdef isnormal

template <class _A1>
_LIBCPP_HIDE_FROM_ABI
bool
__libcpp_isnormal(_A1 __x) _NOEXCEPT
{
#if __has_builtin(__builtin_isnormal)
    return __builtin_isnormal(__x);
#else
    return isnormal(__x);
#endif
}

#undef isnormal

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_floating_point<_A1>::value, bool>::type
isnormal(_A1 __x) _NOEXCEPT
{
    return __libcpp_isnormal(__x);
}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, bool>::type
isnormal(_A1 __x) _NOEXCEPT
{ return __x != 0; }

#endif // isnormal

// isgreater

#ifdef isgreater

template <class _A1, class _A2>
_LIBCPP_HIDE_FROM_ABI
bool
__libcpp_isgreater(_A1 __x, _A2 __y) _NOEXCEPT
{
    return isgreater(__x, __y);
}

#undef isgreater

template <class _A1, class _A2>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if
<
    std::is_arithmetic<_A1>::value &&
    std::is_arithmetic<_A2>::value,
    bool
>::type
isgreater(_A1 __x, _A2 __y) _NOEXCEPT
{
    typedef typename std::__promote<_A1, _A2>::type type;
    return __libcpp_isgreater((type)__x, (type)__y);
}

#endif // isgreater

// isgreaterequal

#ifdef isgreaterequal

template <class _A1, class _A2>
_LIBCPP_HIDE_FROM_ABI
bool
__libcpp_isgreaterequal(_A1 __x, _A2 __y) _NOEXCEPT
{
    return isgreaterequal(__x, __y);
}

#undef isgreaterequal

template <class _A1, class _A2>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if
<
    std::is_arithmetic<_A1>::value &&
    std::is_arithmetic<_A2>::value,
    bool
>::type
isgreaterequal(_A1 __x, _A2 __y) _NOEXCEPT
{
    typedef typename std::__promote<_A1, _A2>::type type;
    return __libcpp_isgreaterequal((type)__x, (type)__y);
}

#endif // isgreaterequal

// isless

#ifdef isless

template <class _A1, class _A2>
_LIBCPP_HIDE_FROM_ABI
bool
__libcpp_isless(_A1 __x, _A2 __y) _NOEXCEPT
{
    return isless(__x, __y);
}

#undef isless

template <class _A1, class _A2>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if
<
    std::is_arithmetic<_A1>::value &&
    std::is_arithmetic<_A2>::value,
    bool
>::type
isless(_A1 __x, _A2 __y) _NOEXCEPT
{
    typedef typename std::__promote<_A1, _A2>::type type;
    return __libcpp_isless((type)__x, (type)__y);
}

#endif // isless

// islessequal

#ifdef islessequal

template <class _A1, class _A2>
_LIBCPP_HIDE_FROM_ABI
bool
__libcpp_islessequal(_A1 __x, _A2 __y) _NOEXCEPT
{
    return islessequal(__x, __y);
}

#undef islessequal

template <class _A1, class _A2>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if
<
    std::is_arithmetic<_A1>::value &&
    std::is_arithmetic<_A2>::value,
    bool
>::type
islessequal(_A1 __x, _A2 __y) _NOEXCEPT
{
    typedef typename std::__promote<_A1, _A2>::type type;
    return __libcpp_islessequal((type)__x, (type)__y);
}

#endif // islessequal

// islessgreater

#ifdef islessgreater

template <class _A1, class _A2>
_LIBCPP_HIDE_FROM_ABI
bool
__libcpp_islessgreater(_A1 __x, _A2 __y) _NOEXCEPT
{
    return islessgreater(__x, __y);
}

#undef islessgreater

template <class _A1, class _A2>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if
<
    std::is_arithmetic<_A1>::value &&
    std::is_arithmetic<_A2>::value,
    bool
>::type
islessgreater(_A1 __x, _A2 __y) _NOEXCEPT
{
    typedef typename std::__promote<_A1, _A2>::type type;
    return __libcpp_islessgreater((type)__x, (type)__y);
}

#endif // islessgreater

// isunordered

#ifdef isunordered

template <class _A1, class _A2>
_LIBCPP_HIDE_FROM_ABI
bool
__libcpp_isunordered(_A1 __x, _A2 __y) _NOEXCEPT
{
    return isunordered(__x, __y);
}

#undef isunordered

template <class _A1, class _A2>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if
<
    std::is_arithmetic<_A1>::value &&
    std::is_arithmetic<_A2>::value,
    bool
>::type
isunordered(_A1 __x, _A2 __y) _NOEXCEPT
{
    typedef typename std::__promote<_A1, _A2>::type type;
    return __libcpp_isunordered((type)__x, (type)__y);
}

#endif // isunordered

// abs
//
// handled in stdlib.h

// div
//
// handled in stdlib.h

// acos

#    if !defined(__sun__)
inline _LIBCPP_HIDE_FROM_ABI float       acos(float __x) _NOEXCEPT       {return ::acosf(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double acos(long double __x) _NOEXCEPT {return ::acosl(__x);}
#    endif

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
acos(_A1 __x) _NOEXCEPT {return ::acos((double)__x);}

// asin

#    if !defined(__sun__)
inline _LIBCPP_HIDE_FROM_ABI float       asin(float __x) _NOEXCEPT       {return ::asinf(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double asin(long double __x) _NOEXCEPT {return ::asinl(__x);}
#    endif

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
asin(_A1 __x) _NOEXCEPT {return ::asin((double)__x);}

// atan

#    if !defined(__sun__)
inline _LIBCPP_HIDE_FROM_ABI float       atan(float __x) _NOEXCEPT       {return ::atanf(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double atan(long double __x) _NOEXCEPT {return ::atanl(__x);}
#    endif

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
atan(_A1 __x) _NOEXCEPT {return ::atan((double)__x);}

// atan2

#    if !defined(__sun__)
inline _LIBCPP_HIDE_FROM_ABI float       atan2(float __y, float __x) _NOEXCEPT             {return ::atan2f(__y, __x);}
inline _LIBCPP_HIDE_FROM_ABI long double atan2(long double __y, long double __x) _NOEXCEPT {return ::atan2l(__y, __x);}
#    endif

template <class _A1, class _A2>
inline _LIBCPP_HIDE_FROM_ABI
typename std::__enable_if_t
<
    std::is_arithmetic<_A1>::value &&
    std::is_arithmetic<_A2>::value,
    std::__promote<_A1, _A2>
>::type
atan2(_A1 __y, _A2 __x) _NOEXCEPT
{
    typedef typename std::__promote<_A1, _A2>::type __result_type;
    static_assert((!(std::_IsSame<_A1, __result_type>::value &&
                     std::_IsSame<_A2, __result_type>::value)), "");
    return ::atan2((__result_type)__y, (__result_type)__x);
}

// ceil

#    if !defined(__sun__)
inline _LIBCPP_HIDE_FROM_ABI float       ceil(float __x) _NOEXCEPT       {return ::ceilf(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double ceil(long double __x) _NOEXCEPT {return ::ceill(__x);}
#    endif

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
ceil(_A1 __x) _NOEXCEPT {return ::ceil((double)__x);}

// cos

#    if !defined(__sun__)
inline _LIBCPP_HIDE_FROM_ABI float       cos(float __x) _NOEXCEPT       {return ::cosf(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double cos(long double __x) _NOEXCEPT {return ::cosl(__x);}
#    endif

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
cos(_A1 __x) _NOEXCEPT {return ::cos((double)__x);}

// cosh

#    if !defined(__sun__)
inline _LIBCPP_HIDE_FROM_ABI float       cosh(float __x) _NOEXCEPT       {return ::coshf(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double cosh(long double __x) _NOEXCEPT {return ::coshl(__x);}
#    endif

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
cosh(_A1 __x) _NOEXCEPT {return ::cosh((double)__x);}

// exp

#    if !defined(__sun__)
inline _LIBCPP_HIDE_FROM_ABI float       exp(float __x) _NOEXCEPT       {return ::expf(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double exp(long double __x) _NOEXCEPT {return ::expl(__x);}
#    endif

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
exp(_A1 __x) _NOEXCEPT {return ::exp((double)__x);}

// fabs

#    if !defined(__sun__)
inline _LIBCPP_HIDE_FROM_ABI float       fabs(float __x) _NOEXCEPT       {return ::fabsf(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double fabs(long double __x) _NOEXCEPT {return ::fabsl(__x);}
#    endif

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
fabs(_A1 __x) _NOEXCEPT {return ::fabs((double)__x);}

// floor

#    if !defined(__sun__)
inline _LIBCPP_HIDE_FROM_ABI float       floor(float __x) _NOEXCEPT       {return ::floorf(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double floor(long double __x) _NOEXCEPT {return ::floorl(__x);}
#    endif

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
floor(_A1 __x) _NOEXCEPT {return ::floor((double)__x);}

// fmod

#    if !defined(__sun__)
inline _LIBCPP_HIDE_FROM_ABI float       fmod(float __x, float __y) _NOEXCEPT             {return ::fmodf(__x, __y);}
inline _LIBCPP_HIDE_FROM_ABI long double fmod(long double __x, long double __y) _NOEXCEPT {return ::fmodl(__x, __y);}
#    endif

template <class _A1, class _A2>
inline _LIBCPP_HIDE_FROM_ABI
typename std::__enable_if_t
<
    std::is_arithmetic<_A1>::value &&
    std::is_arithmetic<_A2>::value,
    std::__promote<_A1, _A2>
>::type
fmod(_A1 __x, _A2 __y) _NOEXCEPT
{
    typedef typename std::__promote<_A1, _A2>::type __result_type;
    static_assert((!(std::_IsSame<_A1, __result_type>::value &&
                     std::_IsSame<_A2, __result_type>::value)), "");
    return ::fmod((__result_type)__x, (__result_type)__y);
}

// frexp

#    if !defined(__sun__)
inline _LIBCPP_HIDE_FROM_ABI float       frexp(float __x, int* __e) _NOEXCEPT       {return ::frexpf(__x, __e);}
inline _LIBCPP_HIDE_FROM_ABI long double frexp(long double __x, int* __e) _NOEXCEPT {return ::frexpl(__x, __e);}
#    endif

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
frexp(_A1 __x, int* __e) _NOEXCEPT {return ::frexp((double)__x, __e);}

// ldexp

#    if !defined(__sun__)
inline _LIBCPP_HIDE_FROM_ABI float       ldexp(float __x, int __e) _NOEXCEPT       {return ::ldexpf(__x, __e);}
inline _LIBCPP_HIDE_FROM_ABI long double ldexp(long double __x, int __e) _NOEXCEPT {return ::ldexpl(__x, __e);}
#    endif

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
ldexp(_A1 __x, int __e) _NOEXCEPT {return ::ldexp((double)__x, __e);}

// log

#    if !defined(__sun__)
inline _LIBCPP_HIDE_FROM_ABI float       log(float __x) _NOEXCEPT       {return ::logf(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double log(long double __x) _NOEXCEPT {return ::logl(__x);}
#    endif

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
log(_A1 __x) _NOEXCEPT {return ::log((double)__x);}

// log10

#    if !defined(__sun__)
inline _LIBCPP_HIDE_FROM_ABI float       log10(float __x) _NOEXCEPT       {return ::log10f(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double log10(long double __x) _NOEXCEPT {return ::log10l(__x);}
#    endif

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
log10(_A1 __x) _NOEXCEPT {return ::log10((double)__x);}

// modf

#    if !defined(__sun__)
inline _LIBCPP_HIDE_FROM_ABI float       modf(float __x, float* __y) _NOEXCEPT             {return ::modff(__x, __y);}
inline _LIBCPP_HIDE_FROM_ABI long double modf(long double __x, long double* __y) _NOEXCEPT {return ::modfl(__x, __y);}
#    endif

// pow

#    if !defined(__sun__)
inline _LIBCPP_HIDE_FROM_ABI float       pow(float __x, float __y) _NOEXCEPT             {return ::powf(__x, __y);}
inline _LIBCPP_HIDE_FROM_ABI long double pow(long double __x, long double __y) _NOEXCEPT {return ::powl(__x, __y);}
#    endif

template <class _A1, class _A2>
inline _LIBCPP_HIDE_FROM_ABI
typename std::__enable_if_t
<
    std::is_arithmetic<_A1>::value &&
    std::is_arithmetic<_A2>::value,
    std::__promote<_A1, _A2>
>::type
pow(_A1 __x, _A2 __y) _NOEXCEPT
{
    typedef typename std::__promote<_A1, _A2>::type __result_type;
    static_assert((!(std::_IsSame<_A1, __result_type>::value &&
                     std::_IsSame<_A2, __result_type>::value)), "");
    return ::pow((__result_type)__x, (__result_type)__y);
}

// sin

#    if !defined(__sun__)
inline _LIBCPP_HIDE_FROM_ABI float       sin(float __x) _NOEXCEPT       {return ::sinf(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double sin(long double __x) _NOEXCEPT {return ::sinl(__x);}
#endif

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
sin(_A1 __x) _NOEXCEPT {return ::sin((double)__x);}

// sinh

#    if !defined(__sun__)
inline _LIBCPP_HIDE_FROM_ABI float       sinh(float __x) _NOEXCEPT       {return ::sinhf(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double sinh(long double __x) _NOEXCEPT {return ::sinhl(__x);}
#    endif

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
sinh(_A1 __x) _NOEXCEPT {return ::sinh((double)__x);}

// sqrt

#    if !defined(__sun__)
inline _LIBCPP_HIDE_FROM_ABI float       sqrt(float __x) _NOEXCEPT       {return ::sqrtf(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double sqrt(long double __x) _NOEXCEPT {return ::sqrtl(__x);}
#    endif

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
sqrt(_A1 __x) _NOEXCEPT {return ::sqrt((double)__x);}

// tan

#    if !defined(__sun__)
inline _LIBCPP_HIDE_FROM_ABI float       tan(float __x) _NOEXCEPT       {return ::tanf(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double tan(long double __x) _NOEXCEPT {return ::tanl(__x);}
#    endif

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
tan(_A1 __x) _NOEXCEPT {return ::tan((double)__x);}

// tanh

#    if !defined(__sun__)
inline _LIBCPP_HIDE_FROM_ABI float       tanh(float __x) _NOEXCEPT       {return ::tanhf(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double tanh(long double __x) _NOEXCEPT {return ::tanhl(__x);}
#    endif

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
tanh(_A1 __x) _NOEXCEPT {return ::tanh((double)__x);}

// acosh

inline _LIBCPP_HIDE_FROM_ABI float       acosh(float __x) _NOEXCEPT       {return ::acoshf(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double acosh(long double __x) _NOEXCEPT {return ::acoshl(__x);}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
acosh(_A1 __x) _NOEXCEPT {return ::acosh((double)__x);}

// asinh

inline _LIBCPP_HIDE_FROM_ABI float       asinh(float __x) _NOEXCEPT       {return ::asinhf(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double asinh(long double __x) _NOEXCEPT {return ::asinhl(__x);}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
asinh(_A1 __x) _NOEXCEPT {return ::asinh((double)__x);}

// atanh

inline _LIBCPP_HIDE_FROM_ABI float       atanh(float __x) _NOEXCEPT       {return ::atanhf(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double atanh(long double __x) _NOEXCEPT {return ::atanhl(__x);}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
atanh(_A1 __x) _NOEXCEPT {return ::atanh((double)__x);}

// cbrt

inline _LIBCPP_HIDE_FROM_ABI float       cbrt(float __x) _NOEXCEPT       {return ::cbrtf(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double cbrt(long double __x) _NOEXCEPT {return ::cbrtl(__x);}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
cbrt(_A1 __x) _NOEXCEPT {return ::cbrt((double)__x);}

// copysign

#if __has_builtin(__builtin_copysignf)
_LIBCPP_CONSTEXPR
#endif
inline _LIBCPP_HIDE_FROM_ABI float __libcpp_copysign(float __x, float __y) _NOEXCEPT {
#if __has_builtin(__builtin_copysignf)
  return __builtin_copysignf(__x, __y);
#else
  return ::copysignf(__x, __y);
#endif
}

#if __has_builtin(__builtin_copysign)
_LIBCPP_CONSTEXPR
#endif
inline _LIBCPP_HIDE_FROM_ABI double __libcpp_copysign(double __x, double __y) _NOEXCEPT {
#if __has_builtin(__builtin_copysign)
  return __builtin_copysign(__x, __y);
#else
  return ::copysign(__x, __y);
#endif
}

#if __has_builtin(__builtin_copysignl)
_LIBCPP_CONSTEXPR
#endif
inline _LIBCPP_HIDE_FROM_ABI long double __libcpp_copysign(long double __x, long double __y) _NOEXCEPT {
#if __has_builtin(__builtin_copysignl)
  return __builtin_copysignl(__x, __y);
#else
  return ::copysignl(__x, __y);
#endif
}

template <class _A1, class _A2>
#if __has_builtin(__builtin_copysign)
_LIBCPP_CONSTEXPR
#endif
inline _LIBCPP_HIDE_FROM_ABI
typename std::__enable_if_t
<
    std::is_arithmetic<_A1>::value &&
    std::is_arithmetic<_A2>::value,
    std::__promote<_A1, _A2>
>::type
__libcpp_copysign(_A1 __x, _A2 __y) _NOEXCEPT {
    typedef typename std::__promote<_A1, _A2>::type __result_type;
    static_assert((!(std::_IsSame<_A1, __result_type>::value &&
                     std::_IsSame<_A2, __result_type>::value)), "");
#if __has_builtin(__builtin_copysign)
    return __builtin_copysign((__result_type)__x, (__result_type)__y);
#else
    return ::copysign((__result_type)__x, (__result_type)__y);
#endif
}

inline _LIBCPP_HIDE_FROM_ABI float copysign(float __x, float __y) _NOEXCEPT {
  return ::__libcpp_copysign(__x, __y);
}

inline _LIBCPP_HIDE_FROM_ABI long double copysign(long double __x, long double __y) _NOEXCEPT {
  return ::__libcpp_copysign(__x, __y);
}

template <class _A1, class _A2>
inline _LIBCPP_HIDE_FROM_ABI
typename std::__enable_if_t
<
    std::is_arithmetic<_A1>::value &&
    std::is_arithmetic<_A2>::value,
    std::__promote<_A1, _A2>
>::type
    copysign(_A1 __x, _A2 __y) _NOEXCEPT {
  return ::__libcpp_copysign(__x, __y);
}

// erf

inline _LIBCPP_HIDE_FROM_ABI float       erf(float __x) _NOEXCEPT       {return ::erff(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double erf(long double __x) _NOEXCEPT {return ::erfl(__x);}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
erf(_A1 __x) _NOEXCEPT {return ::erf((double)__x);}

// erfc

inline _LIBCPP_HIDE_FROM_ABI float       erfc(float __x) _NOEXCEPT       {return ::erfcf(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double erfc(long double __x) _NOEXCEPT {return ::erfcl(__x);}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
erfc(_A1 __x) _NOEXCEPT {return ::erfc((double)__x);}

// exp2

inline _LIBCPP_HIDE_FROM_ABI float       exp2(float __x) _NOEXCEPT       {return ::exp2f(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double exp2(long double __x) _NOEXCEPT {return ::exp2l(__x);}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
exp2(_A1 __x) _NOEXCEPT {return ::exp2((double)__x);}

// expm1

inline _LIBCPP_HIDE_FROM_ABI float       expm1(float __x) _NOEXCEPT       {return ::expm1f(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double expm1(long double __x) _NOEXCEPT {return ::expm1l(__x);}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
expm1(_A1 __x) _NOEXCEPT {return ::expm1((double)__x);}

// fdim

inline _LIBCPP_HIDE_FROM_ABI float       fdim(float __x, float __y) _NOEXCEPT             {return ::fdimf(__x, __y);}
inline _LIBCPP_HIDE_FROM_ABI long double fdim(long double __x, long double __y) _NOEXCEPT {return ::fdiml(__x, __y);}

template <class _A1, class _A2>
inline _LIBCPP_HIDE_FROM_ABI
typename std::__enable_if_t
<
    std::is_arithmetic<_A1>::value &&
    std::is_arithmetic<_A2>::value,
    std::__promote<_A1, _A2>
>::type
fdim(_A1 __x, _A2 __y) _NOEXCEPT
{
    typedef typename std::__promote<_A1, _A2>::type __result_type;
    static_assert((!(std::_IsSame<_A1, __result_type>::value &&
                     std::_IsSame<_A2, __result_type>::value)), "");
    return ::fdim((__result_type)__x, (__result_type)__y);
}

// fma

inline _LIBCPP_HIDE_FROM_ABI float       fma(float __x, float __y, float __z) _NOEXCEPT
{
#if __has_builtin(__builtin_fmaf)
    return __builtin_fmaf(__x, __y, __z);
#else
    return ::fmaf(__x, __y, __z);
#endif
}
inline _LIBCPP_HIDE_FROM_ABI long double fma(long double __x, long double __y, long double __z) _NOEXCEPT
{
#if __has_builtin(__builtin_fmal)
    return __builtin_fmal(__x, __y, __z);
#else
    return ::fmal(__x, __y, __z);
#endif
}

template <class _A1, class _A2, class _A3>
inline _LIBCPP_HIDE_FROM_ABI
typename std::__enable_if_t
<
    std::is_arithmetic<_A1>::value &&
    std::is_arithmetic<_A2>::value &&
    std::is_arithmetic<_A3>::value,
    std::__promote<_A1, _A2, _A3>
>::type
fma(_A1 __x, _A2 __y, _A3 __z) _NOEXCEPT
{
    typedef typename std::__promote<_A1, _A2, _A3>::type __result_type;
    static_assert((!(std::_IsSame<_A1, __result_type>::value &&
                     std::_IsSame<_A2, __result_type>::value &&
                     std::_IsSame<_A3, __result_type>::value)), "");
#if __has_builtin(__builtin_fma)
    return __builtin_fma((__result_type)__x, (__result_type)__y, (__result_type)__z);
#else
    return ::fma((__result_type)__x, (__result_type)__y, (__result_type)__z);
#endif
}

// fmax

inline _LIBCPP_HIDE_FROM_ABI float       fmax(float __x, float __y) _NOEXCEPT             {return ::fmaxf(__x, __y);}
inline _LIBCPP_HIDE_FROM_ABI long double fmax(long double __x, long double __y) _NOEXCEPT {return ::fmaxl(__x, __y);}

template <class _A1, class _A2>
inline _LIBCPP_HIDE_FROM_ABI
typename std::__enable_if_t
<
    std::is_arithmetic<_A1>::value &&
    std::is_arithmetic<_A2>::value,
    std::__promote<_A1, _A2>
>::type
fmax(_A1 __x, _A2 __y) _NOEXCEPT
{
    typedef typename std::__promote<_A1, _A2>::type __result_type;
    static_assert((!(std::_IsSame<_A1, __result_type>::value &&
                     std::_IsSame<_A2, __result_type>::value)), "");
    return ::fmax((__result_type)__x, (__result_type)__y);
}

// fmin

inline _LIBCPP_HIDE_FROM_ABI float       fmin(float __x, float __y) _NOEXCEPT             {return ::fminf(__x, __y);}
inline _LIBCPP_HIDE_FROM_ABI long double fmin(long double __x, long double __y) _NOEXCEPT {return ::fminl(__x, __y);}

template <class _A1, class _A2>
inline _LIBCPP_HIDE_FROM_ABI
typename std::__enable_if_t
<
    std::is_arithmetic<_A1>::value &&
    std::is_arithmetic<_A2>::value,
    std::__promote<_A1, _A2>
>::type
fmin(_A1 __x, _A2 __y) _NOEXCEPT
{
    typedef typename std::__promote<_A1, _A2>::type __result_type;
    static_assert((!(std::_IsSame<_A1, __result_type>::value &&
                     std::_IsSame<_A2, __result_type>::value)), "");
    return ::fmin((__result_type)__x, (__result_type)__y);
}

// hypot

inline _LIBCPP_HIDE_FROM_ABI float       hypot(float __x, float __y) _NOEXCEPT             {return ::hypotf(__x, __y);}
inline _LIBCPP_HIDE_FROM_ABI long double hypot(long double __x, long double __y) _NOEXCEPT {return ::hypotl(__x, __y);}

template <class _A1, class _A2>
inline _LIBCPP_HIDE_FROM_ABI
typename std::__enable_if_t
<
    std::is_arithmetic<_A1>::value &&
    std::is_arithmetic<_A2>::value,
    std::__promote<_A1, _A2>
>::type
hypot(_A1 __x, _A2 __y) _NOEXCEPT
{
    typedef typename std::__promote<_A1, _A2>::type __result_type;
    static_assert((!(std::_IsSame<_A1, __result_type>::value &&
                     std::_IsSame<_A2, __result_type>::value)), "");
    return ::hypot((__result_type)__x, (__result_type)__y);
}

// ilogb

inline _LIBCPP_HIDE_FROM_ABI int ilogb(float __x) _NOEXCEPT       {return ::ilogbf(__x);}
inline _LIBCPP_HIDE_FROM_ABI int ilogb(long double __x) _NOEXCEPT {return ::ilogbl(__x);}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, int>::type
ilogb(_A1 __x) _NOEXCEPT {return ::ilogb((double)__x);}

// lgamma

inline _LIBCPP_HIDE_FROM_ABI float       lgamma(float __x) _NOEXCEPT       {return ::lgammaf(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double lgamma(long double __x) _NOEXCEPT {return ::lgammal(__x);}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
lgamma(_A1 __x) _NOEXCEPT {return ::lgamma((double)__x);}

// llrint

inline _LIBCPP_HIDE_FROM_ABI long long llrint(float __x) _NOEXCEPT
{
#if __has_builtin(__builtin_llrintf)
    return __builtin_llrintf(__x);
#else
    return ::llrintf(__x);
#endif
}
inline _LIBCPP_HIDE_FROM_ABI long long llrint(long double __x) _NOEXCEPT
{
#if __has_builtin(__builtin_llrintl)
    return __builtin_llrintl(__x);
#else
    return ::llrintl(__x);
#endif
}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, long long>::type
llrint(_A1 __x) _NOEXCEPT
{
#if __has_builtin(__builtin_llrint)
    return __builtin_llrint((double)__x);
#else
    return ::llrint((double)__x);
#endif
}

// llround

inline _LIBCPP_HIDE_FROM_ABI long long llround(float __x) _NOEXCEPT
{
#if __has_builtin(__builtin_llroundf)
    return __builtin_llroundf(__x);
#else
    return ::llroundf(__x);
#endif
}
inline _LIBCPP_HIDE_FROM_ABI long long llround(long double __x) _NOEXCEPT
{
#if __has_builtin(__builtin_llroundl)
    return __builtin_llroundl(__x);
#else
    return ::llroundl(__x);
#endif
}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, long long>::type
llround(_A1 __x) _NOEXCEPT
{
#if __has_builtin(__builtin_llround)
    return __builtin_llround((double)__x);
#else
    return ::llround((double)__x);
#endif
}

// log1p

inline _LIBCPP_HIDE_FROM_ABI float       log1p(float __x) _NOEXCEPT       {return ::log1pf(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double log1p(long double __x) _NOEXCEPT {return ::log1pl(__x);}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
log1p(_A1 __x) _NOEXCEPT {return ::log1p((double)__x);}

// log2

inline _LIBCPP_HIDE_FROM_ABI float       log2(float __x) _NOEXCEPT       {return ::log2f(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double log2(long double __x) _NOEXCEPT {return ::log2l(__x);}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
log2(_A1 __x) _NOEXCEPT {return ::log2((double)__x);}

// logb

inline _LIBCPP_HIDE_FROM_ABI float       logb(float __x) _NOEXCEPT       {return ::logbf(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double logb(long double __x) _NOEXCEPT {return ::logbl(__x);}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
logb(_A1 __x) _NOEXCEPT {return ::logb((double)__x);}

// lrint

inline _LIBCPP_HIDE_FROM_ABI long lrint(float __x) _NOEXCEPT
{
#if __has_builtin(__builtin_lrintf)
    return __builtin_lrintf(__x);
#else
    return ::lrintf(__x);
#endif
}
inline _LIBCPP_HIDE_FROM_ABI long lrint(long double __x) _NOEXCEPT
{
#if __has_builtin(__builtin_lrintl)
    return __builtin_lrintl(__x);
#else
    return ::lrintl(__x);
#endif
}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, long>::type
lrint(_A1 __x) _NOEXCEPT
{
#if __has_builtin(__builtin_lrint)
    return __builtin_lrint((double)__x);
#else
    return ::lrint((double)__x);
#endif
}

// lround

inline _LIBCPP_HIDE_FROM_ABI long lround(float __x) _NOEXCEPT
{
#if __has_builtin(__builtin_lroundf)
    return __builtin_lroundf(__x);
#else
    return ::lroundf(__x);
#endif
}
inline _LIBCPP_HIDE_FROM_ABI long lround(long double __x) _NOEXCEPT
{
#if __has_builtin(__builtin_lroundl)
    return __builtin_lroundl(__x);
#else
    return ::lroundl(__x);
#endif
}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, long>::type
lround(_A1 __x) _NOEXCEPT
{
#if __has_builtin(__builtin_lround)
    return __builtin_lround((double)__x);
#else
    return ::lround((double)__x);
#endif
}

// nan

// nearbyint

inline _LIBCPP_HIDE_FROM_ABI float       nearbyint(float __x) _NOEXCEPT       {return ::nearbyintf(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double nearbyint(long double __x) _NOEXCEPT {return ::nearbyintl(__x);}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
nearbyint(_A1 __x) _NOEXCEPT {return ::nearbyint((double)__x);}

// nextafter

inline _LIBCPP_HIDE_FROM_ABI float       nextafter(float __x, float __y) _NOEXCEPT             {return ::nextafterf(__x, __y);}
inline _LIBCPP_HIDE_FROM_ABI long double nextafter(long double __x, long double __y) _NOEXCEPT {return ::nextafterl(__x, __y);}

template <class _A1, class _A2>
inline _LIBCPP_HIDE_FROM_ABI
typename std::__enable_if_t
<
    std::is_arithmetic<_A1>::value &&
    std::is_arithmetic<_A2>::value,
    std::__promote<_A1, _A2>
>::type
nextafter(_A1 __x, _A2 __y) _NOEXCEPT
{
    typedef typename std::__promote<_A1, _A2>::type __result_type;
    static_assert((!(std::_IsSame<_A1, __result_type>::value &&
                     std::_IsSame<_A2, __result_type>::value)), "");
    return ::nextafter((__result_type)__x, (__result_type)__y);
}

// nexttoward

inline _LIBCPP_HIDE_FROM_ABI float       nexttoward(float __x, long double __y) _NOEXCEPT       {return ::nexttowardf(__x, __y);}
inline _LIBCPP_HIDE_FROM_ABI long double nexttoward(long double __x, long double __y) _NOEXCEPT {return ::nexttowardl(__x, __y);}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
nexttoward(_A1 __x, long double __y) _NOEXCEPT {return ::nexttoward((double)__x, __y);}

// remainder

inline _LIBCPP_HIDE_FROM_ABI float       remainder(float __x, float __y) _NOEXCEPT             {return ::remainderf(__x, __y);}
inline _LIBCPP_HIDE_FROM_ABI long double remainder(long double __x, long double __y) _NOEXCEPT {return ::remainderl(__x, __y);}

template <class _A1, class _A2>
inline _LIBCPP_HIDE_FROM_ABI
typename std::__enable_if_t
<
    std::is_arithmetic<_A1>::value &&
    std::is_arithmetic<_A2>::value,
    std::__promote<_A1, _A2>
>::type
remainder(_A1 __x, _A2 __y) _NOEXCEPT
{
    typedef typename std::__promote<_A1, _A2>::type __result_type;
    static_assert((!(std::_IsSame<_A1, __result_type>::value &&
                     std::_IsSame<_A2, __result_type>::value)), "");
    return ::remainder((__result_type)__x, (__result_type)__y);
}

// remquo

inline _LIBCPP_HIDE_FROM_ABI float       remquo(float __x, float __y, int* __z) _NOEXCEPT             {return ::remquof(__x, __y, __z);}
inline _LIBCPP_HIDE_FROM_ABI long double remquo(long double __x, long double __y, int* __z) _NOEXCEPT {return ::remquol(__x, __y, __z);}

template <class _A1, class _A2>
inline _LIBCPP_HIDE_FROM_ABI
typename std::__enable_if_t
<
    std::is_arithmetic<_A1>::value &&
    std::is_arithmetic<_A2>::value,
    std::__promote<_A1, _A2>
>::type
remquo(_A1 __x, _A2 __y, int* __z) _NOEXCEPT
{
    typedef typename std::__promote<_A1, _A2>::type __result_type;
    static_assert((!(std::_IsSame<_A1, __result_type>::value &&
                     std::_IsSame<_A2, __result_type>::value)), "");
    return ::remquo((__result_type)__x, (__result_type)__y, __z);
}

// rint

inline _LIBCPP_HIDE_FROM_ABI float       rint(float __x) _NOEXCEPT
{
#if __has_builtin(__builtin_rintf)
    return __builtin_rintf(__x);
#else
    return ::rintf(__x);
#endif
}
inline _LIBCPP_HIDE_FROM_ABI long double rint(long double __x) _NOEXCEPT
{
#if __has_builtin(__builtin_rintl)
    return __builtin_rintl(__x);
#else
    return ::rintl(__x);
#endif
}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
rint(_A1 __x) _NOEXCEPT
{
#if __has_builtin(__builtin_rint)
    return __builtin_rint((double)__x);
#else
    return ::rint((double)__x);
#endif
}

// round

inline _LIBCPP_HIDE_FROM_ABI float       round(float __x) _NOEXCEPT
{
#if __has_builtin(__builtin_round)
    return __builtin_round(__x);
#else
    return ::round(__x);
#endif
}
inline _LIBCPP_HIDE_FROM_ABI long double round(long double __x) _NOEXCEPT
{
#if __has_builtin(__builtin_roundl)
    return __builtin_roundl(__x);
#else
    return ::roundl(__x);
#endif
}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
round(_A1 __x) _NOEXCEPT
{
#if __has_builtin(__builtin_round)
    return __builtin_round((double)__x);
#else
    return ::round((double)__x);
#endif
}

// scalbln

inline _LIBCPP_HIDE_FROM_ABI float       scalbln(float __x, long __y) _NOEXCEPT       {return ::scalblnf(__x, __y);}
inline _LIBCPP_HIDE_FROM_ABI long double scalbln(long double __x, long __y) _NOEXCEPT {return ::scalblnl(__x, __y);}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
scalbln(_A1 __x, long __y) _NOEXCEPT {return ::scalbln((double)__x, __y);}

// scalbn

inline _LIBCPP_HIDE_FROM_ABI float       scalbn(float __x, int __y) _NOEXCEPT       {return ::scalbnf(__x, __y);}
inline _LIBCPP_HIDE_FROM_ABI long double scalbn(long double __x, int __y) _NOEXCEPT {return ::scalbnl(__x, __y);}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
scalbn(_A1 __x, int __y) _NOEXCEPT {return ::scalbn((double)__x, __y);}

// tgamma

inline _LIBCPP_HIDE_FROM_ABI float       tgamma(float __x) _NOEXCEPT       {return ::tgammaf(__x);}
inline _LIBCPP_HIDE_FROM_ABI long double tgamma(long double __x) _NOEXCEPT {return ::tgammal(__x);}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
tgamma(_A1 __x) _NOEXCEPT {return ::tgamma((double)__x);}

// trunc

inline _LIBCPP_HIDE_FROM_ABI float       trunc(float __x) _NOEXCEPT
{
#if __has_builtin(__builtin_trunc)
    return __builtin_trunc(__x);
#else
    return ::trunc(__x);
#endif
}
inline _LIBCPP_HIDE_FROM_ABI long double trunc(long double __x) _NOEXCEPT
{
#if __has_builtin(__builtin_truncl)
    return __builtin_truncl(__x);
#else
    return ::truncl(__x);
#endif
}

template <class _A1>
inline _LIBCPP_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_A1>::value, double>::type
trunc(_A1 __x) _NOEXCEPT
{
#if __has_builtin(__builtin_trunc)
    return __builtin_trunc((double)__x);
#else
    return ::trunc((double)__x);
#endif
}

} // extern "C++"

#endif // __cplusplus

#else // _LIBCPP_MATH_H

// This include lives outside the header guard in order to support an MSVC
// extension which allows users to do:
//
// #define _USE_MATH_DEFINES
// #include <math.h>
//
// and receive the definitions of mathematical constants, even if <math.h>
// has previously been included.
#if defined(_LIBCPP_MSVCRT) && defined(_USE_MATH_DEFINES)
#include_next <math.h>
#endif

#endif // _LIBCPP_MATH_H
