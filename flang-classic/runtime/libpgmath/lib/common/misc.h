//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

//    Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

//

#ifndef __MISC_H__
#define __MISC_H__

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

#ifndef M_PIl
#define M_PIl 3.141592653589793238462643383279502884L
#endif

#ifndef M_1_PI
#define M_1_PI 0.318309886183790671537767526745028724
#endif

#ifndef M_1_PIl
#define M_1_PIl 0.318309886183790671537767526745028724L
#endif

#ifndef M_2_PI
#define M_2_PI 0.636619772367581343075535053490057448
#endif

#ifndef M_2_PIl
#define M_2_PIl 0.636619772367581343075535053490057448L
#endif

//

/*
  PI_A to PI_D are constants that satisfy the following two conditions.

  * For PI_A, PI_B and PI_C, the last 28 bits are zero.
  * PI_A + PI_B + PI_C + PI_D is close to PI as much as possible.

  The argument of a trig function is multiplied by 1/PI, and the
  integral part is divided into two parts, each has at most 28
  bits. So, the maximum argument that could be correctly reduced
  should be 2^(28*2-1) PI = 1.1e+17. However, due to internal
  double precision calculation, the actual maximum argument that can
  be correctly reduced is around 2^50 = 1.1e+15.
 */

#define PI_A 3.1415926218032836914
#define PI_B 3.1786509424591713469e-08
#define PI_C 1.2246467864107188502e-16
#define PI_D 1.2736634327021899816e-24
#define TRIGRANGEMAX 1e+15

/*
  PI_A2 and PI_B2 are constants that satisfy the following two conditions.

  * The last 3 bits of PI_A2 are zero.
  * PI_A2 + PI_B2 is close to PI as much as possible.

  The argument of a trig function is multiplied by 1/PI, and the
  integral part is multiplied by PI_A2. So, the maximum argument that
  could be correctly reduced should be 2^(3-1) PI = 12.6. By testing,
  we confirmed that it correctly reduces the argument up to around 15.
 */

#define PI_A2 3.141592653589793116
#define PI_B2 1.2246467991473532072e-16
#define TRIGRANGEMAX2 15

#define M_2_PI_H 0.63661977236758138243
#define M_2_PI_L -3.9357353350364971764e-17

#define SQRT_DBL_MAX 1.3407807929942596355e+154

#define TRIGRANGEMAX3 1e+9

#define M_4_PI 1.273239544735162542821171882678754627704620361328125

#define L2U .69314718055966295651160180568695068359375
#define L2L .28235290563031577122588448175013436025525412068e-12
#define R_LN2 1.442695040888963407359924681001892137426645954152985934135449406931

//

#define PI_Af 3.140625f
#define PI_Bf 0.0009670257568359375f
#define PI_Cf 6.2771141529083251953e-07f
#define PI_Df 1.2154201256553420762e-10f
#define PI_XDf 1.2141754268668591976e-10f
#define PI_XEf 1.2446743939339977025e-13f
#define TRIGRANGEMAXf 1e+7 // 39000

#define PI_A2f 3.1414794921875f
#define PI_B2f 0.00011315941810607910156f
#define PI_C2f 1.9841872589410058936e-09f
#define TRIGRANGEMAX2f 125.0f

#define PI_A3f 3.14154052734375f
#define PI_B3f 5.212612450122833252e-05f
#define PI_C3f 1.2154188766544393729e-10f
#define PI_D3f 1.2246402351402674302e-16f
#define PI_E3f 6.5640073364868052239e-22f
#define TRIGRANGEMAX3f 5e+9f

#define TRIGRANGEMAX4f 8e+6f

#define SQRT_FLT_MAX 18446743523953729536.0

#define L2Uf 0.693145751953125f
#define L2Lf 1.428606765330187045e-06f

#define R_LN2f 1.442695040888963407359924681001892137426645954152985934135449406931f
#define M_PIf ((float)M_PI)

//

#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#ifndef ABS
#define ABS(x) ((x) < 0 ? -(x) : (x))
#endif

typedef long double longdouble;

#ifndef Sleef_double2_DEFINED
#define Sleef_double2_DEFINED
typedef struct {
  double x, y;
} Sleef_double2;
#endif

#ifndef Sleef_float2_DEFINED
#define Sleef_float2_DEFINED
typedef struct {
  float x, y;
} Sleef_float2;
#endif

#ifndef Sleef_longdouble2_DEFINED
#define Sleef_longdouble2_DEFINED
typedef struct {
  long double x, y;
} Sleef_longdouble2;
#endif

#if defined(ENABLEFLOAT128) && !defined(Sleef_quad2_DEFINED)
#define Sleef_quad2_DEFINED
typedef __float128 Sleef_quad;
typedef struct {
  __float128 x, y;
} Sleef_quad2;
#endif

//

#if defined (__GNUC__) || defined (__clang__) || defined(__INTEL_COMPILER)

#define INLINE __attribute__((always_inline)) inline

#ifndef __INTEL_COMPILER
#define CONST const
#else
#define CONST
#endif

#if defined(__MINGW32__) || defined(__MINGW64__) || defined(__CYGWIN__)
#define EXPORT __stdcall __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

#ifdef INFINITY
#undef INFINITY
#endif

#ifdef NAN
#undef NAN
#endif

#define NAN __builtin_nan("")
#define NANf __builtin_nanf("")
#define NANl __builtin_nanl("")
#define INFINITY __builtin_inf()
#define INFINITYf __builtin_inff()
#define INFINITYl __builtin_infl()

#if defined(__INTEL_COMPILER)
#define INFINITYq __builtin_inf()
#define NANq __builtin_nan("")
#else
#define INFINITYq __builtin_infq()
#define NANq (INFINITYq - INFINITYq)
#endif

#elif defined(_MSC_VER)

#define INLINE __forceinline
#define CONST
#define EXPORT __declspec(dllexport)

#if (defined(__GNUC__) || defined(__CLANG__)) && (defined(__i386__) || defined(__x86_64__))
#include <x86intrin.h>
#endif

#define INFINITYf ((float)INFINITY)
#define NANf ((float)NAN)
#define INFINITYl ((long double)INFINITY)
#define NANl ((long double)NAN)

#if (defined(_M_AMD64) || defined(_M_X64))
#ifndef __SSE2__
#define __SSE2__
#define __SSE3__
#define __SSE4_1__
#endif
#elif _M_IX86_FP == 2
#ifndef __SSE2__
#define __SSE2__
#define __SSE3__
#define __SSE4_1__
#endif
#elif _M_IX86_FP == 1
#ifndef __SSE__
#define __SSE__
#endif
#endif

#if !(defined TARGET_OSX_X8664)
static INLINE CONST int isinff(float x) { return x == INFINITYf || x == -INFINITYf; }
static INLINE CONST int isinfl(long double x) { return x == INFINITYl || x == -INFINITYl; }
static INLINE CONST int isnanf(float x) { return x != x; }
static INLINE CONST int isnanl(long double x) { return x != x; }
#endif

#endif // defined(_MSC_VER)

#ifdef __APPLE__
#if !(defined TARGET_OSX_X8664)
static INLINE CONST int isinff(float x) { return x == INFINITYf || x == -INFINITYf; }
static INLINE CONST int isinfl(long double x) { return x == INFINITYl || x == -INFINITYl; }
static INLINE CONST int isnanf(float x) { return x != x; }
static INLINE CONST int isnanl(long double x) { return x != x; }
#endif
#endif

#endif // #ifndef __MISC_H__
