/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#ifndef OCML_H
#define OCML_H

// This C header declares the functions provided by the OCML library
// Aspects of this library's behavior can be controlled via the
// oclc library.  See the oclc header for further information

// Define here the return values from fpclassify
// These match most host definitions
#define FP_NAN 0
#define FP_INFINITE 1
#define FP_ZERO 2
#define FP_SUBNORMAL 3
#define FP_NORMAL 4

#define OCML_DEPRECATED(X, Replacement) __attribute__((deprecated("use "#Replacement " instead", Replacement)))

#define _MANGLE3(P,N,S) P##_##N##_##S
#define MANGLE3(P,N,S) _MANGLE3(P,N,S)
#define OCML_MANGLE_F32(N) MANGLE3(__ocml, N, f32)
#define OCML_MANGLE_2F32(N) MANGLE3(__ocml, N, 2f32)
#define OCML_MANGLE_F64(N) MANGLE3(__ocml, N, f64)
#define OCML_MANGLE_F16(N) MANGLE3(__ocml, N, f16)
#define OCML_MANGLE_2F16(N) MANGLE3(__ocml, N, 2f16)
#define OCML_MANGLE_S32(N) MANGLE3(__ocml, N, s32)
#define OCML_MANGLE_U32(N) MANGLE3(__ocml, N, u32)
#define OCML_MANGLE_S64(N) MANGLE3(__ocml, N, s64)
#define OCML_MANGLE_U64(N) MANGLE3(__ocml, N, u64)


#define DECL_OCML_UNARY_F32(N) extern float OCML_MANGLE_F32(N)(float);
#define _DECL_X_OCML_UNARY_F32(A,N) extern __attribute__((A)) float OCML_MANGLE_F32(N)(float);
#define DECL_PURE_OCML_UNARY_F32(N) _DECL_X_OCML_UNARY_F32(pure, N)
#define DECL_CONST_OCML_UNARY_F32(N) _DECL_X_OCML_UNARY_F32(const, N)

#define DECL_CONST_OCML_UNARYPRED_F32(N) extern __attribute__((const)) int OCML_MANGLE_F32(N)(float);

#define DECL_OCML_BINARY_F32(N) extern float OCML_MANGLE_F32(N)(float, float);
#define _DECL_X_OCML_BINARY_F32(A,N) extern __attribute__((A)) float OCML_MANGLE_F32(N)(float, float);
#define DECL_PURE_OCML_BINARY_F32(N) _DECL_X_OCML_BINARY_F32(pure, N)
#define DECL_CONST_OCML_BINARY_F32(N) _DECL_X_OCML_BINARY_F32(const, N)

#define DECL_CONST_OCML_BINARYPRED_F32(N) extern __attribute__((const)) int OCML_MANGLE_F32(N)(float, float);

#define _DECL_X_OCML_TERNARY_F32(A,N) extern __attribute__((A)) float OCML_MANGLE_F32(N)(float, float, float);
#define DECL_PURE_OCML_TERNARY_F32(N) _DECL_X_OCML_TERNARY_F32(pure, N)
#define DECL_CONST_OCML_TERNARY_F32(N) _DECL_X_OCML_TERNARY_F32(const, N)

#define _DECL_X_OCML_TERNARY_2F32(A,N) extern __attribute__((A)) float2 OCML_MANGLE_2F32(N)(float2, float2, float2);
#define DECL_PURE_OCML_TERNARY_2F32(N) _DECL_X_OCML_TERNARY_2F32(pure, N)
#define DECL_CONST_OCML_TERNARY_2F32(N) _DECL_X_OCML_TERNARY_2F32(const, N)

#define DECL_OCML_UNARY_F64(N) extern double OCML_MANGLE_F64(N)(double);
#define _DECL_X_OCML_UNARY_F64(A,N) extern __attribute__((A)) double OCML_MANGLE_F64(N)(double);
#define DECL_PURE_OCML_UNARY_F64(N) _DECL_X_OCML_UNARY_F64(pure, N)
#define DECL_CONST_OCML_UNARY_F64(N) _DECL_X_OCML_UNARY_F64(const, N)

#define DECL_CONST_OCML_UNARYPRED_F64(N) extern __attribute__((const)) int OCML_MANGLE_F64(N)(double);

#define DECL_OCML_BINARY_F64(N) extern double OCML_MANGLE_F64(N)(double, double);
#define _DECL_X_OCML_BINARY_F64(A,N) extern __attribute__((A)) double OCML_MANGLE_F64(N)(double, double);
#define DECL_PURE_OCML_BINARY_F64(N) _DECL_X_OCML_BINARY_F64(pure, N)
#define DECL_CONST_OCML_BINARY_F64(N) _DECL_X_OCML_BINARY_F64(const, N)

#define DECL_CONST_OCML_BINARYPRED_F64(N) extern __attribute__((const)) int OCML_MANGLE_F64(N)(double, double);

#define _DECL_X_OCML_TERNARY_F64(A,N) extern __attribute__((A)) double OCML_MANGLE_F64(N)(double, double, double);
#define DECL_PURE_OCML_TERNARY_F64(N) _DECL_X_OCML_TERNARY_F64(pure, N)
#define DECL_CONST_OCML_TERNARY_F64(N) _DECL_X_OCML_TERNARY_F64(const, N)

#define DECL_OCML_UNARY_F16(N) extern half OCML_MANGLE_F16(N)(half);
#define _DECL_X_OCML_UNARY_F16(A,N) extern __attribute__((A)) half OCML_MANGLE_F16(N)(half);
#define DECL_PURE_OCML_UNARY_F16(N) _DECL_X_OCML_UNARY_F16(pure, N)
#define DECL_CONST_OCML_UNARY_F16(N) _DECL_X_OCML_UNARY_F16(const, N)

#define DECL_CONST_OCML_UNARYPRED_F16(N) extern __attribute__((const)) int OCML_MANGLE_F16(N)(half);

#define DECL_OCML_BINARY_F16(N) extern half OCML_MANGLE_F16(N)(half, half);
#define _DECL_X_OCML_BINARY_F16(A,N) extern __attribute__((A)) half OCML_MANGLE_F16(N)(half, half);
#define DECL_PURE_OCML_BINARY_F16(N) _DECL_X_OCML_BINARY_F16(pure, N)
#define DECL_CONST_OCML_BINARY_F16(N) _DECL_X_OCML_BINARY_F16(const, N)

#define DECL_CONST_OCML_BINARYPRED_F16(N) extern __attribute__((const)) int OCML_MANGLE_F16(N)(half, half);

#define _DECL_X_OCML_TERNARY_F16(A,N) extern __attribute__((A)) half OCML_MANGLE_F16(N)(half, half, half);
#define DECL_PURE_OCML_TERNARY_F16(N) _DECL_X_OCML_TERNARY_F16(pure, N)
#define DECL_CONST_OCML_TERNARY_F16(N) _DECL_X_OCML_TERNARY_F16(const, N)

#define DECL_OCML_UNARY_2F16(N) extern half2 OCML_MANGLE_2F16(N)(half2);
#define _DECL_X_OCML_UNARY_2F16(A,N) extern __attribute__((A)) half2 OCML_MANGLE_2F16(N)(half2);
#define DECL_PURE_OCML_UNARY_2F16(N) _DECL_X_OCML_UNARY_2F16(pure, N)
#define DECL_CONST_OCML_UNARY_2F16(N) _DECL_X_OCML_UNARY_2F16(const, N)

#define DECL_CONST_OCML_UNARYPRED_2F16(N) extern __attribute__((const)) short2 OCML_MANGLE_2F16(N)(half2);

#define DECL_OCML_BINARY_2F16(N) extern half2 OCML_MANGLE_2F16(N)(half2, half2);
#define _DECL_X_OCML_BINARY_2F16(A,N) extern __attribute__((A)) half2 OCML_MANGLE_2F16(N)(half2, half2);
#define DECL_PURE_OCML_BINARY_2F16(N) _DECL_X_OCML_BINARY_2F16(pure, N)
#define DECL_CONST_OCML_BINARY_2F16(N) _DECL_X_OCML_BINARY_2F16(const, N)

#define DECL_CONST_OCML_BINARYPRED_2F16(N) extern __attribute__((const)) short2 OCML_MANGLE_2F16(N)(half2, half2);

#define _DECL_X_OCML_TERNARY_2F16(A,N) extern __attribute__((A)) half2 OCML_MANGLE_2F16(N)(half2, half2, half2);
#define DECL_PURE_OCML_TERNARY_2F16(N) _DECL_X_OCML_TERNARY_2F16(pure, N)
#define DECL_CONST_OCML_TERNARY_2F16(N) _DECL_X_OCML_TERNARY_2F16(const, N)

DECL_CONST_OCML_UNARY_F32(acos)
DECL_CONST_OCML_UNARY_F32(acospi)
DECL_CONST_OCML_UNARY_F32(acosh)
DECL_CONST_OCML_UNARY_F32(asin)
DECL_CONST_OCML_UNARY_F32(asinpi)
DECL_CONST_OCML_UNARY_F32(asinh)
DECL_CONST_OCML_BINARY_F32(atan2)
DECL_CONST_OCML_BINARY_F32(atan2pi)
DECL_CONST_OCML_UNARY_F32(atan)
DECL_CONST_OCML_UNARY_F32(atanh)
DECL_CONST_OCML_UNARY_F32(atanpi)
DECL_CONST_OCML_UNARY_F32(cbrt)
DECL_CONST_OCML_UNARY_F32(ceil)
DECL_OCML_UNARY_F32(cos)
DECL_CONST_OCML_UNARY_F32(cosh)
DECL_OCML_UNARY_F32(cospi)
DECL_CONST_OCML_BINARY_F32(copysign)
DECL_CONST_OCML_UNARY_F32(erf)
DECL_CONST_OCML_UNARY_F32(erfc)
DECL_CONST_OCML_UNARY_F32(erfinv)
DECL_CONST_OCML_UNARY_F32(erfcinv)
DECL_CONST_OCML_UNARY_F32(erfcx)
DECL_CONST_OCML_UNARY_F32(exp)
DECL_CONST_OCML_UNARY_F32(exp2)
DECL_CONST_OCML_UNARY_F32(exp10)
DECL_CONST_OCML_UNARY_F32(expm1)
DECL_CONST_OCML_UNARY_F32(fabs)
DECL_CONST_OCML_BINARY_F32(fdim)
DECL_CONST_OCML_UNARY_F32(floor)
DECL_CONST_OCML_TERNARY_F32(fma)
DECL_CONST_OCML_TERNARY_2F32(fma)
DECL_CONST_OCML_TERNARY_F32(fmuladd)
DECL_CONST_OCML_TERNARY_2F32(fmuladd)
DECL_CONST_OCML_BINARY_F32(fmax)
DECL_CONST_OCML_BINARY_F32(fmin)
DECL_CONST_OCML_BINARY_F32(fmod)
DECL_CONST_OCML_UNARYPRED_F32(fpclassify)
extern float OCML_MANGLE_F32(fract)(float, __private float *);
extern float OCML_MANGLE_F32(frexp)(float, __private int *);
DECL_CONST_OCML_BINARY_F32(hypot)
DECL_CONST_OCML_UNARYPRED_F32(ilogb)
DECL_CONST_OCML_UNARYPRED_F32(isfinite)
DECL_CONST_OCML_UNARYPRED_F32(isinf)
DECL_CONST_OCML_UNARYPRED_F32(isnan)
DECL_CONST_OCML_UNARYPRED_F32(isnormal)
DECL_CONST_OCML_UNARY_F32(i0)
DECL_CONST_OCML_UNARY_F32(i1)
DECL_CONST_OCML_UNARY_F32(j0)
DECL_CONST_OCML_UNARY_F32(j1)
extern __attribute__((const)) float OCML_MANGLE_F32(ldexp)(float, int);
DECL_CONST_OCML_TERNARY_F32(len3)
extern __attribute__((const)) float OCML_MANGLE_F32(len4)(float, float, float, float);
DECL_CONST_OCML_UNARY_F32(lgamma)
extern float OCML_MANGLE_F32(lgamma_r)(float, __private int *);
DECL_CONST_OCML_UNARY_F32(log)
DECL_CONST_OCML_UNARY_F32(log2)
DECL_CONST_OCML_UNARY_F32(log10)
DECL_CONST_OCML_UNARY_F32(log1p)
DECL_CONST_OCML_UNARY_F32(logb)
DECL_CONST_OCML_TERNARY_F32(mad)
DECL_CONST_OCML_TERNARY_2F32(mad)
DECL_CONST_OCML_BINARY_F32(max)
DECL_CONST_OCML_BINARY_F32(min)
DECL_CONST_OCML_BINARY_F32(maxmag)
DECL_CONST_OCML_BINARY_F32(minmag)
extern float OCML_MANGLE_F32(modf)(float, __private float *);
extern __attribute__((const)) float OCML_MANGLE_F32(nan)(uint);
DECL_CONST_OCML_UNARY_F32(ncdf)
DECL_CONST_OCML_UNARY_F32(ncdfinv)
DECL_CONST_OCML_UNARY_F32(nearbyint)
DECL_CONST_OCML_BINARY_F32(nextafter)
DECL_CONST_OCML_BINARY_F32(pow)
DECL_CONST_OCML_BINARY_F32(powr)
extern __attribute__((pure)) float OCML_MANGLE_F32(pown)(float, int);
extern __attribute__((pure)) float OCML_MANGLE_F32(rootn)(float, int);
DECL_CONST_OCML_UNARY_F32(pred)
DECL_CONST_OCML_BINARY_F32(remainder)

typedef struct __ocml_remquo_f32_result {
    float rem;
    int quo;
} __ocml_remquo_f32_result;

extern __ocml_remquo_f32_result OCML_MANGLE_F32(remquo2)(float, float);

OCML_DEPRECATED(OCML_MANGLE_F32(remquo), "__ocml_remquo2_f32")
extern float OCML_MANGLE_F32(remquo)(float, float, __private int *);
DECL_CONST_OCML_BINARY_F32(rhypot)
DECL_CONST_OCML_UNARY_F32(rint)
DECL_CONST_OCML_TERNARY_F32(rlen3)
extern __attribute__((const)) float OCML_MANGLE_F32(rlen4)(float, float, float, float);
DECL_CONST_OCML_UNARY_F32(round)
DECL_CONST_OCML_UNARY_F32(rcbrt)
DECL_CONST_OCML_UNARY_F32(rsqrt)
DECL_CONST_OCML_BINARY_F32(scalb)
extern __attribute__((const)) float OCML_MANGLE_F32(scalbn)(float, int);
DECL_CONST_OCML_UNARYPRED_F32(signbit)
DECL_CONST_OCML_UNARY_F32(sin)
DECL_CONST_OCML_UNARY_F32(sinh)
DECL_CONST_OCML_UNARY_F32(sinpi)
extern float OCML_MANGLE_F32(sincos)(float, __private float *);
extern float OCML_MANGLE_F32(sincospi)(float, __private float *);
DECL_CONST_OCML_UNARY_F32(sqrt)
DECL_CONST_OCML_UNARY_F32(succ)
DECL_OCML_UNARY_F32(tan)
DECL_CONST_OCML_UNARY_F32(tanpi)
DECL_CONST_OCML_UNARY_F32(tanh)
DECL_CONST_OCML_UNARY_F32(tgamma)
DECL_CONST_OCML_UNARY_F32(trunc)
DECL_CONST_OCML_UNARY_F32(y0)
DECL_CONST_OCML_UNARY_F32(y1)

DECL_CONST_OCML_BINARY_F32(add_rte)
DECL_CONST_OCML_BINARY_F32(add_rtp)
DECL_CONST_OCML_BINARY_F32(add_rtn)
DECL_CONST_OCML_BINARY_F32(add_rtz)

DECL_CONST_OCML_BINARY_F32(div_rte)
DECL_CONST_OCML_BINARY_F32(div_rtp)
DECL_CONST_OCML_BINARY_F32(div_rtn)
DECL_CONST_OCML_BINARY_F32(div_rtz)

DECL_CONST_OCML_TERNARY_F32(fma_rte)
DECL_CONST_OCML_TERNARY_F32(fma_rtp)
DECL_CONST_OCML_TERNARY_F32(fma_rtn)
DECL_CONST_OCML_TERNARY_F32(fma_rtz)

DECL_CONST_OCML_BINARY_F32(mul_rte)
DECL_CONST_OCML_BINARY_F32(mul_rtp)
DECL_CONST_OCML_BINARY_F32(mul_rtn)
DECL_CONST_OCML_BINARY_F32(mul_rtz)

DECL_CONST_OCML_UNARY_F32(sqrt_rte)
DECL_CONST_OCML_UNARY_F32(sqrt_rtp)
DECL_CONST_OCML_UNARY_F32(sqrt_rtn)
DECL_CONST_OCML_UNARY_F32(sqrt_rtz)

DECL_CONST_OCML_BINARY_F32(sub_rte)
DECL_CONST_OCML_BINARY_F32(sub_rtp)
DECL_CONST_OCML_BINARY_F32(sub_rtn)
DECL_CONST_OCML_BINARY_F32(sub_rtz)


DECL_CONST_OCML_UNARY_F64(acos)
DECL_CONST_OCML_UNARY_F64(acosh)
DECL_CONST_OCML_UNARY_F64(acospi)
DECL_CONST_OCML_UNARY_F64(asin)
DECL_CONST_OCML_UNARY_F64(asinh)
DECL_CONST_OCML_UNARY_F64(asinpi)
DECL_CONST_OCML_UNARY_F64(atan)
DECL_CONST_OCML_UNARY_F64(atanh)
DECL_CONST_OCML_UNARY_F64(atanpi)
DECL_CONST_OCML_BINARY_F64(atan2)
DECL_CONST_OCML_BINARY_F64(atan2pi)
DECL_CONST_OCML_UNARY_F64(cbrt)
DECL_CONST_OCML_UNARY_F64(ceil)
DECL_CONST_OCML_BINARY_F64(copysign)
DECL_CONST_OCML_UNARY_F64(cos)
DECL_CONST_OCML_UNARY_F64(cosh)
DECL_CONST_OCML_UNARY_F64(cospi)
DECL_CONST_OCML_UNARY_F64(erf)
DECL_CONST_OCML_UNARY_F64(erfc)
DECL_CONST_OCML_UNARY_F64(erfinv)
DECL_CONST_OCML_UNARY_F64(erfcinv)
DECL_CONST_OCML_UNARY_F64(erfcx)
DECL_CONST_OCML_UNARY_F64(exp)
DECL_CONST_OCML_UNARY_F64(exp2)
DECL_CONST_OCML_UNARY_F64(exp10)
DECL_CONST_OCML_UNARY_F64(expm1)
DECL_CONST_OCML_UNARY_F64(fabs)
DECL_CONST_OCML_BINARY_F64(fdim)
DECL_CONST_OCML_UNARY_F64(floor)
DECL_CONST_OCML_TERNARY_F64(fma)
DECL_CONST_OCML_TERNARY_F64(fmuladd)
DECL_CONST_OCML_BINARY_F64(fmax)
DECL_CONST_OCML_BINARY_F64(fmin)
DECL_CONST_OCML_BINARY_F64(fmod)
DECL_CONST_OCML_UNARYPRED_F64(fpclassify)
extern double OCML_MANGLE_F64(fract)(double, __private double *);
extern double OCML_MANGLE_F64(frexp)(double, __private int *);
DECL_CONST_OCML_BINARY_F64(hypot)
DECL_CONST_OCML_UNARYPRED_F64(ilogb)
DECL_CONST_OCML_UNARYPRED_F64(isfinite)
DECL_CONST_OCML_UNARYPRED_F64(isinf)
DECL_CONST_OCML_UNARYPRED_F64(isnan)
DECL_CONST_OCML_UNARYPRED_F64(isnormal)
DECL_CONST_OCML_UNARY_F64(i0)
DECL_CONST_OCML_UNARY_F64(i1)
DECL_CONST_OCML_UNARY_F64(j0)
DECL_CONST_OCML_UNARY_F64(j1)
extern __attribute__((const)) double OCML_MANGLE_F64(ldexp)(double, int);
DECL_CONST_OCML_TERNARY_F64(len3)
extern __attribute__((const)) double OCML_MANGLE_F64(len4)(double, double, double, double);
DECL_CONST_OCML_UNARY_F64(lgamma)
extern double OCML_MANGLE_F64(lgamma_r)(double, __private int *);
DECL_CONST_OCML_UNARY_F64(log)
DECL_CONST_OCML_UNARY_F64(log2)
DECL_CONST_OCML_UNARY_F64(log10)
DECL_CONST_OCML_UNARY_F64(log1p)
DECL_CONST_OCML_UNARY_F64(logb)
DECL_CONST_OCML_TERNARY_F64(mad)
DECL_CONST_OCML_BINARY_F64(max)
DECL_CONST_OCML_BINARY_F64(min)
DECL_CONST_OCML_BINARY_F64(maxmag)
DECL_CONST_OCML_BINARY_F64(minmag)
extern double OCML_MANGLE_F64(modf)(double, __private double *);
extern __attribute__((const)) double OCML_MANGLE_F64(nan)(ulong);
DECL_CONST_OCML_UNARY_F64(ncdf)
DECL_CONST_OCML_UNARY_F64(ncdfinv)
DECL_CONST_OCML_UNARY_F64(nearbyint)
DECL_CONST_OCML_BINARY_F64(nextafter)
DECL_CONST_OCML_BINARY_F64(pow)
DECL_CONST_OCML_BINARY_F64(powr)
extern __attribute__((pure)) double OCML_MANGLE_F64(pown)(double, int);
extern __attribute__((pure)) double OCML_MANGLE_F64(rootn)(double, int);
DECL_CONST_OCML_UNARY_F64(pred)
DECL_CONST_OCML_BINARY_F64(remainder)


typedef struct __ocml_remquo_f64_result {
    double rem;
    int quo;
} __ocml_remquo_f64_result;

extern __ocml_remquo_f64_result OCML_MANGLE_F64(remquo2)(double, double);

OCML_DEPRECATED(OCML_MANGLE_F64(remquo), "__ocml_remquo2_f64")
extern double OCML_MANGLE_F64(remquo)(double, double, __private int *);
DECL_CONST_OCML_BINARY_F64(rhypot)
DECL_CONST_OCML_UNARY_F64(rint)
DECL_CONST_OCML_TERNARY_F64(rlen3)
extern __attribute__((const)) double OCML_MANGLE_F64(rlen4)(double, double, double, double);
DECL_CONST_OCML_UNARY_F64(round)
DECL_CONST_OCML_UNARY_F64(rcbrt)
DECL_CONST_OCML_UNARY_F64(rsqrt)
DECL_CONST_OCML_BINARY_F64(scalb)
extern __attribute__((const)) double OCML_MANGLE_F64(scalbn)(double, int);
DECL_CONST_OCML_UNARYPRED_F64(signbit)
DECL_CONST_OCML_UNARY_F64(sin)
extern double OCML_MANGLE_F64(sincos)(double, __private double *);
extern double OCML_MANGLE_F64(sincospi)(double, __private double *);
DECL_CONST_OCML_UNARY_F64(sinh)
DECL_CONST_OCML_UNARY_F64(sinpi)
DECL_CONST_OCML_UNARY_F64(sqrt)
DECL_CONST_OCML_UNARY_F64(succ)
DECL_CONST_OCML_UNARY_F64(tan)
DECL_CONST_OCML_UNARY_F64(tanh)
DECL_CONST_OCML_UNARY_F64(tanpi)
DECL_CONST_OCML_UNARY_F64(tgamma)
DECL_CONST_OCML_UNARY_F64(trunc)
DECL_CONST_OCML_UNARY_F64(y0)
DECL_CONST_OCML_UNARY_F64(y1)

DECL_CONST_OCML_BINARY_F64(add_rte)
DECL_CONST_OCML_BINARY_F64(add_rtp)
DECL_CONST_OCML_BINARY_F64(add_rtn)
DECL_CONST_OCML_BINARY_F64(add_rtz)

DECL_CONST_OCML_BINARY_F64(div_rte)
DECL_CONST_OCML_BINARY_F64(div_rtp)
DECL_CONST_OCML_BINARY_F64(div_rtn)
DECL_CONST_OCML_BINARY_F64(div_rtz)

DECL_CONST_OCML_TERNARY_F64(fma_rte)
DECL_CONST_OCML_TERNARY_F64(fma_rtp)
DECL_CONST_OCML_TERNARY_F64(fma_rtn)
DECL_CONST_OCML_TERNARY_F64(fma_rtz)

DECL_CONST_OCML_BINARY_F64(mul_rte)
DECL_CONST_OCML_BINARY_F64(mul_rtp)
DECL_CONST_OCML_BINARY_F64(mul_rtn)
DECL_CONST_OCML_BINARY_F64(mul_rtz)

DECL_CONST_OCML_UNARY_F64(sqrt_rte)
DECL_CONST_OCML_UNARY_F64(sqrt_rtp)
DECL_CONST_OCML_UNARY_F64(sqrt_rtn)
DECL_CONST_OCML_UNARY_F64(sqrt_rtz)

DECL_CONST_OCML_BINARY_F64(sub_rte)
DECL_CONST_OCML_BINARY_F64(sub_rtp)
DECL_CONST_OCML_BINARY_F64(sub_rtn)
DECL_CONST_OCML_BINARY_F64(sub_rtz)


DECL_CONST_OCML_UNARY_F32(native_recip)
DECL_CONST_OCML_UNARY_F64(native_recip)

DECL_CONST_OCML_UNARY_F32(native_sqrt)
DECL_CONST_OCML_UNARY_F64(native_sqrt)

DECL_CONST_OCML_UNARY_F32(native_rsqrt)
DECL_CONST_OCML_UNARY_F64(native_rsqrt)

DECL_CONST_OCML_UNARY_F32(native_sin)
DECL_CONST_OCML_UNARY_F64(native_sin)

DECL_CONST_OCML_UNARY_F32(native_cos)
DECL_CONST_OCML_UNARY_F64(native_cos)

DECL_CONST_OCML_UNARY_F32(native_exp)
DECL_CONST_OCML_UNARY_F64(native_exp)

DECL_CONST_OCML_UNARY_F32(native_exp2)
DECL_CONST_OCML_UNARY_F64(native_exp2)

DECL_CONST_OCML_UNARY_F32(native_exp10)

DECL_CONST_OCML_UNARY_F32(native_log)
DECL_CONST_OCML_UNARY_F64(native_log)

DECL_CONST_OCML_UNARY_F32(native_log2)
DECL_CONST_OCML_UNARY_F64(native_log2)

DECL_CONST_OCML_UNARY_F32(native_log10)
DECL_CONST_OCML_UNARY_F64(native_log10)


#pragma OPENCL EXTENSION cl_khr_fp16 : enable
DECL_CONST_OCML_UNARY_F16(acos)
DECL_CONST_OCML_UNARY_F16(acosh)
DECL_CONST_OCML_UNARY_F16(acospi)
DECL_CONST_OCML_UNARY_F16(asin)
DECL_CONST_OCML_UNARY_F16(asinh)
DECL_CONST_OCML_UNARY_F16(asinpi)
DECL_CONST_OCML_UNARY_F16(atan)
DECL_CONST_OCML_UNARY_F16(atanh)
DECL_CONST_OCML_UNARY_F16(atanpi)
DECL_CONST_OCML_BINARY_F16(atan2)
DECL_CONST_OCML_BINARY_F16(atan2pi)
DECL_CONST_OCML_UNARY_F16(cbrt)
DECL_CONST_OCML_UNARY_F16(ceil)
DECL_CONST_OCML_BINARY_F16(copysign)
DECL_CONST_OCML_UNARY_F16(cos)
DECL_CONST_OCML_UNARY_F16(cosh)
DECL_CONST_OCML_UNARY_F16(cospi)
DECL_CONST_OCML_UNARY_F16(erf)
DECL_CONST_OCML_UNARY_F16(erfc)
DECL_CONST_OCML_UNARY_F16(erfinv)
DECL_CONST_OCML_UNARY_F16(erfcinv)
DECL_CONST_OCML_UNARY_F16(erfcx)
DECL_CONST_OCML_UNARY_F16(exp)
DECL_CONST_OCML_UNARY_F16(exp2)
DECL_CONST_OCML_UNARY_F16(exp10)
DECL_CONST_OCML_UNARY_F16(expm1)
DECL_CONST_OCML_UNARY_F16(fabs)
DECL_CONST_OCML_BINARY_F16(fdim)
DECL_CONST_OCML_UNARY_F16(floor)
DECL_CONST_OCML_TERNARY_F16(fma)
DECL_CONST_OCML_TERNARY_F16(fmuladd)
DECL_CONST_OCML_TERNARY_F16(fma_rte)
DECL_CONST_OCML_TERNARY_F16(fma_rtp)
DECL_CONST_OCML_TERNARY_F16(fma_rtn)
DECL_CONST_OCML_TERNARY_F16(fma_rtz)
DECL_CONST_OCML_BINARY_F16(fmax)
DECL_CONST_OCML_BINARY_F16(fmin)
DECL_CONST_OCML_BINARY_F16(fmod)
DECL_CONST_OCML_UNARYPRED_F16(fpclassify)
extern half OCML_MANGLE_F16(fract)(half, __private half *);
extern half OCML_MANGLE_F16(frexp)(half, __private int *);
DECL_CONST_OCML_BINARY_F16(hypot)
DECL_CONST_OCML_UNARYPRED_F16(ilogb)
DECL_CONST_OCML_UNARYPRED_F16(isfinite)
DECL_CONST_OCML_UNARYPRED_F16(isinf)
DECL_CONST_OCML_UNARYPRED_F16(isnan)
DECL_CONST_OCML_UNARYPRED_F16(isnormal)
DECL_CONST_OCML_UNARY_F16(i0)
DECL_CONST_OCML_UNARY_F16(i1)
DECL_CONST_OCML_UNARY_F16(j0)
DECL_CONST_OCML_UNARY_F16(j1)
extern __attribute__((const)) half OCML_MANGLE_F16(ldexp)(half, int);
DECL_CONST_OCML_TERNARY_F16(len3)
extern __attribute__((const)) half OCML_MANGLE_F16(len4)(half, half, half, half);
DECL_CONST_OCML_UNARY_F16(lgamma)
extern half OCML_MANGLE_F16(lgamma_r)(half, __private int *);
DECL_CONST_OCML_UNARY_F16(log)
DECL_CONST_OCML_UNARY_F16(logb)
DECL_CONST_OCML_UNARY_F16(log2)
DECL_CONST_OCML_UNARY_F16(log10)
DECL_CONST_OCML_UNARY_F16(log1p)
DECL_CONST_OCML_TERNARY_F16(mad)
DECL_CONST_OCML_BINARY_F16(max)
DECL_CONST_OCML_BINARY_F16(min)
DECL_CONST_OCML_BINARY_F16(maxmag)
DECL_CONST_OCML_BINARY_F16(minmag)
extern half OCML_MANGLE_F16(modf)(half, __private half *);
extern __attribute__((const)) half OCML_MANGLE_F16(nan)(ushort);
DECL_CONST_OCML_UNARY_F16(ncdf)
DECL_CONST_OCML_UNARY_F16(ncdfinv)
DECL_CONST_OCML_UNARY_F16(nearbyint)
DECL_CONST_OCML_BINARY_F16(nextafter)
DECL_CONST_OCML_BINARY_F16(pow)
DECL_CONST_OCML_BINARY_F16(powr)
extern __attribute__((pure)) half OCML_MANGLE_F16(pown)(half, int);
extern __attribute__((pure)) half OCML_MANGLE_F16(rootn)(half, int);
DECL_CONST_OCML_UNARY_F16(pred)
DECL_CONST_OCML_UNARY_F16(rcbrt)
DECL_CONST_OCML_BINARY_F16(remainder)

typedef struct __ocml_remquo_f16_result {
    half rem;
    int quo;
} __ocml_remquo_f16_result;

extern __ocml_remquo_f16_result OCML_MANGLE_F16(remquo2)(half, half);

OCML_DEPRECATED(OCML_MANGLE_F16(remquo), "__ocml_remquo2_f16")
extern half OCML_MANGLE_F16(remquo)(half, half, __private int *);

DECL_CONST_OCML_BINARY_F16(rhypot)
DECL_CONST_OCML_UNARY_F16(rint)
DECL_CONST_OCML_TERNARY_F16(rlen3)
extern __attribute__((const)) half OCML_MANGLE_F16(rlen4)(half, half, half, half);
DECL_CONST_OCML_UNARY_F16(round)
DECL_CONST_OCML_UNARY_F16(rsqrt)
DECL_CONST_OCML_BINARY_F16(scalb)
extern __attribute__((const)) half OCML_MANGLE_F16(scalbn)(half, int);
DECL_CONST_OCML_UNARYPRED_F16(signbit)
DECL_CONST_OCML_UNARY_F16(sin)
DECL_CONST_OCML_UNARY_F16(sinh)
DECL_CONST_OCML_UNARY_F16(sinpi)
extern half OCML_MANGLE_F16(sincos)(half, __private half *);
extern half OCML_MANGLE_F16(sincospi)(half, __private half *);
DECL_CONST_OCML_UNARY_F16(sqrt)
DECL_CONST_OCML_UNARY_F16(sqrt_rte)
DECL_CONST_OCML_UNARY_F16(sqrt_rtp)
DECL_CONST_OCML_UNARY_F16(sqrt_rtn)
DECL_CONST_OCML_UNARY_F16(sqrt_rtz)
DECL_CONST_OCML_UNARY_F16(succ)
DECL_CONST_OCML_UNARY_F16(tan)
DECL_CONST_OCML_UNARY_F16(tanpi)
DECL_CONST_OCML_UNARY_F16(tanh)
DECL_CONST_OCML_UNARY_F16(tgamma)
DECL_CONST_OCML_UNARY_F16(trunc)
DECL_CONST_OCML_UNARY_F16(y0)
DECL_CONST_OCML_UNARY_F16(y1)

DECL_CONST_OCML_BINARY_F16(add_rte)
DECL_CONST_OCML_BINARY_F16(add_rtp)
DECL_CONST_OCML_BINARY_F16(add_rtn)
DECL_CONST_OCML_BINARY_F16(add_rtz)

DECL_CONST_OCML_BINARY_F16(div_rte)
DECL_CONST_OCML_BINARY_F16(div_rtp)
DECL_CONST_OCML_BINARY_F16(div_rtn)
DECL_CONST_OCML_BINARY_F16(div_rtz)

DECL_CONST_OCML_TERNARY_F16(fma_rte)
DECL_CONST_OCML_TERNARY_F16(fma_rtp)
DECL_CONST_OCML_TERNARY_F16(fma_rtn)
DECL_CONST_OCML_TERNARY_F16(fma_rtz)

DECL_CONST_OCML_BINARY_F16(mul_rte)
DECL_CONST_OCML_BINARY_F16(mul_rtp)
DECL_CONST_OCML_BINARY_F16(mul_rtn)
DECL_CONST_OCML_BINARY_F16(mul_rtz)

DECL_CONST_OCML_UNARY_F16(sqrt_rte)
DECL_CONST_OCML_UNARY_F16(sqrt_rtp)
DECL_CONST_OCML_UNARY_F16(sqrt_rtn)
DECL_CONST_OCML_UNARY_F16(sqrt_rtz)

DECL_CONST_OCML_BINARY_F16(sub_rte)
DECL_CONST_OCML_BINARY_F16(sub_rtp)
DECL_CONST_OCML_BINARY_F16(sub_rtn)
DECL_CONST_OCML_BINARY_F16(sub_rtz)

// 2-vector functions
DECL_CONST_OCML_UNARY_2F16(acos)
DECL_CONST_OCML_UNARY_2F16(acosh)
DECL_CONST_OCML_UNARY_2F16(acospi)
DECL_CONST_OCML_UNARY_2F16(asin)
DECL_CONST_OCML_UNARY_2F16(asinh)
DECL_CONST_OCML_UNARY_2F16(asinpi)
DECL_CONST_OCML_UNARY_2F16(atan)
DECL_CONST_OCML_UNARY_2F16(atanh)
DECL_CONST_OCML_UNARY_2F16(atanpi)
DECL_CONST_OCML_BINARY_2F16(atan2)
DECL_CONST_OCML_BINARY_2F16(atan2pi)
DECL_CONST_OCML_UNARY_2F16(cbrt)
DECL_CONST_OCML_UNARY_2F16(ceil)
DECL_CONST_OCML_BINARY_2F16(copysign)
DECL_CONST_OCML_UNARY_2F16(cos)
DECL_CONST_OCML_UNARY_2F16(cosh)
DECL_CONST_OCML_UNARY_2F16(cospi)
DECL_CONST_OCML_UNARY_2F16(erf)
DECL_CONST_OCML_UNARY_2F16(erfc)
DECL_CONST_OCML_UNARY_2F16(erfinv)
DECL_CONST_OCML_UNARY_2F16(erfcinv)
DECL_CONST_OCML_UNARY_2F16(erfcx)
DECL_CONST_OCML_UNARY_2F16(exp)
DECL_CONST_OCML_UNARY_2F16(exp2)
DECL_CONST_OCML_UNARY_2F16(exp10)
DECL_CONST_OCML_UNARY_2F16(expm1)
DECL_CONST_OCML_UNARY_2F16(fabs)
DECL_CONST_OCML_BINARY_2F16(fdim)
DECL_CONST_OCML_UNARY_2F16(floor)
DECL_CONST_OCML_TERNARY_2F16(fma)
DECL_CONST_OCML_TERNARY_2F16(fmuladd)
DECL_CONST_OCML_TERNARY_2F16(fma_rte)
DECL_CONST_OCML_TERNARY_2F16(fma_rtp)
DECL_CONST_OCML_TERNARY_2F16(fma_rtn)
DECL_CONST_OCML_TERNARY_2F16(fma_rtz)
DECL_CONST_OCML_BINARY_2F16(fmax)
DECL_CONST_OCML_BINARY_2F16(fmin)
DECL_CONST_OCML_BINARY_2F16(fmod)
DECL_CONST_OCML_UNARYPRED_2F16(fpclassify)
extern half2 OCML_MANGLE_2F16(fract)(half2, __private half2 *);
extern half2 OCML_MANGLE_2F16(frexp)(half2, __private int2 *);
DECL_CONST_OCML_BINARY_2F16(hypot)
extern __attribute__((const)) int2 OCML_MANGLE_2F16(ilogb)(half2);
DECL_CONST_OCML_UNARYPRED_2F16(isfinite)
DECL_CONST_OCML_UNARYPRED_2F16(isinf)
DECL_CONST_OCML_UNARYPRED_2F16(isnan)
DECL_CONST_OCML_UNARYPRED_2F16(isnormal)
DECL_CONST_OCML_UNARY_2F16(i0)
DECL_CONST_OCML_UNARY_2F16(i1)
DECL_CONST_OCML_UNARY_2F16(j0)
DECL_CONST_OCML_UNARY_2F16(j1)
extern __attribute__((const)) half2 OCML_MANGLE_2F16(ldexp)(half2, int2);
DECL_CONST_OCML_UNARY_2F16(lgamma)
extern half2 OCML_MANGLE_2F16(lgamma_r)(half2, __private int2 *);
DECL_CONST_OCML_UNARY_2F16(log)
DECL_CONST_OCML_UNARY_2F16(logb)
DECL_CONST_OCML_UNARY_2F16(log2)
DECL_CONST_OCML_UNARY_2F16(log10)
DECL_CONST_OCML_UNARY_2F16(log1p)
DECL_CONST_OCML_TERNARY_2F16(mad)
DECL_CONST_OCML_BINARY_2F16(max)
DECL_CONST_OCML_BINARY_2F16(min)
DECL_CONST_OCML_BINARY_2F16(maxmag)
DECL_CONST_OCML_BINARY_2F16(minmag)
extern half2 OCML_MANGLE_2F16(modf)(half2, __private half2 *);
extern __attribute__((const)) half2 OCML_MANGLE_2F16(nan)(ushort2);
DECL_CONST_OCML_UNARY_2F16(ncdf)
DECL_CONST_OCML_UNARY_2F16(ncdfinv)
DECL_CONST_OCML_UNARY_2F16(nearbyint)
DECL_CONST_OCML_BINARY_2F16(nextafter)
DECL_CONST_OCML_BINARY_2F16(pow)
DECL_CONST_OCML_BINARY_2F16(powr)
extern __attribute__((pure)) half2 OCML_MANGLE_2F16(pown)(half2, int2);
extern __attribute__((pure)) half2 OCML_MANGLE_2F16(rootn)(half2, int2);
DECL_CONST_OCML_UNARY_2F16(rcbrt)
DECL_CONST_OCML_BINARY_2F16(remainder)

typedef struct __ocml_remquo_2f16_result {
    half2 rem;
    int2 quo;
} __ocml_remquo_2f16_result;

extern __ocml_remquo_2f16_result OCML_MANGLE_2F16(remquo2)(half2, half2);

OCML_DEPRECATED(OCML_MANGLE_F16(remquo), "__ocml_remquo2_2f16")
extern half2 OCML_MANGLE_2F16(remquo)(half2, half2, __private int2 *);
DECL_CONST_OCML_UNARY_2F16(rint)
DECL_CONST_OCML_UNARY_2F16(round)
DECL_CONST_OCML_UNARY_2F16(rsqrt)
DECL_CONST_OCML_BINARY_2F16(scalb)
extern __attribute__((const)) half2 OCML_MANGLE_2F16(scalbn)(half2, int2);
DECL_CONST_OCML_UNARYPRED_2F16(signbit)
DECL_CONST_OCML_UNARY_2F16(sin)
DECL_CONST_OCML_UNARY_2F16(sinh)
DECL_CONST_OCML_UNARY_2F16(sinpi)
extern half2 OCML_MANGLE_2F16(sincos)(half2, __private half2 *);
extern half2 OCML_MANGLE_2F16(sincospi)(half2, __private half2 *);
DECL_CONST_OCML_UNARY_2F16(sqrt)
DECL_CONST_OCML_UNARY_2F16(sqrt_rte)
DECL_CONST_OCML_UNARY_2F16(sqrt_rtp)
DECL_CONST_OCML_UNARY_2F16(sqrt_rtn)
DECL_CONST_OCML_UNARY_2F16(sqrt_rtz)
DECL_CONST_OCML_UNARY_2F16(tan)
DECL_CONST_OCML_UNARY_2F16(tanpi)
DECL_CONST_OCML_UNARY_2F16(tanh)
DECL_CONST_OCML_UNARY_2F16(tgamma)
DECL_CONST_OCML_UNARY_2F16(trunc)
DECL_CONST_OCML_UNARY_2F16(y0)
DECL_CONST_OCML_UNARY_2F16(y1)

DECL_CONST_OCML_BINARY_2F16(add_rte)
DECL_CONST_OCML_BINARY_2F16(add_rtp)
DECL_CONST_OCML_BINARY_2F16(add_rtn)
DECL_CONST_OCML_BINARY_2F16(add_rtz)

DECL_CONST_OCML_BINARY_2F16(div_rte)
DECL_CONST_OCML_BINARY_2F16(div_rtp)
DECL_CONST_OCML_BINARY_2F16(div_rtn)
DECL_CONST_OCML_BINARY_2F16(div_rtz)

DECL_CONST_OCML_TERNARY_2F16(fma_rte)
DECL_CONST_OCML_TERNARY_2F16(fma_rtp)
DECL_CONST_OCML_TERNARY_2F16(fma_rtn)
DECL_CONST_OCML_TERNARY_2F16(fma_rtz)

DECL_CONST_OCML_BINARY_2F16(mul_rte)
DECL_CONST_OCML_BINARY_2F16(mul_rtp)
DECL_CONST_OCML_BINARY_2F16(mul_rtn)
DECL_CONST_OCML_BINARY_2F16(mul_rtz)

DECL_CONST_OCML_UNARY_2F16(sqrt_rte)
DECL_CONST_OCML_UNARY_2F16(sqrt_rtp)
DECL_CONST_OCML_UNARY_2F16(sqrt_rtn)
DECL_CONST_OCML_UNARY_2F16(sqrt_rtz)

DECL_CONST_OCML_BINARY_2F16(sub_rte)
DECL_CONST_OCML_BINARY_2F16(sub_rtp)
DECL_CONST_OCML_BINARY_2F16(sub_rtn)
DECL_CONST_OCML_BINARY_2F16(sub_rtz)

DECL_CONST_OCML_UNARY_F16(native_recip)
DECL_CONST_OCML_UNARY_F16(native_sqrt)
DECL_CONST_OCML_UNARY_F16(native_rsqrt)
DECL_CONST_OCML_UNARY_F16(native_sin)
DECL_CONST_OCML_UNARY_F16(native_cos)
DECL_CONST_OCML_UNARY_F16(native_exp2)
DECL_CONST_OCML_UNARY_F16(native_log2)

extern __attribute__((const)) float OCML_MANGLE_F32(cabs)(float2);
extern __attribute__((const)) double OCML_MANGLE_F64(cabs)(double2);

extern __attribute__((const)) float2 OCML_MANGLE_F32(cacos)(float2);
extern __attribute__((const)) double2 OCML_MANGLE_F64(cacos)(double2);

extern __attribute__((const)) float2 OCML_MANGLE_F32(cacosh)(float2);
extern __attribute__((const)) double2 OCML_MANGLE_F64(cacosh)(double2);

extern __attribute__((const)) float2 OCML_MANGLE_F32(casin)(float2);
extern __attribute__((const)) double2 OCML_MANGLE_F64(casin)(double2);

extern __attribute__((const)) float2 OCML_MANGLE_F32(casinh)(float2);
extern __attribute__((const)) double2 OCML_MANGLE_F64(casinh)(double2);

extern __attribute__((const)) float2 OCML_MANGLE_F32(catan)(float2);
extern __attribute__((const)) double2 OCML_MANGLE_F64(catan)(double2);

extern __attribute__((const)) float2 OCML_MANGLE_F32(catanh)(float2);
extern __attribute__((const)) double2 OCML_MANGLE_F64(catanh)(double2);

extern __attribute__((const)) float2 OCML_MANGLE_F32(cexp)(float2);
extern __attribute__((const)) double2 OCML_MANGLE_F64(cexp)(double2);

extern __attribute__((const)) float2 OCML_MANGLE_F32(clog)(float2);
extern __attribute__((const)) double2 OCML_MANGLE_F64(clog)(double2);

extern __attribute__((const)) float2 OCML_MANGLE_F32(ccos)(float2);
extern __attribute__((const)) double2 OCML_MANGLE_F64(ccos)(double2);

extern __attribute__((const)) float2 OCML_MANGLE_F32(ccosh)(float2);
extern __attribute__((const)) double2 OCML_MANGLE_F64(ccosh)(double2);

extern __attribute__((const)) float2 OCML_MANGLE_F32(csin)(float2);
extern __attribute__((const)) double2 OCML_MANGLE_F64(csin)(double2);

extern __attribute__((const)) float2 OCML_MANGLE_F32(csinh)(float2);
extern __attribute__((const)) double2 OCML_MANGLE_F64(csinh)(double2);

extern __attribute__((const)) float2 OCML_MANGLE_F32(ctan)(float2);
extern __attribute__((const)) double2 OCML_MANGLE_F64(ctan)(double2);

extern __attribute__((const)) float2 OCML_MANGLE_F32(ctanh)(float2);
extern __attribute__((const)) double2 OCML_MANGLE_F64(ctanh)(double2);

extern __attribute__((const)) float2 OCML_MANGLE_F32(csqrt)(float2);
extern __attribute__((const)) double2 OCML_MANGLE_F64(csqrt)(double2);

extern __attribute__((const)) float2 OCML_MANGLE_F32(cdiv)(float2, float2);
extern __attribute__((const)) double2 OCML_MANGLE_F64(cdiv)(double2, double2);

extern __attribute__((const)) half OCML_MANGLE_F32(cvtrtn_f16)(float a);
extern __attribute__((const)) half OCML_MANGLE_F32(cvtrtp_f16)(float a);
extern __attribute__((const)) half OCML_MANGLE_F32(cvtrtz_f16)(float a);
extern __attribute__((const)) half OCML_MANGLE_F64(cvtrte_f16)(double a);
extern __attribute__((const)) half OCML_MANGLE_F64(cvtrtn_f16)(double a);
extern __attribute__((const)) half OCML_MANGLE_F64(cvtrtp_f16)(double a);
extern __attribute__((const)) half OCML_MANGLE_F64(cvtrtz_f16)(double a);
extern __attribute__((const)) float OCML_MANGLE_F64(cvtrtn_f32)(double a);
extern __attribute__((const)) float OCML_MANGLE_F64(cvtrtp_f32)(double a);
extern __attribute__((const)) float OCML_MANGLE_F64(cvtrtz_f32)(double a);
extern __attribute__((const)) float OCML_MANGLE_S32(cvtrtn_f32)(int);
extern __attribute__((const)) float OCML_MANGLE_S32(cvtrtp_f32)(int);
extern __attribute__((const)) float OCML_MANGLE_S32(cvtrtz_f32)(int);
extern __attribute__((const)) float OCML_MANGLE_U32(cvtrtn_f32)(uint);
extern __attribute__((const)) float OCML_MANGLE_U32(cvtrtp_f32)(uint);
extern __attribute__((const)) float OCML_MANGLE_U32(cvtrtz_f32)(uint);
extern __attribute__((const)) float OCML_MANGLE_S64(cvtrtn_f32)(long);
extern __attribute__((const)) float OCML_MANGLE_S64(cvtrtp_f32)(long);
extern __attribute__((const)) float OCML_MANGLE_S64(cvtrtz_f32)(long);
extern __attribute__((const)) float OCML_MANGLE_U64(cvtrtn_f32)(ulong);
extern __attribute__((const)) float OCML_MANGLE_U64(cvtrtp_f32)(ulong);
extern __attribute__((const)) float OCML_MANGLE_U64(cvtrtz_f32)(ulong);
extern __attribute__((const)) double OCML_MANGLE_S64(cvtrtn_f64)(long);
extern __attribute__((const)) double OCML_MANGLE_S64(cvtrtp_f64)(long);
extern __attribute__((const)) double OCML_MANGLE_S64(cvtrtz_f64)(long);
extern __attribute__((const)) double OCML_MANGLE_U64(cvtrtn_f64)(ulong);
extern __attribute__((const)) double OCML_MANGLE_U64(cvtrtp_f64)(ulong);
extern __attribute__((const)) double OCML_MANGLE_U64(cvtrtz_f64)(ulong);

#pragma OPENCL EXTENSION cl_khr_fp16 : disable

#endif // OCML_H
