/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#include <complex.h>
#include "float128.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Real.
 */
typedef	float128_t vrq1_t;
typedef	double	vrd1_t;
typedef	double	vrd2_t	__attribute__((vector_size(2*sizeof(double))));
typedef	double	vrd4_t	__attribute__((vector_size(4*sizeof(double))));
typedef	double	vrd8_t	__attribute__((vector_size(8*sizeof(double))));
typedef	float	vrs1_t;
typedef	float	vrs4_t	__attribute__((vector_size(4*sizeof(float))));
typedef	float	vrs8_t	__attribute__((vector_size(8*sizeof(float))));
typedef	float	vrs16_t	__attribute__((vector_size(16*sizeof(float))));

/*
 * Complex.
 *
 * Note:
 * Vector structures cannot be made up of structures contaning real and
 * imaginary components.
 * As such, complex vector structures are in name only and simply
 * overloaded to the REALs.  To extract the R and i's, other macros or
 * C constructs must be used.
 */

typedef	double	vcd1_t	__attribute__((vector_size(2*sizeof(double))));
typedef	double	vcd2_t	__attribute__((vector_size(4*sizeof(double))));
typedef	double	vcd4_t	__attribute__((vector_size(8*sizeof(double))));
typedef	float	vcs1_t	__attribute__((vector_size(2*sizeof(float))));
typedef	float	vcs2_t	__attribute__((vector_size(4*sizeof(float))));
typedef	float	vcs4_t	__attribute__((vector_size(8*sizeof(float))));
typedef	float	vcs8_t	__attribute__((vector_size(16*sizeof(float))));

/*
 * Integer.
 */

typedef	int32_t	vis1_t;
typedef	int32_t	vis2_t	__attribute__((vector_size(2*sizeof(int32_t))));
typedef	int32_t	vis4_t	__attribute__((vector_size(4*sizeof(int32_t))));
typedef	int32_t	vis8_t	__attribute__((vector_size(8*sizeof(int32_t))));
typedef	int32_t	vis16_t	__attribute__((vector_size(16*sizeof(int32_t))));
typedef	int64_t	vid1_t;
typedef	int64_t	vid2_t	__attribute__((vector_size(2*sizeof(int64_t))));
typedef	int64_t	vid4_t	__attribute__((vector_size(4*sizeof(int64_t))));
typedef	int64_t	vid8_t	__attribute__((vector_size(8*sizeof(int64_t))));

extern	vrs4_t	__ZGVxN4v__mth_i_vr4		(vrs4_t, float(*)(float));
extern	vrs4_t	__ZGVxM4v__mth_i_vr4		(vrs4_t, vis4_t, float(*)(float));
extern	vrs4_t	__ZGVxN4vv__mth_i_vr4vr4	(vrs4_t, vrs4_t, float(*)(float, float));
extern	vrs4_t	__ZGVxM4vv__mth_i_vr4vr4	(vrs4_t, vrs4_t, vis4_t, float(*)(float, float));
extern	vrd2_t	__ZGVxN2v__mth_i_vr8		(vrd2_t, double(*)(double));
extern	vrd2_t	__ZGVxM2v__mth_i_vr8		(vrd2_t, vid2_t, double(*)(double));
extern	vrd2_t	__ZGVxN2vv__mth_i_vr8vr8	(vrd2_t, vrd2_t, double(*)(double, double));
extern	vrd2_t	__ZGVxM2vv__mth_i_vr8vr8	(vrd2_t, vrd2_t, vid2_t, double(*)(double, double));
extern	vrs8_t	__ZGVyN8v__mth_i_vr4		(vrs8_t, float(*)(float));
extern	vrs8_t	__ZGVyM8v__mth_i_vr4		(vrs8_t, vis8_t, float(*)(float));
extern	vrs8_t	__ZGVyN8vv__mth_i_vr4vr4	(vrs8_t, vrs8_t, float(*)(float, float));
extern	vrs8_t	__ZGVyM8vv__mth_i_vr4vr4	(vrs8_t, vrs8_t, vis8_t, float(*)(float, float));
extern	vrd4_t	__ZGVyN4v__mth_i_vr8		(vrd4_t, double(*)(double));
extern	vrd4_t	__ZGVyM4v__mth_i_vr8		(vrd4_t, vid4_t, double(*)(double));
extern	vrd4_t	__ZGVyN4vv__mth_i_vr8vr8	(vrd4_t, vrd4_t, double(*)(double, double));
extern	vrd4_t	__ZGVyM4vv__mth_i_vr8vr8	(vrd4_t, vrd4_t, vid4_t, double(*)(double, double));
extern	vrs16_t	__ZGVzN16v__mth_i_vr4		(vrs16_t, float(*)(float));
extern	vrs16_t	__ZGVzM16v__mth_i_vr4		(vrs16_t, vis16_t, float(*)(float));
extern	vrs16_t	__ZGVzN16vv__mth_i_vr4vr4	(vrs16_t, vrs16_t, float(*)(float, float));
extern	vrs16_t	__ZGVzM16vv__mth_i_vr4vr4	(vrs16_t, vrs16_t, vis16_t, float(*)(float, float));
extern	vrd8_t	__ZGVzN8v__mth_i_vr8		(vrd8_t, double(*)(double));
extern	vrd8_t	__ZGVzM8v__mth_i_vr8		(vrd8_t, vid8_t, double(*)(double));
extern	vrd8_t	__ZGVzN8vv__mth_i_vr8vr8	(vrd8_t, vrd8_t, double(*)(double, double));
extern	vrd8_t	__ZGVzM8vv__mth_i_vr8vr8	(vrd8_t, vrd8_t, vid8_t, double(*)(double, double));

#ifdef __cplusplus
} /* extern "C" */
#endif

/* Complex */
#ifdef __cplusplus
extern "C" {
#endif
extern	vcs1_t	__ZGVxN1v__mth_i_vc4		(vcs1_t, _Complex float func(_Complex float));
extern	vcs1_t	__ZGVxN1vv__mth_i_vc4vc4	(vcs1_t, vcs1_t, _Complex float func(_Complex float, _Complex float));
extern	vcs2_t	__ZGVxN2v__mth_i_vc4		(vcs2_t, _Complex float func(_Complex float));
extern	vcs2_t	__ZGVxN2vv__mth_i_vc4vc4	(vcs2_t, vcs2_t, _Complex float func(_Complex float, _Complex float));
extern	vcd1_t	__ZGVxN1v__mth_i_vc8		(vcd1_t, _Complex double func(_Complex double));
extern	vcd1_t	__ZGVxN1vv__mth_i_vc8vc8	(vcd1_t, vcd1_t, _Complex double func(_Complex double, _Complex double));

extern	vcs4_t	__ZGVyN4v__mth_i_vc4		(vcs4_t, _Complex float func(_Complex float));
extern	vcs4_t	__ZGVyN4vv__mth_i_vc4vc4	(vcs4_t, vcs4_t, _Complex float func(_Complex float, _Complex float));
extern	vcd2_t	__ZGVyN2v__mth_i_vc8		(vcd2_t, _Complex double func(_Complex double));
extern	vcd2_t	__ZGVyN2vv__mth_i_vc8vc8	(vcd2_t, vcd2_t, _Complex double func(_Complex double, _Complex double));

extern	vcs8_t	__ZGVzN8v__mth_i_vc4		(vcs8_t, _Complex float func(_Complex float));
extern	vcs8_t	__ZGVzN8vv__mth_i_vc4vc4	(vcs8_t, vcs8_t, _Complex float func(_Complex float, _Complex float));
extern	vcd4_t	__ZGVzN4v__mth_i_vc8		(vcd4_t, _Complex double func(_Complex double));
extern	vcd4_t	__ZGVzN4vv__mth_i_vc8vc8	(vcd4_t, vcd4_t, _Complex double func(_Complex double, _Complex double));

extern	vrs4_t  __ZGVxN4v__mth_i_vr4si4   (vrs4_t, int32_t, float (*)(float, int32_t));
extern	vrs4_t  __ZGVxM4v__mth_i_vr4si4   (vrs4_t, int32_t, vis4_t, float (*)(float, int32_t));
extern	vrs4_t  __ZGVxN4vv__mth_i_vr4vi4  (vrs4_t, vis4_t, float (*)(float, int32_t));
extern	vrs4_t  __ZGVxM4vv__mth_i_vr4vi4  (vrs4_t, vis4_t, vis4_t, float (*)(float, int32_t));
extern	vrs4_t  __ZGVxN4v__mth_i_vr4si8   (vrs4_t, long long, float (*)(float, long long));
extern	vrs4_t  __ZGVxM4v__mth_i_vr4si8   (vrs4_t, long long, vis4_t, float (*)(float, long long));
extern	vrs4_t  __ZGVxN4vv__mth_i_vr4vi8  (vrs4_t, vid2_t, vid2_t, float (*)(float, long long));
extern	vrs4_t  __ZGVxM4vv__mth_i_vr4vi8  (vrs4_t, vid2_t, vid2_t, vis4_t, float (*)(float, long long));
extern	vrd2_t  __ZGVxN2v__mth_i_vr8si4   (vrd2_t, int32_t, double (*)(double, int32_t));
extern	vrd2_t  __ZGVxM2v__mth_i_vr8si4   (vrd2_t, int32_t, vid2_t, double (*)(double, int32_t));

#ifdef __cplusplus
} /* extern "C" */
#endif

/*
 * POWER architecture needs the 32-bit integer vector array to be defined as a
 * full vector size - not required for X86-64 architectures.
 * Technically these worker functions should be defined as
 * extern	vrd2_t  __ZGVxN2vv__mth_i_vr8vi4  (vrd2_t, vis2_t, double (*)(double, int32_t));
 * extern	vrd2_t  __ZGVxM2vv__mth_i_vr8vi4  (vrd2_t, vis2_t, vid2_t, double (*)(double, int32_t));
 */
#ifdef __cplusplus
extern "C" {
#endif

extern	vrd2_t  __ZGVxN2vv__mth_i_vr8vi4  (vrd2_t, vis4_t, double (*)(double, int32_t));
extern	vrd2_t  __ZGVxM2vv__mth_i_vr8vi4  (vrd2_t, vis4_t, vid2_t, double (*)(double, int32_t));
extern	vrd2_t  __ZGVxN2v__mth_i_vr8si8   (vrd2_t, long long, double (*)(double, long long));
extern	vrd2_t  __ZGVxM2v__mth_i_vr8si8   (vrd2_t, long long, vid2_t, double (*)(double, long long));
extern	vrd2_t  __ZGVxN2vv__mth_i_vr8vi8  (vrd2_t, vid2_t, double (*)(double, long long));
extern	vrd2_t  __ZGVxM2vv__mth_i_vr8vi8  (vrd2_t, vid2_t, vid2_t, double (*)(double, long long));
extern	vrs8_t  __ZGVyN8v__mth_i_vr4si4   (vrs8_t, int32_t, float (*)(float, int32_t));
extern	vrs8_t  __ZGVyM8v__mth_i_vr4si4   (vrs8_t, int32_t, vis8_t, float (*)(float, int32_t));
extern	vrs8_t  __ZGVyN8vv__mth_i_vr4vi4  (vrs8_t, vis8_t, float (*)(float, int32_t));
extern	vrs8_t  __ZGVyM8vv__mth_i_vr4vi4  (vrs8_t, vis8_t, vis8_t, float (*)(float, int32_t));
extern	vrs8_t  __ZGVyN8v__mth_i_vr4si8   (vrs8_t, long long, float (*)(float, long long));
extern	vrs8_t  __ZGVyM8v__mth_i_vr4si8   (vrs8_t, long long, vis8_t, float (*)(float, long long));
extern	vrs8_t  __ZGVyN8vv__mth_i_vr4vi8  (vrs8_t, vid4_t, vid4_t, float (*)(float, long long));
extern	vrs8_t  __ZGVyM8vv__mth_i_vr4vi8  (vrs8_t, vid4_t, vid4_t, vis8_t, float (*)(float, long long));
extern	vrd4_t  __ZGVyN4v__mth_i_vr8si4   (vrd4_t, int32_t, double (*)(double, int32_t));
extern	vrd4_t  __ZGVyM4v__mth_i_vr8si4   (vrd4_t, int32_t, vid4_t, double (*)(double, int32_t));
extern	vrd4_t  __ZGVyN4vv__mth_i_vr8vi4  (vrd4_t, vis4_t, double (*)(double, int32_t));
extern	vrd4_t  __ZGVyM4vv__mth_i_vr8vi4  (vrd4_t, vis4_t, vid4_t, double (*)(double, int32_t));
extern	vrd4_t  __ZGVyN4v__mth_i_vr8si8   (vrd4_t, long long, double (*)(double, long long));
extern	vrd4_t  __ZGVyM4v__mth_i_vr8si8   (vrd4_t, long long, vid4_t, double (*)(double, long long));
extern	vrd4_t  __ZGVyN4vv__mth_i_vr8vi8  (vrd4_t, vid4_t, double (*)(double, long long));
extern	vrd4_t  __ZGVyM4vv__mth_i_vr8vi8  (vrd4_t, vid4_t, vid4_t, double (*)(double, long long));
extern	vrs16_t __ZGVzN16v__mth_i_vr4si4  (vrs16_t, int32_t, float (*)(float, int32_t));
extern	vrs16_t __ZGVzM16v__mth_i_vr4si4  (vrs16_t, int32_t, vis16_t, float (*)(float, int32_t));
extern	vrs16_t __ZGVzN16vv__mth_i_vr4vi4 (vrs16_t, vis16_t, float (*)(float, int32_t));
extern	vrs16_t __ZGVzM16vv__mth_i_vr4vi4 (vrs16_t, vis16_t, vis16_t, float (*)(float, int32_t));
extern	vrs16_t __ZGVzN16v__mth_i_vr4si8  (vrs16_t, long long, float (*)(float, long long));
extern	vrs16_t __ZGVzM16v__mth_i_vr4si8  (vrs16_t, long long, vis16_t, float (*)(float, long long));
extern	vrs16_t __ZGVzN16vv__mth_i_vr4vi8 (vrs16_t, vid8_t, vid8_t, float (*)(float, long long));
extern	vrs16_t __ZGVzM16vv__mth_i_vr4vi8 (vrs16_t, vid8_t, vid8_t, vis16_t, float (*)(float, long long));
extern	vrd8_t  __ZGVzN8v__mth_i_vr8si4   (vrd8_t, int32_t, double (*)(double, int32_t));
extern	vrd8_t  __ZGVzM8v__mth_i_vr8si4   (vrd8_t, int32_t, vid8_t, double (*)(double, int32_t));
extern	vrd8_t  __ZGVzN8vv__mth_i_vr8vi4  (vrd8_t, vis8_t, double (*)(double, int32_t));
extern	vrd8_t  __ZGVzM8vv__mth_i_vr8vi4  (vrd8_t, vis8_t, vid8_t, double (*)(double, int32_t));
extern	vrd8_t  __ZGVzN8v__mth_i_vr8si8   (vrd8_t, long long, double (*)(double, long long));
extern	vrd8_t  __ZGVzM8v__mth_i_vr8si8   (vrd8_t, long long, vid8_t, double (*)(double, long long));
extern	vrd8_t  __ZGVzN8vv__mth_i_vr8vi8  (vrd8_t, vid8_t, double (*)(double, long long));
extern	vrd8_t  __ZGVzM8vv__mth_i_vr8vi8  (vrd8_t, vid8_t, vid8_t, double (*)(double, long long));
extern	vcs1_t	__ZGVxN1v__mth_i_vc4si4   (vcs1_t, int32_t, _Complex float func(_Complex float, int32_t));
extern	vcs1_t	__ZGVxN1v__mth_i_vc4si8   (vcs1_t, long long, _Complex float func(_Complex float, long long));
extern	vcd1_t	__ZGVxN1v__mth_i_vc8si4   (vcd1_t, int32_t, _Complex double func(_Complex double, int32_t));
extern	vcd1_t	__ZGVxN1v__mth_i_vc8si8   (vcd1_t, long long, _Complex double func(_Complex double, long long));

#ifdef __cplusplus
} /* extern "C" */
#endif

#include "mthdecls.h"
