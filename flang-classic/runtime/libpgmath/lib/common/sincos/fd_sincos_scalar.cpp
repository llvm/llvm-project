
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


#include <assert.h>
#include <stdio.h>
#include <math.h>
#if     defined(__x86_64__)
#include <immintrin.h>
#endif


#define SINCOS_COMMA
#if     defined(SINE) && !(defined(COSINE) || defined(SINCOS))
#define S(...) __VA_ARGS__
#define C(...)
#define FNAME   sin
#elif   defined(COSINE) && !(defined(SINE) || defined(SINCOS))
#define S(...)
#define C(...) __VA_ARGS__
#define FNAME   cos
#elif   defined(SINCOS) && !(defined(SINE) || defined(COSINE))
#define S(...) __VA_ARGS__
#define C(...) __VA_ARGS__
#define FNAME   sincos
#include <complex.h>
#undef  SINCOS_COMMA
#define SINCOS_COMMA    ,
#else
#error  One of SINE, COSINE, or SINCOS must be defined.
#endif

#define _CONCAT(l,r) l##r
#define CONCAT(l,r) _CONCAT(l,r)

#define FCN_NAME    CONCAT(CONCAT(__fd_,FNAME),_1_avx2)

#include "sincos.h"

extern	"C" double
#if     defined(SINCOS)
    _Complex
#endif
FCN_NAME(double const x);

double
#if     defined(SINCOS)
_Complex
#endif
__attribute__((noinline)) FCN_NAME(double const x)
{
    S(double as, ks, rs;)
    S(uint64_t hs;)
    S(double ss, fs, ts;)

    C(double ac, kc, rc;)
    C(uint64_t hc;)
    C(double sc, fc, tc;)

    double t;
    uint64_t p;

    p = double_as_ll(x) & 0x7fffffffffffffffULL;

    if (__builtin_expect(p > double_as_ll(THRESHOLD), 0)) {
//        a = p >= 0x7ff0000000000000ULL ? x * 0.0 : reduction_slowpath(x, &h);
        if (p >= 0x7ff0000000000000ULL) {
            S(as = x * 0.0);
            C(ac = x * 0.0);
        } else {
            reduction_slowpath(x, S(&as, &hs) SINCOS_COMMA C(&ac, &hc));
        }
    } else {
        S(ks = FMA(x, _1_OVER_PI, 6755399441055744.0);)
        S(hs = double_as_ll(ks) << 63;)
        S(ks -= 6755399441055744.0;)

        C(kc = FMA(ll_as_double(p), _1_OVER_PI, -0.5);)
        C(kc += 6755399441055744.0;)
        C(hc = double_as_ll(kc) << 63;)
        C(kc -= 6755399441055744.0;)
        C(kc += 0.5;)

        S(as = FMA(ks, -PI_HI, x);)
        C(ac = FMA(kc, -PI_HI, ll_as_double(p));)

        S(as = FMA(ks, -PI_MI, as);)
        C(ac = FMA(kc, -PI_MI, ac);)

        S(as = FMA(ks, -PI_LO, as);)
        C(ac = FMA(kc, -PI_LO, ac);)
    }

#if     1 && defined(SINCOS) && defined (__x86_64__) && defined(__AVX2__)
    {
    const __m128d ama = _mm_set_pd(-A_D, A_D);     
    const __m128d bmb = _mm_set_pd(-B_D, B_D);
    const __m128d cmc = _mm_set_pd(-C_D, C_D);
    const __m128d dmd = _mm_set_pd(-D_D, D_D);
    const __m128d eme = _mm_set_pd(-E_D, E_D);
    const __m128d fmf = _mm_set_pd(-F_D, F_D);
    const __m128d gmg = _mm_set_pd(-G_D, G_D);
    const __m128d hmh = _mm_set_pd(-H_D, H_D);
    const __m128d omo = _mm_set_pd(-1.0, 1.0);

    __m128d va, vf, vs, vr, vt;
    va = _mm_unpacklo_pd(_mm_set1_pd(as), _mm_set1_pd(ac));
    vs = va * va;
    vr = ama;
    vr = _mm_fmadd_pd(vr, vs, bmb);
    vr = _mm_fmadd_pd(vr, vs, cmc);
    vr = _mm_fmadd_pd(vr, vs, dmd);
    vr = _mm_fmadd_pd(vr, vs, eme);
    vr = _mm_fmadd_pd(vr, vs, fmf);
    vr = _mm_fmadd_pd(vr, vs, gmg);
    vr = _mm_fmadd_pd(vr, vs, hmh);
    vf = _mm_unpacklo_pd(_mm_castsi128_pd(_mm_set1_epi64x(hs)), _mm_castsi128_pd(_mm_set1_epi64x(hc)));

    vf = _mm_xor_pd(va, vf);
    vt = _mm_fmadd_pd(vs, vf, _mm_set1_pd(0.0));
    vf = _mm_mul_pd(vf, omo);
    vr = _mm_fmadd_pd(vr, vt, vf);

    rs = _mm_cvtsd_f64(vr);
    rc = _mm_cvtsd_f64(_mm_permute_pd(vr, 1));
    }
#else
    S(ss = as * as;)
    C(sc = ac * ac;)

    S(rs = A_D;)
    C(rc = -A_D;)

    S(rs = FMA(rs, ss, B_D);)
    C(rc = FMA(rc, sc, -B_D);)

    S(rs = FMA(rs, ss, C_D);)
    C(rc = FMA(rc, sc, -C_D);)

    S(rs = FMA(rs, ss, D_D);)
    C(rc = FMA(rc, sc, -D_D);)

    S(rs = FMA(rs, ss, E_D);)
    C(rc = FMA(rc, sc, -E_D);)

    S(rs = FMA(rs, ss, F_D);)
    C(rc = FMA(rc, sc, -F_D);)

    S(rs = FMA(rs, ss, G_D);)
    C(rc = FMA(rc, sc, -G_D);)

    S(rs = FMA(rs, ss, H_D);)
    C(rc = FMA(rc, sc, -H_D);)

    S(fs = ll_as_double(double_as_ll(as) ^ hs);)
    C(fc = ll_as_double(double_as_ll(ac) ^ hc);)

    S(ts = FMA(ss, fs, 0.0);)
    C(tc = FMA(sc, fc, 0.0);)

    S(rs = FMA(rs, ts, fs);)
    C(rc = FMA(rc, tc, -fc);)
#endif


#if     defined(SINCOS)
    struct {
        union {
            double _Complex c;
            double          d[2];
        };
    } ret_cmplx;
    ret_cmplx.d[0] = rs;
    ret_cmplx.d[1] = rc;
        
    return ret_cmplx.c;
#else
    S(return rs;)
    C(return rc;)
#endif
}


#ifdef  UNIT_TEST
int
main()
{
    //float a = 40000+M_PI/6;
    //float a = -40000-M_PI/6;
    double a = -M_PI;
    double args[] = {
                    -M_PI+(-M_PI/6),//, -M_PI/6,
                    -0.0,
                    -(THRESHOLD*2)-M_PI/6, (THRESHOLD*2)+M_PI/6,
                    -1, 0, 1,
                    -M_PI*100, M_PI*100,
                    (-M_PI+(M_PI/6))*100.0,
                    -M_PI-(M_PI/6), M_PI+(M_PI/6),
                    -M_PI/6, M_PI/6,
                    0.1250000,
    };
    double rs;
    double rc;
//printf("THRESHOLD=%f\n", THRESHOLD);
#ifdef  SINCOS
    double _Complex ri;

    for (int i = 0 ; i < sizeof args / sizeof *args; ++i) {
    a = args[i];
    printf("%f\n", a);
    ri = FCN_NAME(a);
    printf("sincos:sin=%f %f %f\n", creal(ri), sin(a), creal(ri)-sin(a));
    printf("sincos:cos=%f %f %f\n", cimag(ri), cos(a), cimag(ri)-cos(a));
    }
#else
    for (int i = 0 ; i < sizeof args / sizeof *args; ++i) {
    a = args[i];
    printf("%f\n", a);
    S(rs = FCN_NAME(a);)
    C(rc = FCN_NAME(a);)
    S(printf("sin=%f %f %f\n", rs, sin(a), rs-sin(a));)
    C(printf("cos=%f %f %f\n", rc, cos(a), rc-cos(a));)
    }

#endif

    return 0;
}
#endif
// vim: ts=4 expandtab

