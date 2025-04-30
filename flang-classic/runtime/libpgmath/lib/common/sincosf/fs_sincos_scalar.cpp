
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h>
#include "sincosf.h"


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
#undef  SINCOS_COMMA
#define SINCOS_COMMA    ,
#else
#error  One of SINE, COSINE, or SINCOS must be defined.
#endif

#define _CONCAT(l,r) l##r
#define CONCAT(l,r) _CONCAT(l,r)

#define FCN_NAME    CONCAT(CONCAT(__fs_,FNAME),_1_avx2)

extern	"C" float
FCN_NAME(const float x);



/* Payne-Hanek style argument reduction. */
static void __attribute__((noinline))
reduction_slowpath(float const a,
S(float *rs, int32_t *hs) SINCOS_COMMA C(float *rc, int32_t *hc))
{
    uint2 m;
    uint32_t ia = float_as_int(a);
    S(uint32_t ss = ia & 0x80000000;)
    uint32_t result[7];
    uint32_t hi, lo;
    uint32_t e;
    int32_t idx;
    int32_t q;
    e = ((ia >> 23) & 0xff) - 127;
    ia = (ia << 8) | 0x80000000;

    /* compute x * 1/pi */
    idx = 4 - ((e >> 5) & 3);

    hi = 0;
    for (q = 0; q < 6; q++) {
        m = umad32wide(i1opi_f[q], ia, hi);
        lo = m.x;
        hi = m.y;
        result[q] = lo;
    }
    result[q] = hi;

    e = e & 31;
    /* shift result such that hi:lo<63:63> is the least significant
       integer bit, and hi:lo<62:0> are the fractional bits of the result
    */

    uint64_t p = ((uint64_t)result[idx + 2] << 32) | result[idx + 1];

    if (e) {
        q = 32 - e;
        p = (p << e) | (result[idx] >> q);
    }

    p &= 0x7fffffffffffffffULL;

    S(
        uint64_t ps = p;
        int32_t qs = q;

        /* fraction */
        qs = (result[idx + 2] << e) & 0x80000000;

        if (ps & 0x4000000000000000ULL) {
            ps |= 0x8000000000000000ULL;
            qs ^= 0x80000000;
        }
        *hs = qs ^ ss;

        double ds = (double)(int64_t)ps;
        ds *= PI_2_M63;
        *rs = (float)ds;
    )


    C(
        uint64_t pc = p;
        /* fraction */
        *hc = (result[idx + 2] << e) & 0x80000000;
        /* subtract 0.5 */
        pc = (int64_t)pc - 0x4000000000000000LL;

        double dc = (double)(int64_t)pc;
        dc *= PI_2_M63;
        *rc = (float)dc;
    )
}

float
__attribute__((noinline)) FCN_NAME(const float x)
{

    float p;
    S(float ks, rs, ss, ts, xs;)
    S(int hs = 0;)
    C(float kc, rc, sc, tc, xc;)
    C(int hc = 0;)

    p = int_as_float(float_as_int(x) & 0x7fffffff);

#if     defined(COSINE)
    // Cosine only!  Don't use macro "C"!
    if (float_as_int(p) <= 0x39800000) {
        return 1.0f;
    }
#endif

    if (float_as_int(p) > float_as_int(THRESHOLD_F)) {
        if (float_as_int(p) >= 0x7f800000) {
            S(xs = x * 0.0f;)
            C(xc = x * 0.0f;)
        } else {
            reduction_slowpath(x, S(&xs, &hs) SINCOS_COMMA C(&xc, &hc));
        }
    } else {
        S(ks = FMAF(x, _1_OVER_PI_F, 12582912.0f);)
        S(hs = float_as_int(ks) << 31;)
        S(ks -= 12582912.0f;)

        C(kc = FMAF(p, _1_OVER_PI_F, -0.5f);)
        C(kc += 12582912.0f;)
        C(hc = float_as_int(kc) << 31;)
        C(kc -= 12582912.0f;)
        C(kc += 0.5;)

        S(xs = x;)
        C(xc = p;)
        S(xs = FMAF(ks, -PI_HI_F, xs);)
        C(xc = FMAF(kc, -PI_HI_F, xc);)
        S(xs = FMAF(ks, -PI_MI_F, xs);)
        C(xc = FMAF(kc, -PI_MI_F, xc);)
        S(xs = FMAF(ks, -PI_LO_F, xs);)
        C(xc = FMAF(kc, -PI_LO_F, xc);)
    }

#if     defined(SINCOS) && defined(__x86_64__) && defined(__AVX2__)
    {
    const __m128  ama = _mm_set_ps(0.0, 0.0, -A_F, A_F);
    const __m128  bmb = _mm_set_ps(0.0, 0.0, -B_F, B_F);
    const __m128  cmc = _mm_set_ps(0.0, 0.0, -C_F, C_F);
    const __m128  dmd = _mm_set_ps(0.0, 0.0, -D_F, D_F);
    const __m128  eme = _mm_set_ps(0.0, 0.0, -E_F, E_F);
    const __m128  omo = _mm_set_ps(0.0, 0.0, -1.0, 1.0);
    __m128 va, vf, vs, vr, vt;

    vr = ama;
    va = _mm_unpacklo_ps(_mm_set1_ps(xs), _mm_set1_ps(xc));
    vs = va * va;
    vr = _mm_fmadd_ps(vr, vs, bmb);
    vr = _mm_fmadd_ps(vr, vs, cmc);
    vr = _mm_fmadd_ps(vr, vs, dmd);
    vr = _mm_fmadd_ps(vr, vs, eme);
    vf = _mm_castsi128_ps(_mm_set_epi32(0, 0, hc, hs));

    vf = _mm_xor_ps(va, vf);
    vt = _mm_fmadd_ps(vs, vf, _mm_set1_ps(0.0));
    vf = _mm_mul_ps(vf, omo);
    vr = _mm_fmadd_ps(vr, vt, vf);

    rs = _mm_cvtss_f32(vr);
    rc = _mm_cvtss_f32(_mm_permute_ps(vr, 1));
    }
#else
    S(ss = xs * xs;)
    C(sc = xc * xc;)

    S(rs = A_F;)
    C(rc = -A_F;)

    S(rs = FMAF(rs, ss, B_F);)
    C(rc = FMAF(rc, sc, -B_F);)

    S(rs = FMAF(rs, ss, C_F);)
    C(rc = FMAF(rc, sc, -C_F);)

    S(rs = FMAF(rs, ss, D_F);)
    C(rc = FMAF(rc, sc, -D_F);)

    S(rs = FMAF(rs, ss, E_F);)
    C(rc = FMAF(rc, sc, -E_F);)

    S(xs = int_as_float(float_as_int(xs) ^ hs);)
    C(xc = int_as_float(float_as_int(xc) ^ hc);)

    S(ts = FMAF(ss, xs, 0.0);)
    C(tc = FMAF(sc, xc, 0.0);)

    S(rs = FMAF(rs, ts, xs);)
    C(rc = FMAF(rc, tc, -xc);)
#endif

#if     defined(SINCOS)
//    {
        /*
         * Probably not the best choice to use memory to get the proper value
         * for "rc" when p <= 0x39800000, but the above block (#if 0) causes
         * the sine/cosine FMAs and other operations to no longer be
         * interleaved when compiling on x86_64 with clang 5.0.1.
         *
         * BTW, using a mask and merging good values and 1.0 also breaks the
         * interleaved FMA operations.
         */
        bool    pgt0x39800000 = float_as_int(p) > 0x39800000;
        float   rc_ret_val[2] = {1.0, rc};  // DO NOT MAKE STATIC!
        rc = rc_ret_val[pgt0x39800000];

        // Variant 2b
        //float   zero_one[] = {0.0,1.0,0.0}; // false, true, true+1
        //rc = (rc * zero_one[pgt0x39800000]) + (1.0 * zero_one[1+pgt0x39800000]);

        // Variant 2c
        //uint32_t mask = 0 - pgt0x39800000;
        //uint32_t t = (mask & *(uint32_t *)&rc) | (~mask & 0x3f800000);
        //rc = *(float *)&t;
 //   }

    asm("vmovss\t%0,%%xmm1" : : "m"(rc) : "%xmm1");
    return rs;
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
    float a = -M_PI;
    float args[] = {
                    -0.0
    };
    float rs;
    float rc;
#ifdef  SINCOS
    float _Complex ri;

    for (int i = 0 ; i < sizeof args / sizeof *args; ++i) {
    a = args[i];
    printf("%f %#x\n", a, *(int *)&a);
    ri = FCN_NAME(a);
    printf("sincos:sin=%f %f %f\n", crealf(ri), sinf(a), crealf(ri)-sinf(a));
    printf("sincos:cos=%f %f %f\n", cimagf(ri), cosf(a), cimagf(ri)-cosf(a));
    }
#else
    for (int i = 0 ; i < sizeof args / sizeof *args; ++i) {
    a = args[i];
    printf("%f\n", a);
    S(rs = FCN_NAME(a);)
    C(rc = FCN_NAME(a);)
    S(printf("sin=%f %f %f\n", rs, sinf(a), rs-sinf(a));)
    C(printf("cos=%f %f %f\n", rc, cosf(a), rc-cosf(a));)
    }

#endif

    return 0;
}
#endif
// vim: ts=4 expandtab

