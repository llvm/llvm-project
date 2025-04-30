
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#if     ! defined(VL)
#error  VL must be specified
#endif

#define	CONFIG	1
#if     VL == 4
#include "helperavx2_128.h"
#elif   VL == 8
#include "helperavx2.h"
#elif   VL == 16
#include "helperavx512f.h"
#else
#error  VL must be 4, 8, or 16
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
#define _CONCAT4(ll,l,r,rr) ll##l##r##rr
#define CONCAT4(ll,l,r,rr) _CONCAT4(ll,l,r,rr)


#if     VL != 16
#define FCN_NAME    CONCAT(CONCAT4(__fs_,FNAME,_,VL),_avx2)
#else
#define FCN_NAME    CONCAT(CONCAT4(__fs_,FNAME,_,VL),FCN_AVX512())
#endif

#include "sincosf.h"

#if     defined(SINCOS)
extern  "C" vfloat __mth_return2vectors(const vfloat, const vfloat);
#endif

extern	"C" vfloat FCN_NAME(const vfloat x);

void static INLINE
__reduction_slowpath(vfloat const a,
    S(vfloat *sr, vmask *sh) SINCOS_COMMA C(vfloat *cr, vmask *ch) )
{
    vint2 ia, e, idx, q, p;
    S(vint2 s;)
    vint2 ia_a, ia_b, p_a, p_b, hi_a, hi_b;
    vint2 hi, lo, ll, prev, prev2;

    vmask i1opi_vec[] = {
        vcast_vm_i_i(0, i1opi_f[0]),
        vcast_vm_i_i(0, i1opi_f[1]),
        vcast_vm_i_i(0, i1opi_f[2]),
        vcast_vm_i_i(0, i1opi_f[3]),
        vcast_vm_i_i(0, i1opi_f[4]),
        vcast_vm_i_i(0, i1opi_f[5]),
    };

    ia = (vint2)a;
    S(s = vand_vi2_vi2_vi2(ia, vcast_vi2_i(0x80000000));)
    /* e = ((ia >> 23) & 0xff) - 127; */
    e = vsrl_vi2_vi2_i(ia, 23);
    e = vand_vi2_vi2_vi2(e, vcast_vi2_i(0xff));
    e = vsub_vi2_vi2_vi2(e, vcast_vi2_i(127));
    /* ia = (ia << 8) | 0x80000000; */
    ia = vsll_vi2_vi2_i(ia, 8);
    ia = vor_vi2_vi2_vi2(ia, vcast_vi2_i(0x80000000));

    /* compute x * 1/pi */
    /* idx = 6 - ((e >> 5) & 3); */
    idx = vsrl_vi2_vi2_i(e, 5);
    idx = vand_vi2_vi2_vi2(idx, vcast_vi2_i(3));
    idx = vsub_vi2_vi2_vi2(vcast_vi2_i(6), idx);

    ia_a = vsrl64_vi2_vi2_i(ia, 32);
    ia_b = ia;
    hi_a = vcast_vi2_i(0);
    hi_b = vcast_vi2_i(0);

    q = vcast_vi2_i(0);
    for (int i = 0; i < 6; i++) {
        p_a = vmulu_vi2_vi2_vi2((vint2)i1opi_vec[i], ia_a);
        p_b = vmulu_vi2_vi2_vi2((vint2)i1opi_vec[i], ia_b);
        p_a = vadd64_vi2_vi2_vi2(p_a, hi_a);
        p_b = vadd64_vi2_vi2_vi2(p_b, hi_b);

        hi_a = vsrl64_vi2_vi2_i(p_a, 32);
        hi_b = vsrl64_vi2_vi2_i(p_b, 32);

        p_a = vsll64_vi2_vi2_i(p_a, 32);
        p_b = vand_vi2_vi2_vi2(p_b, vcast_vm_i_i(0, 0xffffffff));

        p = vor_vi2_vi2_vi2(p_a, p_b);

        vopmask m = veq_vo_vi2_vi2(idx, q);
        hi = vsel_vi2_vo_vi2_vi2(m, p, hi);
        lo = vsel_vi2_vo_vi2_vi2(m, prev, lo);
        ll = vsel_vi2_vo_vi2_vi2(m, prev2, ll);

        prev2 = prev;
        prev = p;

        q = vadd_vi2_vi2_vi2(q, vcast_vi2_i(1));
    }
    p = vor_vi2_vi2_vi2(vsll64_vi2_vi2_i(hi_a, 32), hi_b);

    vopmask m = veq_vo_vi2_vi2(idx, q);
    hi = vsel_vi2_vo_vi2_vi2(m, p, hi);
    lo = vsel_vi2_vo_vi2_vi2(m, prev, lo);
    ll = vsel_vi2_vo_vi2_vi2(m, prev2, ll);

    e = vand_vi2_vi2_vi2(e, vcast_vi2_i(31));

    union {
        vint2 v;
        uint32_t t[sizeof(vint2) / sizeof(uint32_t)];
    } ue, uhi, ulo, ull, us,
    S(suh, sur, sus) SINCOS_COMMA C(cuh, cur)
    ;

    ue.v = e; uhi.v = hi; ulo.v = lo; ull.v = ll;
    S(sus.v = s;)

    for (unsigned i = 0; i < sizeof(vint2) / sizeof(uint32_t); i++) {
        uint32_t e = ue.t[i];
        S(uint32_t sq;)
        S(uint64_t sp = ((uint64_t)uhi.t[i] << 32) | ulo.t[i];)
        C(uint32_t cq;)
        C(uint64_t cp = ((uint64_t)uhi.t[i] << 32) | ulo.t[i];)

        if (e) {
            S(sq = 32 - e;)
            S(sp = (sp << e) | (ull.t[i] >> sq);)
            C(cq = 32 - e;)
            C(cp = (cp << e) | (ull.t[i] >> cq);)
        }

        S(
            sq = (uhi.t[i] << e) & 0x80000000;
            sp &= 0x7fffffffffffffffULL;

            if (sp & 0x4000000000000000ULL) {
                sp |= 0x8000000000000000ULL;
                sq ^= 0x80000000;
            }
            suh.t[i] = sq ^ sus.t[i];
        )

        C(
            cuh.t[i] = (uhi.t[i] << e) & 0x80000000;
            cp &= 0x7fffffffffffffffULL;
            cp = (int64_t)cp - 0x4000000000000000LL;
        )

        S(
            double sd = (double)(int64_t)sp;
            sd *= PI_2_M63;
            float sr = (float)sd;
            sur.t[i] = float_as_int(sr);
        )

        C(
            double cd = (double)(int64_t)cp;
            cd *= PI_2_M63;
            float cr = (float)cd;
            cur.t[i] = float_as_int(cr);
        )

    }

    S(vstore_v_p_vf((float*)sh, (vfloat)suh.v);)
    S(vstore_v_p_vf((float*)sr, (vfloat)sur.v);)
    C(vstore_v_p_vf((float*)ch, (vfloat)cuh.v);)
    C(vstore_v_p_vf((float*)cr, (vfloat)cur.v);)
}


vfloat __attribute__((noinline)) FCN_NAME(const vfloat x)
{
    vfloat xabs;
    vfloat vt;              // General temp register
    S(vfloat as, ks, rs;)
    S(vint2 hs;)
    S(vfloat ss, fs, ts;)

    C(vfloat ac, kc, rc;)
    C(vint2 hc;)
    C(vfloat sc, fc, tc;)

    xabs = vabs_vf_vf(x);


/*
 *  Reduce x for sine.
 */
    S(ks = vfma_vf_vf_vf_vf(x, vcast_vf_f(_1_OVER_PI_F), vcast_vf_f(12582912.0f));)
    S(hs = vsll_vi2_vi2_i((vint2)ks, 31);)
    S(ks = vsub_vf_vf_vf(ks, vcast_vf_f(12582912.0f));)


/*
 *  Reduce x for cosine.
 */
    C(kc = vfma_vf_vf_vf_vf(xabs, vcast_vf_f(_1_OVER_PI_F), vcast_vf_f(-0.5f));)
    C(kc = vadd_vf_vf_vf(kc, vcast_vf_f(12582912.0f));)
    C(hc = vsll_vi2_vi2_i((vint2)kc, 31);)
    C(kc = vsub_vf_vf_vf(kc, vcast_vf_f(12582912.0f));)
    //C(kc = vfma_vf_vf_vf_vf(vcast_vf_f(2.0f), kc, vcast_vf_f(1.0f));)
    C(kc = vadd_vf_vf_vf(kc, vcast_vf_f(0.5f));)


    S(as = vfma_vf_vf_vf_vf(ks, vcast_vf_f(-PI_HI_F), x);)
    C(ac = vfma_vf_vf_vf_vf(kc, vcast_vf_f(-PI_HI_F), xabs);)

    S(as = vfma_vf_vf_vf_vf(ks, vcast_vf_f(-PI_MI_F), as);)
    C(ac = vfma_vf_vf_vf_vf(kc, vcast_vf_f(-PI_MI_F), ac);)

    S(as = vfma_vf_vf_vf_vf(ks, vcast_vf_f(-PI_LO_F), as);)
    C(ac = vfma_vf_vf_vf_vf(kc, vcast_vf_f(-PI_LO_F), ac);)

    vopmask m_lrg_args = vgt_vo_vi2_vi2((vint2)xabs, (vint2)vcast_vf_f(THRESHOLD_F));
    if (__builtin_expect(!vtestz_i_vo(m_lrg_args), 0)) {
    /*
     * Handle large arguments.
     */
        S(vfloat spas;)
        S(vint2 sphs;)
        C(vfloat spac;)
        C(vint2 sphc;)

        __reduction_slowpath(x,
            S(&spas, &sphs) SINCOS_COMMA C(&spac, &sphc));
        S(as = vsel_vf_vo_vf_vf(m_lrg_args, spas, as);)
        C(ac = vsel_vf_vo_vf_vf(m_lrg_args, spac, ac);)
        S(hs = (vmask)vsel_vi2_vo_vi2_vi2(m_lrg_args, (vint2)sphs, (vint2)hs);)
        C(hc = (vmask)vsel_vi2_vo_vi2_vi2(m_lrg_args, (vint2)sphc, (vint2)hc);)
    }

/*
 *
    rs = __sin_kernel(as, hs);
 */

    S(ss = vmul_vf_vf_vf(as, as);)
    C(sc = vmul_vf_vf_vf(ac, ac);)

    S(rs = vcast_vf_f(A_F);)
    C(rc = vcast_vf_f(-A_F);)

    S(rs = vfma_vf_vf_vf_vf(rs, ss, vcast_vf_f(B_F));)
    C(rc = vfmapn_vf_vf_vf_vf(rc, sc, vcast_vf_f(B_F));)

    S(rs = vfma_vf_vf_vf_vf(rs, ss, vcast_vf_f(C_F));)
    C(rc = vfmapn_vf_vf_vf_vf(rc, sc, vcast_vf_f(C_F));)

    S(rs = vfma_vf_vf_vf_vf(rs, ss, vcast_vf_f(D_F));)
    C(rc = vfmapn_vf_vf_vf_vf(rc, sc, vcast_vf_f(D_F));)

    S(rs = vfma_vf_vf_vf_vf(rs, ss, vcast_vf_f(E_F));)
    C(rc = vfmapn_vf_vf_vf_vf(rc, sc, vcast_vf_f(E_F));)

    S(fs = (vfloat)vxor_vi2_vi2_vi2((vint2)as, hs);)
    C(fc = (vfloat)vxor_vi2_vi2_vi2((vint2)ac, hc);)

    S(ts = vfma_vf_vf_vf_vf(ss, fs, vcast_vf_f(0.0));)
    C(tc = vfma_vf_vf_vf_vf(sc, fc, vcast_vf_f(0.0));)

    S(rs = vfma_vf_vf_vf_vf(rs, ts, fs);)
    C(rc = vfmapn_vf_vf_vf_vf(rc, tc, fc);)

    /*
     * Cosine args less than equal to 0x39800000 return 1.0.
     */
    C(vopmask m0 = vgt_vo_vi2_vi2((vint2)xabs, vcast_vi2_i(0x39800000));)
    C(rc = vsel_vf_vo_vf_vf(m0, rc, vcast_vf_f(1.0f));)

    if (__builtin_expect(vtestz_i_vo(m_lrg_args), 0)) { // No large args
    #if   defined (SINCOS)
        return  __mth_return2vectors(rs, rc);
    #else
        S(return rs;)
        C(return rc;)
    #endif
    }

    vopmask ninf = vgt_vo_vi2_vi2(vcast_vi2_i(0x7f800000), (vint2)xabs);
    S(rs = vsel_vf_vo_vf_vf(ninf, rs, vmul_vf_vf_vf(x, vcast_vf_f(0.0)));)
    C(rc = vsel_vf_vo_vf_vf(ninf, rc, vmul_vf_vf_vf(x, vcast_vf_f(0.0)));)

#if   defined (SINCOS)
    return  __mth_return2vectors(rs, rc);
#else
    S(return rs;)
    C(return rc;)
#endif
}

#ifdef	UNIT_TEST

int
main()
{
#if   VL == 4
  //vfloat a = {-M_PI+(-M_PI/6), -M_PI/6, M_PI/6, M_PI-(M_PI/6)};
  //vfloat a = {10*(-M_PI+(-M_PI/6)), 10*(-M_PI/6), 10*(M_PI/6), 10*(M_PI-(M_PI/6))};
  //vfloat a = {11*(-M_PI+(-M_PI/6)), 11*(-M_PI/6), 11*(M_PI/6), 11*(M_PI-(M_PI/6))};
  //vfloat a = {0.1250000, 0.1250000, 0.1250000, 0.1250000};
  //vfloat a = {-M_PI+(M_PI/3), -M_PI/3, M_PI/3, M_PI-(M_PI/3)};
  //vfloat a = {-40000-M_PI+(M_PI/3), -40000-M_PI/3, 40000+M_PI/3, 40000+M_PI-(M_PI/3)};
  //vfloat a = {-40000-M_PI+(M_PI/3), -INFINITY, 40000+M_PI/3, 40000+M_PI-(M_PI/3)};
  //vfloat a = {-.40023992E+05, 0.40840695E+05, -.40551832E+05, 0.40543305E+05};
  //bad vfloat a = {1, 2, NAN, 3.1415926};
  vfloat a = {1, NAN, 2, 3.1415926};
  vfloat r;
//  uint32_t inf32 = 0x7f800000;
//  a[1] = *(float *)&inf32;
  #ifdef    SINCOS
  vfloat    rs, rc;
   rs = FCN_NAME(a);
  asm(
    //"vmovups  %%xmm0,%0\n"
    "vmovups  %%xmm1,%1\n"
    : "=m"(rs), "=m"(rc) : :);
  #else
  S(r = FCN_NAME(a);)
  C(r = FCN_NAME(a);)
  #endif
  printf("%f %f %f %f\n", a[0], a[1], a[2], a[3]);
  S(printf("ref(sin)\t%f %f %f %f\n", sinf(a[0]), sinf(a[1]), sinf(a[2]), sinf(a[3]));)
  C(printf("ref(cos)\t%f %f %f %f\n", cosf(a[0]), cosf(a[1]), cosf(a[2]), cosf(a[3]));)
  #ifdef    SINCOS
  printf("new(sin)\t%f %f %f %f\n", rs[0], rs[1], rs[2], rs[3]);
  printf("new(cos)\t%f %f %f %f\n", rc[0], rc[1], rc[2], rc[3]);
  #else
  printf("%f %f %f %f\n", r[0], r[1], r[2], r[3]);
  #endif
#endif

  return 0;
}

asm ("\n\
    .text\n\
    .globl  ret2vrs\n\
    .type   ret2vrs,@function\n\
ret2vrs:\n\
    ret\n\
    .size   ret2vrs,.-ret2vrs\n\
    ");
#endif		// #ifdef UNIT_TEST
// vim: ts=4 expandtab
