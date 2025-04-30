
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
#if     VL == 2
#include "helperavx2_128.h"
#elif   VL == 4
#include "helperavx2.h"
#elif   VL == 8
#include "helperavx512f.h"
#else
#error  VL must be 2, 4, or 8
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


#if     VL != 8
#define FCN_NAME    CONCAT(CONCAT4(__fd_,FNAME,_,VL),_avx2)
#else
#define FCN_NAME    CONCAT(CONCAT4(__fd_,FNAME,_,VL),FCN_AVX512())
#endif

#include "sincos.h"

#if     defined(SINCOS)
extern  "C" vdouble __mth_return2vectors(const vdouble, const vdouble);
#endif

extern	"C" vdouble FCN_NAME(const vdouble x);

static void INLINE
__reduction_slowpath(vdouble const a,
S(vdouble *vrs, vmask *vhs) SINCOS_COMMA C(vdouble *vrc, vmask *vhc))
{
    int ivl = sizeof a / sizeof(a[0]);
    union {
        vdouble vd;
        vmask vm;
        double sd[sizeof(vdouble) / sizeof(double)];
        uint64_t sm[sizeof(vmask) / sizeof(uint64_t)];
    } S(rs, hs) SINCOS_COMMA C(rc, hc);

    for (int i = 0; i < ivl; ++i) {
        reduction_slowpath(a[i], S(&rs.sd[i], &hs.sm[i]) SINCOS_COMMA
                                 C(&rc.sd[i], &hc.sm[i]));
    }
    S(*vrs = rs.vd;)
    S(*vhs = hs.vm;)
    C(*vrc = rc.vd;)
    C(*vhc = hc.vm;)
}


vdouble __attribute__((noinline)) FCN_NAME(const vdouble x)
{

    S(vdouble as, ks, rs;)
    S(vint2 hs;)
    S(vdouble ss, fs, ts;)
    C(vdouble ac, kc, rc;)
    C(vint2 hc;)
    C(vdouble sc, fc, tc;)
    vint2 p;

    p = vand_vi2_vi2_vi2((vint2)x, (vint2)vcast_vm_i_i(0x7fffffff, 0xffffffff));
    S(ks = vfma_vd_vd_vd_vd(x, vcast_vd_d(_1_OVER_PI), vcast_vd_d(6755399441055744.0));)
    S(hs = vsll64_vi2_vi2_i((vint2)ks, 63);)
    S(ks = vsub_vd_vd_vd(ks, vcast_vd_d(6755399441055744.0));)

    C(kc = vfma_vd_vd_vd_vd(x, vcast_vd_d(_1_OVER_PI), vcast_vd_d(-0.5));)
    C(kc = vadd_vd_vd_vd(kc, vcast_vd_d(6755399441055744.0));)
    C(hc = vsll64_vi2_vi2_i((vint2)kc, 63);)
    C(kc = vsub_vd_vd_vd(kc, vcast_vd_d(6755399441055744.0));)
    C(kc = vadd_vd_vd_vd(kc, vcast_vd_d(0.5));)

    S(as = vfma_vd_vd_vd_vd(ks, vcast_vd_d(-PI_HI), x);)
    C(ac = vfma_vd_vd_vd_vd(kc, vcast_vd_d(-PI_HI), x);)

    S(as = vfma_vd_vd_vd_vd(ks, vcast_vd_d(-PI_MI), as);)
    C(ac = vfma_vd_vd_vd_vd(kc, vcast_vd_d(-PI_MI), ac);)

    S(as = vfma_vd_vd_vd_vd(ks, vcast_vd_d(-PI_LO), as);)
    C(ac = vfma_vd_vd_vd_vd(kc, vcast_vd_d(-PI_LO), ac);)

    vopmask mask_lrg_args = vgt64_vo_vi2_vi2(p, (vint2)vcast_vd_d(THRESHOLD));
    if (__builtin_expect(!vtestz_i_vo(mask_lrg_args), 0)) {
        S(vdouble spas;)
        S(vmask sphs;)
        C(vdouble spac;)
        C(vmask sphc;)
        __reduction_slowpath(x, S(&spas, &sphs) SINCOS_COMMA C(&spac, &sphc));

        S(as = vsel_vd_vo_vd_vd(mask_lrg_args, spas, as);)
        C(ac = vsel_vd_vo_vd_vd(mask_lrg_args, spac, ac);)
        S(hs = (vmask)vsel_vi2_vo_vi2_vi2(mask_lrg_args, (vint2)sphs, (vint2)hs);)
        C(hc = (vmask)vsel_vi2_vo_vi2_vi2(mask_lrg_args, (vint2)sphc, (vint2)hc);)
    }

    S(ss = vmul_vd_vd_vd(as, as);)
    C(sc = vmul_vd_vd_vd(ac, ac);)

    S(rs = vfma_vd_vd_vd_vd(vcast_vd_d(A_D), ss, vcast_vd_d(B_D));)
    C(rc = vfmapn_vd_vd_vd_vd(vcast_vd_d(-A_D), sc, vcast_vd_d(B_D));)

    S(rs = vfma_vd_vd_vd_vd(rs, ss, vcast_vd_d(C_D));)
    C(rc = vfmapn_vd_vd_vd_vd(rc, sc, vcast_vd_d(C_D));)

    S(rs = vfma_vd_vd_vd_vd(rs, ss, vcast_vd_d(D_D));)
    C(rc = vfmapn_vd_vd_vd_vd(rc, sc, vcast_vd_d(D_D));)

    S(rs = vfma_vd_vd_vd_vd(rs, ss, vcast_vd_d(E_D));)
    C(rc = vfmapn_vd_vd_vd_vd(rc, sc, vcast_vd_d(E_D));)

    S(rs = vfma_vd_vd_vd_vd(rs, ss, vcast_vd_d(F_D));)
    C(rc = vfmapn_vd_vd_vd_vd(rc, sc, vcast_vd_d(F_D));)

    S(rs = vfma_vd_vd_vd_vd(rs, ss, vcast_vd_d(G_D));)
    C(rc = vfmapn_vd_vd_vd_vd(rc, sc, vcast_vd_d(G_D));)

    S(rs = vfma_vd_vd_vd_vd(rs, ss, vcast_vd_d(H_D));)
    C(rc = vfmapn_vd_vd_vd_vd(rc, sc, vcast_vd_d(H_D));)

    S(fs = (vdouble)vxor_vi2_vi2_vi2((vint2)as, hs);)
    C(fc = (vdouble)vxor_vi2_vi2_vi2((vint2)ac, hc);)

    S(ts = vfma_vd_vd_vd_vd(ss, fs, vcast_vd_d(0.0));)
    C(tc = vfma_vd_vd_vd_vd(sc, fc, vcast_vd_d(0.0));)

    S(rs = vfma_vd_vd_vd_vd(rs, ts, fs);)
    C(rc = vfmapn_vd_vd_vd_vd(rc, tc, fc);)


    C(vopmask m0 = vgt64_vo_vi2_vi2(p, (vint2)vcast_vm_i_i(0x3e46a09e, 0x667f3bcc));)
    C(rc = vsel_vd_vo_vd_vd(m0, rc, vcast_vd_d(1.0));)

    vopmask ninf = vgt64_vo_vi2_vi2((vint2)vcast_vm_i_i(0x7ff00000, 0), p);
    S(rs = vsel_vd_vo_vd_vd(ninf, rs, vmul_vd_vd_vd(x, vcast_vd_d(0.0)));)
    C(rc = vsel_vd_vo_vd_vd(ninf, rc, vmul_vd_vd_vd(x, vcast_vd_d(0.0)));)

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
#if   VL == 2
  //vdouble a = {-M_PI+(-M_PI/6), -M_PI/6, M_PI/6, M_PI-(M_PI/6)};
  //vdouble a = {-M_PI+(-M_PI/6), -M_PI/6};
  //vdouble a = {10*(-M_PI+(-M_PI/6)), 10*(-M_PI/6), 10*(M_PI/6), 10*(M_PI-(M_PI/6))};
  //vdouble a = {11*(-M_PI+(-M_PI/6)), 11*(-M_PI/6)};
  //vdouble a = {11*(M_PI/6), 11*(M_PI-(M_PI/6))};
  vdouble a = {0.1250000, 0.1250000*2.5};
  //vdouble a = {-M_PI+(M_PI/3), -M_PI/3, M_PI/3, M_PI-(M_PI/3)};
  //vdouble a = {-M_PI+(M_PI/3), -M_PI/4};
  //vdouble a = {-40000-M_PI+(M_PI/3), -40000-M_PI/3, 40000+M_PI/3, 40000+M_PI-(M_PI/3)};
  //vdouble a = {-40000-M_PI+(M_PI/3), -INFINITY, 40000+M_PI/3, 40000+M_PI-(M_PI/3)};
  //vdouble a = {-40000-M_PI+(M_PI/3), -INFINITY};
  //vdouble a = {.1, .2};
  //vdouble a = {THRESHOLD*2+M_PI/6, -THRESHOLD*2-M_PI/4};
  //vdouble a = {THRESHOLD*100000+M_PI/6, -THRESHOLD*2-M_PI/4};
  //vdouble a = {THRESHOLD*100000+M_PI/6, INFINITY};
  //vdouble a = {M_PI/6, 0.00000001053671213};
  //vdouble a = {M_PI/6, 0.00000000000671213};
  vdouble r;
//  uint32_t inf32 = 0x7f800000;
//  a[1] = *(float *)&inf32;
  uint64_t someval = 0x3e46a09e667f3bcc;
  printf("someval=%20.17f\n", *(double *)&someval);
  #ifdef    SINCOS
  vdouble    rs, rc;
   rs = FCN_NAME(a);
  asm(
    //"vmovups  %%xmm0,%0\n"
    "vmovups  %%xmm1,%1\n"
    : "=m"(rs), "=m"(rc) : :);
  #else
  S(r = FCN_NAME(a);)
  C(r = FCN_NAME(a);)
  #endif
  printf("%f %f\n", a[0], a[1]);
  S(printf("ref(sin)\t%f %f\n", sin(a[0]), sin(a[1]));)
  C(printf("ref(cos)\t%f %f\n", cos(a[0]), cos(a[1]));)
  #ifdef    SINCOS
  printf("new(sin)\t%f %f %f %f\n", rs[0], rs[1], rs[0]-sin(a[0]), rs[1]-sin(a[1]));
  printf("new(cos)\t%f %f %f %f\n", rc[0], rc[1], rc[0]-cos(a[0]), rc[1]-cos(a[1]));
  #else
  printf("%f %f\n", r[0], r[1]);
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
