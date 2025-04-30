/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mth_intrinsics.h"
#include "mth_tbldefs.h"
#if     defined(TARGET_POWER)
#warning  TARGET_POWER currently disabled - see mth_128defs_pwr.S
#endif

/*
 * The X86 architecture is peculiar with regards to floating point comparisons.
 * When testing if a FP value is a particular constant, compilers first have
 * to test for whether or not the argument is NaN, thus incurring the
 * cost of two jumps. One for the NaN, and then the second for some form of
 * equality (==, <, >, ...).
 *
 * Given the following function as an example:
 * double f(x)
 * {
 *   if (0.0 == x)
 *     return (x);
 *   return (x*x);
 * }
 *
 * We have the following generated code:
 *         .p2align 4,,15
 *         .globl  f
 *         .type   f, @function 
 * f:
 * .LFB0:  
 *         .cfi_startproc
 *         ucomisd .LC0(%rip), %xmm0	<=== Compare x and constant 0.0
 *         movapd  %xmm0, %xmm1
 *         jp      .L5 		<======== If x is NaN, return x*x
 *         je      .L2		<======== If 0.0 == x, return x (%xmm1)
 * .L5:
 *         movapd  %xmm1, %xmm0 
 *         mulsd   %xmm1, %xmm0 
 * .L2:
 *         rep ret
 *         .cfi_endproc
 * .LFE0:
 *         .size   f, .-f 
 *         .section        .rodata.cst8,"aM",@progbits,8
 *         .align 8
 * .LC0:   
 *         .long   0
 *         .long   0
 *
 *
 * As an optimization in the scalar single and double precision EXP()
 * functions, we test to see if the argument is zero and if so short circuit
 * the logic and immediately return 1.0.
 *
 * But the cost of two jump tests is expensive.  We instead compare the
 * input argument as an integer looking for +- zero thus only having
 * a single jump in the instruction sequence.
 *
 * For X8664 platforms the macros _ISFZEROPT0(float) and _ISDZEROPT0(double)
 * do the appropriate recasting of the arguments to test against integer
 * 0. Other platforms, the floating point test against 0.0 remains.
 *
 */

#ifdef  LINUX8664
#define _ISFZEROPT0(_x) (0 == (*(int32_t *)&(_x)) << 1)
#define _ISDZEROPT0(_x) (0 == (*(int64_t *)&(_x)) << 1)
#else
#define _ISFZEROPT0(_x) (0.0 == (_x))
#define _ISDZEROPT0(_x) (0.0 == (_x))
#endif

vrs1_t
MTH_DISPATCH_FUNC(__fs_acos_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_ss,frp_f);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_acos][sv_ss][frp_f];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__rs_acos_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_ss,frp_r);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_acos][sv_ss][frp_r];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__ps_acos_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_ss,frp_p);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_acos][sv_ss][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_acos_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_sv4,frp_f);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_acos][sv_sv4][frp_f];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_acos_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_sv4,frp_r);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_acos][sv_sv4][frp_r];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_acos_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_sv4,frp_p);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_acos][sv_sv4][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_acos_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_sv4m,frp_f);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_acos][sv_sv4m][frp_f];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_acos_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_sv4m,frp_r);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_acos][sv_sv4m][frp_r];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_acos_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_sv4m,frp_p);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_acos][sv_sv4m][frp_p];
  return (fptr(x, m));
}

vrd1_t
MTH_DISPATCH_FUNC(__fd_acos_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_ds,frp_f);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_acos][sv_ds][frp_f];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__rd_acos_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_ds,frp_r);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_acos][sv_ds][frp_r];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__pd_acos_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_ds,frp_p);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_acos][sv_ds][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_acos_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_dv2,frp_f);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_acos][sv_dv2][frp_f];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_acos_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_dv2,frp_r);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_acos][sv_dv2][frp_r];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_acos_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_dv2,frp_p);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_acos][sv_dv2][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_acos_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_dv2m,frp_f);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_acos][sv_dv2m][frp_f];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_acos_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_dv2m,frp_r);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_acos][sv_dv2m][frp_r];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_acos_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_dv2m,frp_p);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_acos][sv_dv2m][frp_p];
  return (fptr(x, m));
}

#ifdef TARGET_SUPPORTS_QUADFP
vrq1_t
MTH_DISPATCH_FUNC(__fq_acos_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos, sv_qs, frp_f);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_acos][sv_qs][frp_f];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__rq_acos_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos, sv_qs, frp_r);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_acos][sv_qs][frp_r];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__pq_acos_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos, sv_qs, frp_p);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_acos][sv_qs][frp_p];
  return (fptr(x));
}
#endif

vrs1_t
MTH_DISPATCH_FUNC(__fs_asin_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_ss,frp_f);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_asin][sv_ss][frp_f];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__rs_asin_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_ss,frp_r);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_asin][sv_ss][frp_r];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__ps_asin_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_ss,frp_p);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_asin][sv_ss][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_asin_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_sv4,frp_f);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_asin][sv_sv4][frp_f];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_asin_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_sv4,frp_r);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_asin][sv_sv4][frp_r];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_asin_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_sv4,frp_p);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_asin][sv_sv4][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_asin_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_sv4m,frp_f);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_asin][sv_sv4m][frp_f];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_asin_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_sv4m,frp_r);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_asin][sv_sv4m][frp_r];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_asin_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_sv4m,frp_p);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_asin][sv_sv4m][frp_p];
  return (fptr(x, m));
}

vrd1_t
MTH_DISPATCH_FUNC(__fd_asin_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_ds,frp_f);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_asin][sv_ds][frp_f];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__rd_asin_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_ds,frp_r);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_asin][sv_ds][frp_r];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__pd_asin_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_ds,frp_p);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_asin][sv_ds][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_asin_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_dv2,frp_f);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_asin][sv_dv2][frp_f];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_asin_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_dv2,frp_r);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_asin][sv_dv2][frp_r];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_asin_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_dv2,frp_p);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_asin][sv_dv2][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_asin_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_dv2m,frp_f);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_asin][sv_dv2m][frp_f];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_asin_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_dv2m,frp_r);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_asin][sv_dv2m][frp_r];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_asin_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_dv2m,frp_p);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_asin][sv_dv2m][frp_p];
  return (fptr(x, m));
}

#ifdef TARGET_SUPPORTS_QUADFP
vrq1_t
MTH_DISPATCH_FUNC(__fq_asin_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin, sv_qs, frp_f);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_asin][sv_qs][frp_f];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__rq_asin_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin, sv_qs, frp_r);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_asin][sv_qs][frp_r];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__pq_asin_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin, sv_qs, frp_p);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_asin][sv_qs][frp_p];
  return (fptr(x));
}
#endif

vrs1_t
MTH_DISPATCH_FUNC(__fs_atan_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_ss,frp_f);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_atan][sv_ss][frp_f];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__rs_atan_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_ss,frp_r);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_atan][sv_ss][frp_r];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__ps_atan_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_ss,frp_p);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_atan][sv_ss][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_atan_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_sv4,frp_f);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_atan][sv_sv4][frp_f];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_atan_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_sv4,frp_r);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_atan][sv_sv4][frp_r];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_atan_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_sv4,frp_p);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_atan][sv_sv4][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_atan_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_sv4m,frp_f);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_atan][sv_sv4m][frp_f];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_atan_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_sv4m,frp_r);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_atan][sv_sv4m][frp_r];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_atan_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_sv4m,frp_p);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_atan][sv_sv4m][frp_p];
  return (fptr(x, m));
}

vrd1_t
MTH_DISPATCH_FUNC(__fd_atan_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_ds,frp_f);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_atan][sv_ds][frp_f];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__rd_atan_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_ds,frp_r);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_atan][sv_ds][frp_r];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__pd_atan_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_ds,frp_p);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_atan][sv_ds][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_atan_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_dv2,frp_f);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_atan][sv_dv2][frp_f];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_atan_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_dv2,frp_r);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_atan][sv_dv2][frp_r];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_atan_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_dv2,frp_p);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_atan][sv_dv2][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_atan_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_dv2m,frp_f);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_atan][sv_dv2m][frp_f];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_atan_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_dv2m,frp_r);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_atan][sv_dv2m][frp_r];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_atan_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_dv2m,frp_p);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_atan][sv_dv2m][frp_p];
  return (fptr(x, m));
}

#ifdef TARGET_SUPPORTS_QUADFP
vrq1_t
MTH_DISPATCH_FUNC(__fq_atan_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan, sv_qs, frp_f);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_atan][sv_qs][frp_f];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__rq_atan_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan, sv_qs, frp_r);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_atan][sv_qs][frp_r];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__pq_atan_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan, sv_qs, frp_p);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_atan][sv_qs][frp_p];
  return (fptr(x));
}
#endif

vrs1_t
MTH_DISPATCH_FUNC(__fs_atan2_1)(vrs1_t x, vrs1_t y)
{
  vrs1_t (*fptr)(vrs1_t, vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_ss,frp_f);
  fptr = (vrs1_t(*)(vrs1_t, vrs1_t))MTH_DISPATCH_TBL[func_atan2][sv_ss][frp_f];
  return (fptr(x, y));
}

vrs1_t
MTH_DISPATCH_FUNC(__rs_atan2_1)(vrs1_t x, vrs1_t y)
{
  vrs1_t (*fptr)(vrs1_t, vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_ss,frp_r);
  fptr = (vrs1_t(*)(vrs1_t, vrs1_t))MTH_DISPATCH_TBL[func_atan2][sv_ss][frp_r];
  return (fptr(x, y));
}

vrs1_t
MTH_DISPATCH_FUNC(__ps_atan2_1)(vrs1_t x, vrs1_t y)
{
  vrs1_t (*fptr)(vrs1_t, vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_ss,frp_p);
  fptr = (vrs1_t(*)(vrs1_t, vrs1_t))MTH_DISPATCH_TBL[func_atan2][sv_ss][frp_p];
  return (fptr(x, y));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_atan2_4)(vrs4_t x, vrs4_t y)
{
  vrs4_t (*fptr)(vrs4_t, vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_sv4,frp_f);
  fptr = (vrs4_t(*)(vrs4_t, vrs4_t))MTH_DISPATCH_TBL[func_atan2][sv_sv4][frp_f];
  return (fptr(x, y));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_atan2_4)(vrs4_t x, vrs4_t y)
{
  vrs4_t (*fptr)(vrs4_t, vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_sv4,frp_r);
  fptr = (vrs4_t(*)(vrs4_t, vrs4_t))MTH_DISPATCH_TBL[func_atan2][sv_sv4][frp_r];
  return (fptr(x, y));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_atan2_4)(vrs4_t x, vrs4_t y)
{
  vrs4_t (*fptr)(vrs4_t, vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_sv4,frp_p);
  fptr = (vrs4_t(*)(vrs4_t, vrs4_t))MTH_DISPATCH_TBL[func_atan2][sv_sv4][frp_p];
  return (fptr(x, y));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_atan2_4m)(vrs4_t x, vrs4_t y, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_sv4m,frp_f);
  fptr = (vrs4_t(*)(vrs4_t, vrs4_t, vis4_t))MTH_DISPATCH_TBL[func_atan2][sv_sv4m][frp_f];
  return (fptr(x, y, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_atan2_4m)(vrs4_t x, vrs4_t y, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_sv4m,frp_r);
  fptr = (vrs4_t(*)(vrs4_t, vrs4_t, vis4_t))MTH_DISPATCH_TBL[func_atan2][sv_sv4m][frp_r];
  return (fptr(x, y, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_atan2_4m)(vrs4_t x, vrs4_t y, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_sv4m,frp_p);
  fptr = (vrs4_t(*)(vrs4_t, vrs4_t, vis4_t))MTH_DISPATCH_TBL[func_atan2][sv_sv4m][frp_p];
  return (fptr(x, y, m));
}

vrd1_t
MTH_DISPATCH_FUNC(__fd_atan2_1)(vrd1_t x, vrd1_t y)
{
  vrd1_t (*fptr)(vrd1_t, vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_ds,frp_f);
  fptr = (vrd1_t(*)(vrd1_t, vrd1_t))MTH_DISPATCH_TBL[func_atan2][sv_ds][frp_f];
  return (fptr(x, y));
}

vrd1_t
MTH_DISPATCH_FUNC(__rd_atan2_1)(vrd1_t x, vrd1_t y)
{
  vrd1_t (*fptr)(vrd1_t, vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_ds,frp_r);
  fptr = (vrd1_t(*)(vrd1_t, vrd1_t))MTH_DISPATCH_TBL[func_atan2][sv_ds][frp_r];
  return (fptr(x, y));
}

vrd1_t
MTH_DISPATCH_FUNC(__pd_atan2_1)(vrd1_t x, vrd1_t y)
{
  vrd1_t (*fptr)(vrd1_t, vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_ds,frp_p);
  fptr = (vrd1_t(*)(vrd1_t, vrd1_t))MTH_DISPATCH_TBL[func_atan2][sv_ds][frp_p];
  return (fptr(x, y));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_atan2_2)(vrd2_t x, vrd2_t y)
{
  vrd2_t (*fptr)(vrd2_t, vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_dv2,frp_f);
  fptr = (vrd2_t(*)(vrd2_t, vrd2_t))MTH_DISPATCH_TBL[func_atan2][sv_dv2][frp_f];
  return (fptr(x, y));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_atan2_2)(vrd2_t x, vrd2_t y)
{
  vrd2_t (*fptr)(vrd2_t, vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_dv2,frp_r);
  fptr = (vrd2_t(*)(vrd2_t, vrd2_t))MTH_DISPATCH_TBL[func_atan2][sv_dv2][frp_r];
  return (fptr(x, y));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_atan2_2)(vrd2_t x, vrd2_t y)
{
  vrd2_t (*fptr)(vrd2_t, vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_dv2,frp_p);
  fptr = (vrd2_t(*)(vrd2_t, vrd2_t))MTH_DISPATCH_TBL[func_atan2][sv_dv2][frp_p];
  return (fptr(x, y));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_atan2_2m)(vrd2_t x, vrd2_t y, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_dv2m,frp_f);
  fptr = (vrd2_t(*)(vrd2_t, vrd2_t, vid2_t))MTH_DISPATCH_TBL[func_atan2][sv_dv2m][frp_f];
  return (fptr(x, y, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_atan2_2m)(vrd2_t x, vrd2_t y, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_dv2m,frp_r);
  fptr = (vrd2_t(*)(vrd2_t, vrd2_t, vid2_t))MTH_DISPATCH_TBL[func_atan2][sv_dv2m][frp_r];
  return (fptr(x, y, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_atan2_2m)(vrd2_t x, vrd2_t y, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_dv2m,frp_p);
  fptr = (vrd2_t(*)(vrd2_t, vrd2_t, vid2_t))MTH_DISPATCH_TBL[func_atan2][sv_dv2m][frp_p];
  return (fptr(x, y, m));
}

#ifdef TARGET_SUPPORTS_QUADFP
vrq1_t
MTH_DISPATCH_FUNC(__fq_atan2_1)(vrq1_t x, vrq1_t y)
{
  vrq1_t (*fptr)(vrq1_t, vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2, sv_qs, frp_f);
  fptr = (vrq1_t(*)(vrq1_t, vrq1_t))MTH_DISPATCH_TBL[func_atan2][sv_qs][frp_f];
  return (fptr(x, y));
}

vrq1_t
MTH_DISPATCH_FUNC(__rq_atan2_1)(vrq1_t x, vrq1_t y)
{
  vrq1_t (*fptr)(vrq1_t, vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2, sv_qs, frp_r);
  fptr = (vrq1_t(*)(vrq1_t, vrq1_t))MTH_DISPATCH_TBL[func_atan2][sv_qs][frp_r];
  return (fptr(x, y));
}

vrq1_t
MTH_DISPATCH_FUNC(__pq_atan2_1)(vrq1_t x, vrq1_t y)
{
  vrq1_t (*fptr)(vrq1_t, vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2, sv_qs, frp_p);
  fptr = (vrq1_t(*)(vrq1_t, vrq1_t))MTH_DISPATCH_TBL[func_atan2][sv_qs][frp_p];
  return (fptr(x, y));
}
#endif

vrs1_t
MTH_DISPATCH_FUNC(__fs_cos_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_ss,frp_f);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_cos][sv_ss][frp_f];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__rs_cos_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_ss,frp_r);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_cos][sv_ss][frp_r];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__ps_cos_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_ss,frp_p);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_cos][sv_ss][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_cos_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_sv4,frp_f);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_cos][sv_sv4][frp_f];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_cos_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_sv4,frp_r);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_cos][sv_sv4][frp_r];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_cos_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_sv4,frp_p);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_cos][sv_sv4][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_cos_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_sv4m,frp_f);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_cos][sv_sv4m][frp_f];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_cos_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_sv4m,frp_r);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_cos][sv_sv4m][frp_r];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_cos_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_sv4m,frp_p);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_cos][sv_sv4m][frp_p];
  return (fptr(x, m));
}

vrd1_t
MTH_DISPATCH_FUNC(__fd_cos_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_ds,frp_f);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_cos][sv_ds][frp_f];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__rd_cos_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_ds,frp_r);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_cos][sv_ds][frp_r];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__pd_cos_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_ds,frp_p);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_cos][sv_ds][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_cos_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_dv2,frp_f);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_cos][sv_dv2][frp_f];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_cos_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_dv2,frp_r);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_cos][sv_dv2][frp_r];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_cos_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_dv2,frp_p);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_cos][sv_dv2][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_cos_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_dv2m,frp_f);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_cos][sv_dv2m][frp_f];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_cos_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_dv2m,frp_r);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_cos][sv_dv2m][frp_r];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_cos_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_dv2m,frp_p);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_cos][sv_dv2m][frp_p];
  return (fptr(x, m));
}

#ifdef TARGET_SUPPORTS_QUADFP
vrq1_t
MTH_DISPATCH_FUNC(__fq_cos_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos, sv_qs, frp_f);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_cos][sv_qs][frp_f];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__rq_cos_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos, sv_qs, frp_r);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_cos][sv_qs][frp_r];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__pq_cos_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos, sv_qs, frp_p);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_cos][sv_qs][frp_p];
  return (fptr(x));
}
#endif

vrs1_t
MTH_DISPATCH_FUNC(__fs_sin_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_ss,frp_f);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_sin][sv_ss][frp_f];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__rs_sin_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_ss,frp_r);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_sin][sv_ss][frp_r];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__ps_sin_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_ss,frp_p);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_sin][sv_ss][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_sin_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_sv4,frp_f);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_sin][sv_sv4][frp_f];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_sin_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_sv4,frp_r);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_sin][sv_sv4][frp_r];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_sin_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_sv4,frp_p);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_sin][sv_sv4][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_sin_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_sv4m,frp_f);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_sin][sv_sv4m][frp_f];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_sin_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_sv4m,frp_r);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_sin][sv_sv4m][frp_r];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_sin_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_sv4m,frp_p);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_sin][sv_sv4m][frp_p];
  return (fptr(x, m));
}

vrd1_t
MTH_DISPATCH_FUNC(__fd_sin_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_ds,frp_f);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_sin][sv_ds][frp_f];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__rd_sin_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_ds,frp_r);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_sin][sv_ds][frp_r];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__pd_sin_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_ds,frp_p);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_sin][sv_ds][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_sin_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_dv2,frp_f);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_sin][sv_dv2][frp_f];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_sin_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_dv2,frp_r);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_sin][sv_dv2][frp_r];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_sin_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_dv2,frp_p);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_sin][sv_dv2][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_sin_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_dv2m,frp_f);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_sin][sv_dv2m][frp_f];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_sin_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_dv2m,frp_r);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_sin][sv_dv2m][frp_r];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_sin_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_dv2m,frp_p);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_sin][sv_dv2m][frp_p];
  return (fptr(x, m));
}

#ifdef TARGET_SUPPORTS_QUADFP
vrq1_t
MTH_DISPATCH_FUNC(__fq_sin_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin, sv_qs, frp_f);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_sin][sv_qs][frp_f];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__rq_sin_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin, sv_qs, frp_r);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_sin][sv_qs][frp_r];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__pq_sin_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin, sv_qs, frp_p);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_sin][sv_qs][frp_p];
  return (fptr(x));
}
#endif

vrs1_t
MTH_DISPATCH_FUNC(__fs_tan_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_ss,frp_f);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_tan][sv_ss][frp_f];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__rs_tan_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_ss,frp_r);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_tan][sv_ss][frp_r];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__ps_tan_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_ss,frp_p);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_tan][sv_ss][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_tan_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_sv4,frp_f);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_tan][sv_sv4][frp_f];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_tan_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_sv4,frp_r);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_tan][sv_sv4][frp_r];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_tan_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_sv4,frp_p);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_tan][sv_sv4][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_tan_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_sv4m,frp_f);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_tan][sv_sv4m][frp_f];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_tan_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_sv4m,frp_r);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_tan][sv_sv4m][frp_r];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_tan_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_sv4m,frp_p);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_tan][sv_sv4m][frp_p];
  return (fptr(x, m));
}

vrd1_t
MTH_DISPATCH_FUNC(__fd_tan_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_ds,frp_f);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_tan][sv_ds][frp_f];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__rd_tan_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_ds,frp_r);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_tan][sv_ds][frp_r];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__pd_tan_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_ds,frp_p);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_tan][sv_ds][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_tan_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_dv2,frp_f);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_tan][sv_dv2][frp_f];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_tan_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_dv2,frp_r);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_tan][sv_dv2][frp_r];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_tan_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_dv2,frp_p);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_tan][sv_dv2][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_tan_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_dv2m,frp_f);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_tan][sv_dv2m][frp_f];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_tan_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_dv2m,frp_r);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_tan][sv_dv2m][frp_r];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_tan_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_dv2m,frp_p);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_tan][sv_dv2m][frp_p];
  return (fptr(x, m));
}

#ifdef TARGET_SUPPORTS_QUADFP
vrq1_t
MTH_DISPATCH_FUNC(__fq_tan_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan, sv_qs, frp_f);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_tan][sv_qs][frp_f];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__rq_tan_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan, sv_qs, frp_r);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_tan][sv_qs][frp_r];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__pq_tan_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan, sv_qs, frp_p);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_tan][sv_qs][frp_p];
  return (fptr(x));
}
#endif

vrs1_t
MTH_DISPATCH_FUNC(__fs_cosh_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_ss,frp_f);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_cosh][sv_ss][frp_f];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__rs_cosh_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_ss,frp_r);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_cosh][sv_ss][frp_r];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__ps_cosh_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_ss,frp_p);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_cosh][sv_ss][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_cosh_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_sv4,frp_f);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_cosh][sv_sv4][frp_f];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_cosh_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_sv4,frp_r);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_cosh][sv_sv4][frp_r];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_cosh_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_sv4,frp_p);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_cosh][sv_sv4][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_cosh_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_sv4m,frp_f);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_cosh][sv_sv4m][frp_f];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_cosh_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_sv4m,frp_r);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_cosh][sv_sv4m][frp_r];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_cosh_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_sv4m,frp_p);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_cosh][sv_sv4m][frp_p];
  return (fptr(x, m));
}

vrd1_t
MTH_DISPATCH_FUNC(__fd_cosh_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_ds,frp_f);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_cosh][sv_ds][frp_f];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__rd_cosh_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_ds,frp_r);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_cosh][sv_ds][frp_r];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__pd_cosh_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_ds,frp_p);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_cosh][sv_ds][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_cosh_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_dv2,frp_f);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_cosh][sv_dv2][frp_f];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_cosh_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_dv2,frp_r);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_cosh][sv_dv2][frp_r];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_cosh_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_dv2,frp_p);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_cosh][sv_dv2][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_cosh_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_dv2m,frp_f);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_cosh][sv_dv2m][frp_f];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_cosh_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_dv2m,frp_r);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_cosh][sv_dv2m][frp_r];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_cosh_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_dv2m,frp_p);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_cosh][sv_dv2m][frp_p];
  return (fptr(x, m));
}

#ifdef TARGET_SUPPORTS_QUADFP
vrq1_t
MTH_DISPATCH_FUNC(__fq_cosh_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh, sv_qs, frp_f);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_cosh][sv_qs][frp_f];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__rq_cosh_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh, sv_qs, frp_r);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_cosh][sv_qs][frp_r];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__pq_cosh_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh, sv_qs, frp_p);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_cosh][sv_qs][frp_p];
  return (fptr(x));
}
#endif

vrs1_t
MTH_DISPATCH_FUNC(__fs_sinh_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_ss,frp_f);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_sinh][sv_ss][frp_f];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__rs_sinh_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_ss,frp_r);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_sinh][sv_ss][frp_r];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__ps_sinh_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_ss,frp_p);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_sinh][sv_ss][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_sinh_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_sv4,frp_f);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_sinh][sv_sv4][frp_f];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_sinh_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_sv4,frp_r);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_sinh][sv_sv4][frp_r];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_sinh_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_sv4,frp_p);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_sinh][sv_sv4][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_sinh_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_sv4m,frp_f);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_sinh][sv_sv4m][frp_f];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_sinh_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_sv4m,frp_r);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_sinh][sv_sv4m][frp_r];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_sinh_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_sv4m,frp_p);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_sinh][sv_sv4m][frp_p];
  return (fptr(x, m));
}

vrd1_t
MTH_DISPATCH_FUNC(__fd_sinh_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_ds,frp_f);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_sinh][sv_ds][frp_f];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__rd_sinh_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_ds,frp_r);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_sinh][sv_ds][frp_r];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__pd_sinh_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_ds,frp_p);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_sinh][sv_ds][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_sinh_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_dv2,frp_f);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_sinh][sv_dv2][frp_f];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_sinh_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_dv2,frp_r);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_sinh][sv_dv2][frp_r];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_sinh_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_dv2,frp_p);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_sinh][sv_dv2][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_sinh_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_dv2m,frp_f);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_sinh][sv_dv2m][frp_f];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_sinh_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_dv2m,frp_r);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_sinh][sv_dv2m][frp_r];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_sinh_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_dv2m,frp_p);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_sinh][sv_dv2m][frp_p];
  return (fptr(x, m));
}

#ifdef TARGET_SUPPORTS_QUADFP
vrq1_t
MTH_DISPATCH_FUNC(__fq_sinh_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh, sv_qs, frp_f);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_sinh][sv_qs][frp_f];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__rq_sinh_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh, sv_qs, frp_r);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_sinh][sv_qs][frp_r];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__pq_sinh_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh, sv_qs, frp_p);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_sinh][sv_qs][frp_p];
  return (fptr(x));
}
#endif

vrs1_t
MTH_DISPATCH_FUNC(__fs_tanh_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_ss,frp_f);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_tanh][sv_ss][frp_f];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__rs_tanh_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_ss,frp_r);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_tanh][sv_ss][frp_r];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__ps_tanh_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_ss,frp_p);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_tanh][sv_ss][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_tanh_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_sv4,frp_f);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_tanh][sv_sv4][frp_f];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_tanh_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_sv4,frp_r);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_tanh][sv_sv4][frp_r];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_tanh_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_sv4,frp_p);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_tanh][sv_sv4][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_tanh_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_sv4m,frp_f);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_tanh][sv_sv4m][frp_f];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_tanh_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_sv4m,frp_r);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_tanh][sv_sv4m][frp_r];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_tanh_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_sv4m,frp_p);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_tanh][sv_sv4m][frp_p];
  return (fptr(x, m));
}

vrd1_t
MTH_DISPATCH_FUNC(__fd_tanh_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_ds,frp_f);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_tanh][sv_ds][frp_f];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__rd_tanh_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_ds,frp_r);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_tanh][sv_ds][frp_r];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__pd_tanh_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_ds,frp_p);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_tanh][sv_ds][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_tanh_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_dv2,frp_f);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_tanh][sv_dv2][frp_f];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_tanh_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_dv2,frp_r);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_tanh][sv_dv2][frp_r];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_tanh_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_dv2,frp_p);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_tanh][sv_dv2][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_tanh_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_dv2m,frp_f);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_tanh][sv_dv2m][frp_f];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_tanh_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_dv2m,frp_r);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_tanh][sv_dv2m][frp_r];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_tanh_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_dv2m,frp_p);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_tanh][sv_dv2m][frp_p];
  return (fptr(x, m));
}

#ifdef TARGET_SUPPORTS_QUADFP
vrq1_t
MTH_DISPATCH_FUNC(__fq_tanh_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh, sv_qs, frp_f);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_tanh][sv_qs][frp_f];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__rq_tanh_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh, sv_qs, frp_r);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_tanh][sv_qs][frp_r];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__pq_tanh_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh, sv_qs, frp_p);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_tanh][sv_qs][frp_p];
  return (fptr(x));
}
#endif

vrs1_t
MTH_DISPATCH_FUNC(__fs_exp_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_ss,frp_f);
#if     ! defined(TARGET_X8664)
  if (_ISFZEROPT0(x))
    return 1.0;
#endif

  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_exp][sv_ss][frp_f];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__rs_exp_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_ss,frp_r);
#if     ! defined(TARGET_X8664)
  if (_ISFZEROPT0(x))
    return 1.0;
#endif

  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_exp][sv_ss][frp_r];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__ps_exp_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_ss,frp_p);
#if     ! defined(TARGET_X8664)
  if (_ISFZEROPT0(x))
    return 1.0;
#endif

  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_exp][sv_ss][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_exp_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_sv4,frp_f);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_exp][sv_sv4][frp_f];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_exp_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_sv4,frp_r);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_exp][sv_sv4][frp_r];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_exp_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_sv4,frp_p);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_exp][sv_sv4][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_exp_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_sv4m,frp_f);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_exp][sv_sv4m][frp_f];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_exp_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_sv4m,frp_r);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_exp][sv_sv4m][frp_r];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_exp_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_sv4m,frp_p);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_exp][sv_sv4m][frp_p];
  return (fptr(x, m));
}

vrd1_t
MTH_DISPATCH_FUNC(__fd_exp_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_ds,frp_f);
#if     ! defined(TARGET_X8664)
  if (_ISDZEROPT0(x))
    return 1.0;
#endif

  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_exp][sv_ds][frp_f];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__rd_exp_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_ds,frp_r);
#if     ! defined(TARGET_X8664)
  if (_ISDZEROPT0(x))
    return 1.0;
#endif

  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_exp][sv_ds][frp_r];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__pd_exp_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_ds,frp_p);
#if     ! defined(TARGET_X8664)
  if (_ISDZEROPT0(x))
    return 1.0;
#endif

  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_exp][sv_ds][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_exp_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_dv2,frp_f);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_exp][sv_dv2][frp_f];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_exp_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_dv2,frp_r);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_exp][sv_dv2][frp_r];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_exp_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_dv2,frp_p);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_exp][sv_dv2][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_exp_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_dv2m,frp_f);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_exp][sv_dv2m][frp_f];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_exp_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_dv2m,frp_r);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_exp][sv_dv2m][frp_r];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_exp_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_dv2m,frp_p);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_exp][sv_dv2m][frp_p];
  return (fptr(x, m));
}

#ifdef TARGET_SUPPORTS_QUADFP
vrq1_t
MTH_DISPATCH_FUNC(__fq_exp_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp, sv_qs, frp_f);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_exp][sv_qs][frp_f];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__rq_exp_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp, sv_qs, frp_r);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_exp][sv_qs][frp_r];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__pq_exp_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp, sv_qs, frp_p);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_exp][sv_qs][frp_p];
  return (fptr(x));
}
#endif

vrs1_t
MTH_DISPATCH_FUNC(__fs_log_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_ss,frp_f);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_log][sv_ss][frp_f];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__rs_log_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_ss,frp_r);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_log][sv_ss][frp_r];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__ps_log_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_ss,frp_p);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_log][sv_ss][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_log_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_sv4,frp_f);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_log][sv_sv4][frp_f];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_log_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_sv4,frp_r);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_log][sv_sv4][frp_r];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_log_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_sv4,frp_p);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_log][sv_sv4][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_log_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_sv4m,frp_f);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_log][sv_sv4m][frp_f];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_log_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_sv4m,frp_r);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_log][sv_sv4m][frp_r];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_log_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_sv4m,frp_p);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_log][sv_sv4m][frp_p];
  return (fptr(x, m));
}

vrd1_t
MTH_DISPATCH_FUNC(__fd_log_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_ds,frp_f);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_log][sv_ds][frp_f];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__rd_log_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_ds,frp_r);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_log][sv_ds][frp_r];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__pd_log_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_ds,frp_p);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_log][sv_ds][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_log_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_dv2,frp_f);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_log][sv_dv2][frp_f];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_log_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_dv2,frp_r);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_log][sv_dv2][frp_r];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_log_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_dv2,frp_p);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_log][sv_dv2][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_log_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_dv2m,frp_f);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_log][sv_dv2m][frp_f];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_log_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_dv2m,frp_r);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_log][sv_dv2m][frp_r];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_log_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_dv2m,frp_p);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_log][sv_dv2m][frp_p];
  return (fptr(x, m));
}

#ifdef TARGET_SUPPORTS_QUADFP
vrq1_t
MTH_DISPATCH_FUNC(__fq_log_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log, sv_qs, frp_f);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_log][sv_qs][frp_f];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__rq_log_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log, sv_qs, frp_r);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_log][sv_qs][frp_r];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__pq_log_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log, sv_qs, frp_p);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_log][sv_qs][frp_p];
  return (fptr(x));
}
#endif

vrs1_t
MTH_DISPATCH_FUNC(__fs_log10_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_ss,frp_f);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_log10][sv_ss][frp_f];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__rs_log10_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_ss,frp_r);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_log10][sv_ss][frp_r];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__ps_log10_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_ss,frp_p);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_log10][sv_ss][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_log10_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_sv4,frp_f);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_log10][sv_sv4][frp_f];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_log10_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_sv4,frp_r);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_log10][sv_sv4][frp_r];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_log10_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_sv4,frp_p);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_log10][sv_sv4][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_log10_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_sv4m,frp_f);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_log10][sv_sv4m][frp_f];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_log10_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_sv4m,frp_r);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_log10][sv_sv4m][frp_r];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_log10_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_sv4m,frp_p);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_log10][sv_sv4m][frp_p];
  return (fptr(x, m));
}

vrd1_t
MTH_DISPATCH_FUNC(__fd_log10_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_ds,frp_f);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_log10][sv_ds][frp_f];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__rd_log10_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_ds,frp_r);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_log10][sv_ds][frp_r];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__pd_log10_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_ds,frp_p);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_log10][sv_ds][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_log10_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_dv2,frp_f);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_log10][sv_dv2][frp_f];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_log10_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_dv2,frp_r);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_log10][sv_dv2][frp_r];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_log10_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_dv2,frp_p);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_log10][sv_dv2][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_log10_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_dv2m,frp_f);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_log10][sv_dv2m][frp_f];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_log10_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_dv2m,frp_r);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_log10][sv_dv2m][frp_r];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_log10_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_dv2m,frp_p);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_log10][sv_dv2m][frp_p];
  return (fptr(x, m));
}

#ifdef TARGET_SUPPORTS_QUADFP
vrq1_t
MTH_DISPATCH_FUNC(__fq_log10_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10, sv_qs, frp_f);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_log10][sv_qs][frp_f];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__rq_log10_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10, sv_qs, frp_r);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_log10][sv_qs][frp_r];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__pq_log10_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10, sv_qs, frp_p);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_log10][sv_qs][frp_p];
  return (fptr(x));
}
#endif

vrs1_t
MTH_DISPATCH_FUNC(__fs_mod_1)(vrs1_t x, vrs1_t y)
{
  vrs1_t (*fptr)(vrs1_t, vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_ss,frp_f);
  fptr = (vrs1_t(*)(vrs1_t, vrs1_t))MTH_DISPATCH_TBL[func_mod][sv_ss][frp_f];
  return (fptr(x, y));
}

vrs1_t
MTH_DISPATCH_FUNC(__rs_mod_1)(vrs1_t x, vrs1_t y)
{
  vrs1_t (*fptr)(vrs1_t, vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_ss,frp_r);
  fptr = (vrs1_t(*)(vrs1_t, vrs1_t))MTH_DISPATCH_TBL[func_mod][sv_ss][frp_r];
  return (fptr(x, y));
}

vrs1_t
MTH_DISPATCH_FUNC(__ps_mod_1)(vrs1_t x, vrs1_t y)
{
  vrs1_t (*fptr)(vrs1_t, vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_ss,frp_p);
  fptr = (vrs1_t(*)(vrs1_t, vrs1_t))MTH_DISPATCH_TBL[func_mod][sv_ss][frp_p];
  return (fptr(x, y));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_mod_4)(vrs4_t x, vrs4_t y)
{
  vrs4_t (*fptr)(vrs4_t, vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_sv4,frp_f);
  fptr = (vrs4_t(*)(vrs4_t, vrs4_t))MTH_DISPATCH_TBL[func_mod][sv_sv4][frp_f];
  return (fptr(x, y));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_mod_4)(vrs4_t x, vrs4_t y)
{
  vrs4_t (*fptr)(vrs4_t, vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_sv4,frp_r);
  fptr = (vrs4_t(*)(vrs4_t, vrs4_t))MTH_DISPATCH_TBL[func_mod][sv_sv4][frp_r];
  return (fptr(x, y));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_mod_4)(vrs4_t x, vrs4_t y)
{
  vrs4_t (*fptr)(vrs4_t, vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_sv4,frp_p);
  fptr = (vrs4_t(*)(vrs4_t, vrs4_t))MTH_DISPATCH_TBL[func_mod][sv_sv4][frp_p];
  return (fptr(x, y));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_mod_4m)(vrs4_t x, vrs4_t y, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_sv4m,frp_f);
  fptr = (vrs4_t(*)(vrs4_t, vrs4_t, vis4_t))MTH_DISPATCH_TBL[func_mod][sv_sv4m][frp_f];
  return (fptr(x, y, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_mod_4m)(vrs4_t x, vrs4_t y, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_sv4m,frp_r);
  fptr = (vrs4_t(*)(vrs4_t, vrs4_t, vis4_t))MTH_DISPATCH_TBL[func_mod][sv_sv4m][frp_r];
  return (fptr(x, y, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_mod_4m)(vrs4_t x, vrs4_t y, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_sv4m,frp_p);
  fptr = (vrs4_t(*)(vrs4_t, vrs4_t, vis4_t))MTH_DISPATCH_TBL[func_mod][sv_sv4m][frp_p];
  return (fptr(x, y, m));
}

vrd1_t
MTH_DISPATCH_FUNC(__fd_mod_1)(vrd1_t x, vrd1_t y)
{
  vrd1_t (*fptr)(vrd1_t, vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_ds,frp_f);
  fptr = (vrd1_t(*)(vrd1_t, vrd1_t))MTH_DISPATCH_TBL[func_mod][sv_ds][frp_f];
  return (fptr(x, y));
}

vrd1_t
MTH_DISPATCH_FUNC(__rd_mod_1)(vrd1_t x, vrd1_t y)
{
  vrd1_t (*fptr)(vrd1_t, vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_ds,frp_r);
  fptr = (vrd1_t(*)(vrd1_t, vrd1_t))MTH_DISPATCH_TBL[func_mod][sv_ds][frp_r];
  return (fptr(x, y));
}

vrd1_t
MTH_DISPATCH_FUNC(__pd_mod_1)(vrd1_t x, vrd1_t y)
{
  vrd1_t (*fptr)(vrd1_t, vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_ds,frp_p);
  fptr = (vrd1_t(*)(vrd1_t, vrd1_t))MTH_DISPATCH_TBL[func_mod][sv_ds][frp_p];
  return (fptr(x, y));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_mod_2)(vrd2_t x, vrd2_t y)
{
  vrd2_t (*fptr)(vrd2_t, vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_dv2,frp_f);
  fptr = (vrd2_t(*)(vrd2_t, vrd2_t))MTH_DISPATCH_TBL[func_mod][sv_dv2][frp_f];
  return (fptr(x, y));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_mod_2)(vrd2_t x, vrd2_t y)
{
  vrd2_t (*fptr)(vrd2_t, vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_dv2,frp_r);
  fptr = (vrd2_t(*)(vrd2_t, vrd2_t))MTH_DISPATCH_TBL[func_mod][sv_dv2][frp_r];
  return (fptr(x, y));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_mod_2)(vrd2_t x, vrd2_t y)
{
  vrd2_t (*fptr)(vrd2_t, vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_dv2,frp_p);
  fptr = (vrd2_t(*)(vrd2_t, vrd2_t))MTH_DISPATCH_TBL[func_mod][sv_dv2][frp_p];
  return (fptr(x, y));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_mod_2m)(vrd2_t x, vrd2_t y, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_dv2m,frp_f);
  fptr = (vrd2_t(*)(vrd2_t, vrd2_t, vid2_t))MTH_DISPATCH_TBL[func_mod][sv_dv2m][frp_f];
  return (fptr(x, y, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_mod_2m)(vrd2_t x, vrd2_t y, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_dv2m,frp_r);
  fptr = (vrd2_t(*)(vrd2_t, vrd2_t, vid2_t))MTH_DISPATCH_TBL[func_mod][sv_dv2m][frp_r];
  return (fptr(x, y, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_mod_2m)(vrd2_t x, vrd2_t y, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_dv2m,frp_p);
  fptr = (vrd2_t(*)(vrd2_t, vrd2_t, vid2_t))MTH_DISPATCH_TBL[func_mod][sv_dv2m][frp_p];
  return (fptr(x, y, m));
}

#ifdef TARGET_SUPPORTS_QUADFP
vrq1_t
MTH_DISPATCH_FUNC(__fq_mod_1)(vrq1_t x, vrq1_t y)
{
  vrq1_t (*fptr)(vrq1_t, vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod, sv_qs, frp_f);
  fptr = (vrq1_t(*)(vrq1_t, vrq1_t))MTH_DISPATCH_TBL[func_mod][sv_qs][frp_f];
  return (fptr(x, y));
}

vrq1_t
MTH_DISPATCH_FUNC(__rq_mod_1)(vrq1_t x, vrq1_t y)
{
  vrq1_t (*fptr)(vrq1_t, vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod, sv_qs, frp_r);
  fptr = (vrq1_t(*)(vrq1_t, vrq1_t))MTH_DISPATCH_TBL[func_mod][sv_qs][frp_r];
  return (fptr(x, y));
}

vrq1_t
MTH_DISPATCH_FUNC(__pq_mod_1)(vrq1_t x, vrq1_t y)
{
  vrq1_t (*fptr)(vrq1_t, vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod, sv_qs, frp_p);
  fptr = (vrq1_t(*)(vrq1_t, vrq1_t))MTH_DISPATCH_TBL[func_mod][sv_qs][frp_p];
  return (fptr(x, y));
}
#endif

vrs1_t
MTH_DISPATCH_FUNC(__fs_pow_1)(vrs1_t x, vrs1_t y)
{
  vrs1_t (*fptr)(vrs1_t, vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_ss,frp_f);
  fptr = (vrs1_t(*)(vrs1_t, vrs1_t))MTH_DISPATCH_TBL[func_pow][sv_ss][frp_f];
  return (fptr(x, y));
}

vrs1_t
MTH_DISPATCH_FUNC(__rs_pow_1)(vrs1_t x, vrs1_t y)
{
  vrs1_t (*fptr)(vrs1_t, vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_ss,frp_r);
  fptr = (vrs1_t(*)(vrs1_t, vrs1_t))MTH_DISPATCH_TBL[func_pow][sv_ss][frp_r];
  return (fptr(x, y));
}

vrs1_t
MTH_DISPATCH_FUNC(__ps_pow_1)(vrs1_t x, vrs1_t y)
{
  vrs1_t (*fptr)(vrs1_t, vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_ss,frp_p);
  fptr = (vrs1_t(*)(vrs1_t, vrs1_t))MTH_DISPATCH_TBL[func_pow][sv_ss][frp_p];
  return (fptr(x, y));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_pow_4)(vrs4_t x, vrs4_t y)
{
  vrs4_t (*fptr)(vrs4_t, vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_sv4,frp_f);
  fptr = (vrs4_t(*)(vrs4_t, vrs4_t))MTH_DISPATCH_TBL[func_pow][sv_sv4][frp_f];
  return (fptr(x, y));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_pow_4)(vrs4_t x, vrs4_t y)
{
  vrs4_t (*fptr)(vrs4_t, vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_sv4,frp_r);
  fptr = (vrs4_t(*)(vrs4_t, vrs4_t))MTH_DISPATCH_TBL[func_pow][sv_sv4][frp_r];
  return (fptr(x, y));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_pow_4)(vrs4_t x, vrs4_t y)
{
  vrs4_t (*fptr)(vrs4_t, vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_sv4,frp_p);
  fptr = (vrs4_t(*)(vrs4_t, vrs4_t))MTH_DISPATCH_TBL[func_pow][sv_sv4][frp_p];
  return (fptr(x, y));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_pow_4m)(vrs4_t x, vrs4_t y, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_sv4m,frp_f);
  fptr = (vrs4_t(*)(vrs4_t, vrs4_t, vis4_t))MTH_DISPATCH_TBL[func_pow][sv_sv4m][frp_f];
  return (fptr(x, y, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_pow_4m)(vrs4_t x, vrs4_t y, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_sv4m,frp_r);
  fptr = (vrs4_t(*)(vrs4_t, vrs4_t, vis4_t))MTH_DISPATCH_TBL[func_pow][sv_sv4m][frp_r];
  return (fptr(x, y, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_pow_4m)(vrs4_t x, vrs4_t y, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_sv4m,frp_p);
  fptr = (vrs4_t(*)(vrs4_t, vrs4_t, vis4_t))MTH_DISPATCH_TBL[func_pow][sv_sv4m][frp_p];
  return (fptr(x, y, m));
}

vrd1_t
MTH_DISPATCH_FUNC(__fd_pow_1)(vrd1_t x, vrd1_t y)
{
  vrd1_t (*fptr)(vrd1_t, vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_ds,frp_f);
  fptr = (vrd1_t(*)(vrd1_t, vrd1_t))MTH_DISPATCH_TBL[func_pow][sv_ds][frp_f];
  return (fptr(x, y));
}

vrd1_t
MTH_DISPATCH_FUNC(__rd_pow_1)(vrd1_t x, vrd1_t y)
{
  vrd1_t (*fptr)(vrd1_t, vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_ds,frp_r);
  fptr = (vrd1_t(*)(vrd1_t, vrd1_t))MTH_DISPATCH_TBL[func_pow][sv_ds][frp_r];
  return (fptr(x, y));
}

vrd1_t
MTH_DISPATCH_FUNC(__pd_pow_1)(vrd1_t x, vrd1_t y)
{
  vrd1_t (*fptr)(vrd1_t, vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_ds,frp_p);
  fptr = (vrd1_t(*)(vrd1_t, vrd1_t))MTH_DISPATCH_TBL[func_pow][sv_ds][frp_p];
  return (fptr(x, y));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_pow_2)(vrd2_t x, vrd2_t y)
{
  vrd2_t (*fptr)(vrd2_t, vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_dv2,frp_f);
  fptr = (vrd2_t(*)(vrd2_t, vrd2_t))MTH_DISPATCH_TBL[func_pow][sv_dv2][frp_f];
  return (fptr(x, y));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_pow_2)(vrd2_t x, vrd2_t y)
{
  vrd2_t (*fptr)(vrd2_t, vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_dv2,frp_r);
  fptr = (vrd2_t(*)(vrd2_t, vrd2_t))MTH_DISPATCH_TBL[func_pow][sv_dv2][frp_r];
  return (fptr(x, y));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_pow_2)(vrd2_t x, vrd2_t y)
{
  vrd2_t (*fptr)(vrd2_t, vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_dv2,frp_p);
  fptr = (vrd2_t(*)(vrd2_t, vrd2_t))MTH_DISPATCH_TBL[func_pow][sv_dv2][frp_p];
  return (fptr(x, y));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_pow_2m)(vrd2_t x, vrd2_t y, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_dv2m,frp_f);
  fptr = (vrd2_t(*)(vrd2_t, vrd2_t, vid2_t))MTH_DISPATCH_TBL[func_pow][sv_dv2m][frp_f];
  return (fptr(x, y, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_pow_2m)(vrd2_t x, vrd2_t y, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_dv2m,frp_r);
  fptr = (vrd2_t(*)(vrd2_t, vrd2_t, vid2_t))MTH_DISPATCH_TBL[func_pow][sv_dv2m][frp_r];
  return (fptr(x, y, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_pow_2m)(vrd2_t x, vrd2_t y, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_dv2m,frp_p);
  fptr = (vrd2_t(*)(vrd2_t, vrd2_t, vid2_t))MTH_DISPATCH_TBL[func_pow][sv_dv2m][frp_p];
  return (fptr(x, y, m));
}

#ifdef TARGET_SUPPORTS_QUADFP
vrq1_t
MTH_DISPATCH_FUNC(__fq_pow_1)(vrq1_t x, vrq1_t y)
{
  vrq1_t (*fptr)(vrq1_t, vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow, sv_qs, frp_f);
  fptr = (vrq1_t(*)(vrq1_t, vrq1_t))MTH_DISPATCH_TBL[func_pow][sv_qs][frp_f];
  return (fptr(x, y));
}

vrq1_t
MTH_DISPATCH_FUNC(__rq_pow_1)(vrq1_t x, vrq1_t y)
{
  vrq1_t (*fptr)(vrq1_t, vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow, sv_qs, frp_r);
  fptr = (vrq1_t(*)(vrq1_t, vrq1_t))MTH_DISPATCH_TBL[func_pow][sv_qs][frp_r];
  return (fptr(x, y));
}

vrq1_t
MTH_DISPATCH_FUNC(__pq_pow_1)(vrq1_t x, vrq1_t y)
{
  vrq1_t (*fptr)(vrq1_t, vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow, sv_qs, frp_p);
  fptr = (vrq1_t(*)(vrq1_t, vrq1_t))MTH_DISPATCH_TBL[func_pow][sv_qs][frp_p];
  return (fptr(x, y));
}
#endif

vrs1_t
MTH_DISPATCH_FUNC(__fs_powi1_1)(vrs1_t x, int32_t iy)
{
  vrs1_t (*fptr)(vrs1_t, int32_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_ss,frp_f);
  fptr = (vrs1_t(*)(vrs1_t, int32_t))MTH_DISPATCH_TBL[func_powi1][sv_ss][frp_f];
  return(fptr(x,iy));
}

vrs1_t
MTH_DISPATCH_FUNC(__rs_powi1_1)(vrs1_t x, int32_t iy)
{
  vrs1_t (*fptr)(vrs1_t, int32_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_ss,frp_r);
  fptr = (vrs1_t(*)(vrs1_t, int32_t))MTH_DISPATCH_TBL[func_powi1][sv_ss][frp_r];
  return(fptr(x,iy));
}

vrs1_t
MTH_DISPATCH_FUNC(__ps_powi1_1)(vrs1_t x, int32_t iy)
{
  vrs1_t (*fptr)(vrs1_t, int32_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_ss,frp_p);
  fptr = (vrs1_t(*)(vrs1_t, int32_t))MTH_DISPATCH_TBL[func_powi1][sv_ss][frp_p];
  return(fptr(x,iy));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_powi1_4)(vrs4_t x, int32_t iy)
{
  vrs4_t (*fptr)(vrs4_t, int32_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_sv4,frp_f);
  fptr = (vrs4_t(*)(vrs4_t, int32_t))MTH_DISPATCH_TBL[func_powi1][sv_sv4][frp_f];
  return(fptr(x,iy));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_powi1_4)(vrs4_t x, int32_t iy)
{
  vrs4_t (*fptr)(vrs4_t, int32_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_sv4,frp_r);
  fptr = (vrs4_t(*)(vrs4_t, int32_t))MTH_DISPATCH_TBL[func_powi1][sv_sv4][frp_r];
  return(fptr(x,iy));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_powi1_4)(vrs4_t x, int32_t iy)
{
  vrs4_t (*fptr)(vrs4_t, int32_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_sv4,frp_p);
  fptr = (vrs4_t(*)(vrs4_t, int32_t))MTH_DISPATCH_TBL[func_powi1][sv_sv4][frp_p];
  return(fptr(x,iy));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_powi1_4m)(vrs4_t x, int32_t iy, vis4_t mask)
{
  vrs4_t (*fptr)(vrs4_t, int32_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_sv4m,frp_f);
  fptr = (vrs4_t(*)(vrs4_t, int32_t, vis4_t))MTH_DISPATCH_TBL[func_powi1][sv_sv4m][frp_f];
  return(fptr(x,iy, mask));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_powi1_4m)(vrs4_t x, int32_t iy, vis4_t mask)
{
  vrs4_t (*fptr)(vrs4_t, int32_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_sv4m,frp_r);
  fptr = (vrs4_t(*)(vrs4_t, int32_t, vis4_t))MTH_DISPATCH_TBL[func_powi1][sv_sv4m][frp_r];
  return(fptr(x,iy, mask));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_powi1_4m)(vrs4_t x, int32_t iy, vis4_t mask)
{
  vrs4_t (*fptr)(vrs4_t, int32_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_sv4m,frp_p);
  fptr = (vrs4_t(*)(vrs4_t, int32_t, vis4_t))MTH_DISPATCH_TBL[func_powi1][sv_sv4m][frp_p];
  return(fptr(x,iy, mask));
}

vrs1_t
MTH_DISPATCH_FUNC(__fs_powi_1)(vrs1_t x, vis1_t iy)
{
  vrs1_t (*fptr)(vrs1_t, vis1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_ss,frp_f);
  fptr = (vrs1_t(*)(vrs1_t, vis1_t))MTH_DISPATCH_TBL[func_powi][sv_ss][frp_f];
  return(fptr(x, iy));
}

vrs1_t
MTH_DISPATCH_FUNC(__rs_powi_1)(vrs1_t x, vis1_t iy)
{
  vrs1_t (*fptr)(vrs1_t, vis1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_ss,frp_r);
  fptr = (vrs1_t(*)(vrs1_t, vis1_t))MTH_DISPATCH_TBL[func_powi][sv_ss][frp_r];
  return(fptr(x, iy));
}

vrs1_t
MTH_DISPATCH_FUNC(__ps_powi_1)(vrs1_t x, vis1_t iy)
{
  vrs1_t (*fptr)(vrs1_t, vis1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_ss,frp_p);
  fptr = (vrs1_t(*)(vrs1_t, vis1_t))MTH_DISPATCH_TBL[func_powi][sv_ss][frp_p];
  return(fptr(x, iy));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_powi_4)(vrs4_t x, vis4_t iy)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_sv4,frp_f);
  fptr = (vrs4_t(*)(vrs4_t, vis4_t))MTH_DISPATCH_TBL[func_powi][sv_sv4][frp_f];
  return(fptr(x, iy));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_powi_4)(vrs4_t x, vis4_t iy)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_sv4,frp_r);
  fptr = (vrs4_t(*)(vrs4_t, vis4_t))MTH_DISPATCH_TBL[func_powi][sv_sv4][frp_r];
  return(fptr(x, iy));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_powi_4)(vrs4_t x, vis4_t iy)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_sv4,frp_p);
  fptr = (vrs4_t(*)(vrs4_t, vis4_t))MTH_DISPATCH_TBL[func_powi][sv_sv4][frp_p];
  return(fptr(x, iy));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_powi_4m)(vrs4_t x, vis4_t iy, vis4_t mask)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_sv4m,frp_f);
  fptr = (vrs4_t(*)(vrs4_t, vis4_t, vis4_t))MTH_DISPATCH_TBL[func_powi][sv_sv4m][frp_f];
  return(fptr(x, iy, mask));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_powi_4m)(vrs4_t x, vis4_t iy, vis4_t mask)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_sv4m,frp_r);
  fptr = (vrs4_t(*)(vrs4_t, vis4_t, vis4_t))MTH_DISPATCH_TBL[func_powi][sv_sv4m][frp_r];
  return(fptr(x, iy, mask));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_powi_4m)(vrs4_t x, vis4_t iy, vis4_t mask)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_sv4m,frp_p);
  fptr = (vrs4_t(*)(vrs4_t, vis4_t, vis4_t))MTH_DISPATCH_TBL[func_powi][sv_sv4m][frp_p];
  return(fptr(x, iy, mask));
}

vrs1_t
MTH_DISPATCH_FUNC(__fs_powk1_1)(vrs1_t x, int64_t iy)
{
  vrs1_t (*fptr)(vrs1_t, int64_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_ss,frp_f);
  fptr = (vrs1_t(*)(vrs1_t, int64_t))MTH_DISPATCH_TBL[func_powk1][sv_ss][frp_f];
  return(fptr(x,iy));
}

vrs1_t
MTH_DISPATCH_FUNC(__rs_powk1_1)(vrs1_t x, int64_t iy)
{
  vrs1_t (*fptr)(vrs1_t, int64_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_ss,frp_r);
  fptr = (vrs1_t(*)(vrs1_t, int64_t))MTH_DISPATCH_TBL[func_powk1][sv_ss][frp_r];
  return(fptr(x,iy));
}

vrs1_t
MTH_DISPATCH_FUNC(__ps_powk1_1)(vrs1_t x, int64_t iy)
{
  vrs1_t (*fptr)(vrs1_t, int64_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_ss,frp_p);
  fptr = (vrs1_t(*)(vrs1_t, int64_t))MTH_DISPATCH_TBL[func_powk1][sv_ss][frp_p];
  return(fptr(x,iy));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_powk1_4)(vrs4_t x, long long iy)
{
  vrs4_t (*fptr)(vrs4_t, long long);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_sv4,frp_f);
  fptr = (vrs4_t(*)(vrs4_t, long long))MTH_DISPATCH_TBL[func_powk1][sv_sv4][frp_f];
  return(fptr(x, iy));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_powk1_4)(vrs4_t x, long long iy)
{
  vrs4_t (*fptr)(vrs4_t, long long);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_sv4,frp_r);
  fptr = (vrs4_t(*)(vrs4_t, long long))MTH_DISPATCH_TBL[func_powk1][sv_sv4][frp_r];
  return(fptr(x, iy));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_powk1_4)(vrs4_t x, long long iy)
{
  vrs4_t (*fptr)(vrs4_t, long long);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_sv4,frp_p);
  fptr = (vrs4_t(*)(vrs4_t, long long))MTH_DISPATCH_TBL[func_powk1][sv_sv4][frp_p];
  return(fptr(x, iy));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_powk1_4m)(vrs4_t x, long long iy, vis4_t mask)
{
  vrs4_t (*fptr)(vrs4_t, long long, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_sv4m,frp_f);
  fptr = (vrs4_t(*)(vrs4_t, long long, vis4_t))MTH_DISPATCH_TBL[func_powk1][sv_sv4m][frp_f];
  return(fptr(x, iy, mask));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_powk1_4m)(vrs4_t x, long long iy, vis4_t mask)
{
  vrs4_t (*fptr)(vrs4_t, long long, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_sv4m,frp_r);
  fptr = (vrs4_t(*)(vrs4_t, long long, vis4_t))MTH_DISPATCH_TBL[func_powk1][sv_sv4m][frp_r];
  return(fptr(x, iy, mask));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_powk1_4m)(vrs4_t x, long long iy, vis4_t mask)
{
  vrs4_t (*fptr)(vrs4_t, long long, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_sv4m,frp_p);
  fptr = (vrs4_t(*)(vrs4_t, long long, vis4_t))MTH_DISPATCH_TBL[func_powk1][sv_sv4m][frp_p];
  return(fptr(x, iy, mask));
}

vrs1_t
MTH_DISPATCH_FUNC(__fs_powk_1)(vrs1_t x, vid1_t iy)
{
  vrs1_t (*fptr)(vrs1_t, vid1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_ss,frp_f);
  fptr = (vrs1_t(*)(vrs1_t, vid1_t))MTH_DISPATCH_TBL[func_powk][sv_ss][frp_f];
  return(fptr(x, iy));
}

vrs1_t
MTH_DISPATCH_FUNC(__rs_powk_1)(vrs1_t x, vid1_t iy)
{
  vrs1_t (*fptr)(vrs1_t, vid1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_ss,frp_r);
  fptr = (vrs1_t(*)(vrs1_t, vid1_t))MTH_DISPATCH_TBL[func_powk][sv_ss][frp_r];
  return(fptr(x, iy));
}

vrs1_t
MTH_DISPATCH_FUNC(__ps_powk_1)(vrs1_t x, vid1_t iy)
{
  vrs1_t (*fptr)(vrs1_t, vid1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_ss,frp_p);
  fptr = (vrs1_t(*)(vrs1_t, vid1_t))MTH_DISPATCH_TBL[func_powk][sv_ss][frp_p];
  return(fptr(x, iy));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_powk_4)(vrs4_t x, vid2_t iyu, vid2_t iyl)
{
  vrs4_t (*fptr)(vrs4_t, vid2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_sv4,frp_f);
  fptr = (vrs4_t(*)(vrs4_t, vid2_t, vid2_t))MTH_DISPATCH_TBL[func_powk][sv_sv4][frp_f];
  return(fptr(x, iyu, iyl));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_powk_4)(vrs4_t x, vid2_t iyu, vid2_t iyl)
{
  vrs4_t (*fptr)(vrs4_t, vid2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_sv4,frp_r);
  fptr = (vrs4_t(*)(vrs4_t, vid2_t, vid2_t))MTH_DISPATCH_TBL[func_powk][sv_sv4][frp_r];
  return(fptr(x, iyu, iyl));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_powk_4)(vrs4_t x, vid2_t iyu, vid2_t iyl)
{
  vrs4_t (*fptr)(vrs4_t, vid2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_sv4,frp_p);
  fptr = (vrs4_t(*)(vrs4_t, vid2_t, vid2_t))MTH_DISPATCH_TBL[func_powk][sv_sv4][frp_p];
  return(fptr(x, iyu, iyl));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_powk_4m)(vrs4_t x, vid2_t iyu, vid2_t iyl, vis4_t mask)
{
  vrs4_t (*fptr)(vrs4_t, vid2_t, vid2_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_sv4m,frp_f);
  fptr = (vrs4_t(*)(vrs4_t, vid2_t, vid2_t, vis4_t))MTH_DISPATCH_TBL[func_powk][sv_sv4m][frp_f];
  return(fptr(x, iyu, iyl, mask));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_powk_4m)(vrs4_t x, vid2_t iyu, vid2_t iyl, vis4_t mask)
{
  vrs4_t (*fptr)(vrs4_t, vid2_t, vid2_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_sv4m,frp_r);
  fptr = (vrs4_t(*)(vrs4_t, vid2_t, vid2_t, vis4_t))MTH_DISPATCH_TBL[func_powk][sv_sv4m][frp_r];
  return(fptr(x, iyu, iyl, mask));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_powk_4m)(vrs4_t x, vid2_t iyu, vid2_t iyl, vis4_t mask)
{
  vrs4_t (*fptr)(vrs4_t, vid2_t, vid2_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_sv4m,frp_p);
  fptr = (vrs4_t(*)(vrs4_t, vid2_t, vid2_t, vis4_t))MTH_DISPATCH_TBL[func_powk][sv_sv4m][frp_p];
  return(fptr(x, iyu, iyl, mask));
}

vrd1_t
MTH_DISPATCH_FUNC(__fd_powi1_1)(vrd1_t x, int32_t iy)
{
  vrd1_t (*fptr)(vrd1_t, int32_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_ds,frp_f);
  fptr = (vrd1_t(*)(vrd1_t, int32_t))MTH_DISPATCH_TBL[func_powi1][sv_ds][frp_f];
  return(fptr(x,iy));
}

vrd1_t
MTH_DISPATCH_FUNC(__rd_powi1_1)(vrd1_t x, int32_t iy)
{
  vrd1_t (*fptr)(vrd1_t, int32_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_ds,frp_r);
  fptr = (vrd1_t(*)(vrd1_t, int32_t))MTH_DISPATCH_TBL[func_powi1][sv_ds][frp_r];
  return(fptr(x,iy));
}

vrd1_t
MTH_DISPATCH_FUNC(__pd_powi1_1)(vrd1_t x, int32_t iy)
{
  vrd1_t (*fptr)(vrd1_t, int32_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_ds,frp_p);
  fptr = (vrd1_t(*)(vrd1_t, int32_t))MTH_DISPATCH_TBL[func_powi1][sv_ds][frp_p];
  return(fptr(x,iy));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_powi1_2)(vrd2_t x, int32_t iy)
{
  vrd2_t (*fptr)(vrd2_t, int32_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_dv2,frp_f);
  fptr = (vrd2_t(*)(vrd2_t, int32_t))MTH_DISPATCH_TBL[func_powi1][sv_dv2][frp_f];
  return(fptr(x,iy));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_powi1_2)(vrd2_t x, int32_t iy)
{
  vrd2_t (*fptr)(vrd2_t, int32_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_dv2,frp_r);
  fptr = (vrd2_t(*)(vrd2_t, int32_t))MTH_DISPATCH_TBL[func_powi1][sv_dv2][frp_r];
  return(fptr(x,iy));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_powi1_2)(vrd2_t x, int32_t iy)
{
  vrd2_t (*fptr)(vrd2_t, int32_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_dv2,frp_p);
  fptr = (vrd2_t(*)(vrd2_t, int32_t))MTH_DISPATCH_TBL[func_powi1][sv_dv2][frp_p];
  return(fptr(x,iy));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_powi1_2m)(vrd2_t x, int32_t iy, vid2_t mask)
{
  vrd2_t (*fptr)(vrd2_t, int32_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_dv2m,frp_f);
  fptr = (vrd2_t(*)(vrd2_t, int32_t, vid2_t))MTH_DISPATCH_TBL[func_powi1][sv_dv2m][frp_f];
  return(fptr(x,iy, mask));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_powi1_2m)(vrd2_t x, int32_t iy, vid2_t mask)
{
  vrd2_t (*fptr)(vrd2_t, int32_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_dv2m,frp_r);
  fptr = (vrd2_t(*)(vrd2_t, int32_t, vid2_t))MTH_DISPATCH_TBL[func_powi1][sv_dv2m][frp_r];
  return(fptr(x,iy, mask));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_powi1_2m)(vrd2_t x, int32_t iy, vid2_t mask)
{
  vrd2_t (*fptr)(vrd2_t, int32_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_dv2m,frp_p);
  fptr = (vrd2_t(*)(vrd2_t, int32_t, vid2_t))MTH_DISPATCH_TBL[func_powi1][sv_dv2m][frp_p];
  return(fptr(x,iy, mask));
}

vrd1_t
MTH_DISPATCH_FUNC(__fd_powi_1)(vrd1_t x, vis1_t iy)
{
  vrd1_t (*fptr)(vrd1_t, vis1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_ds,frp_f);
  fptr = (vrd1_t(*)(vrd1_t, vis1_t))MTH_DISPATCH_TBL[func_powi][sv_ds][frp_f];
  return(fptr(x, iy));
}

vrd1_t
MTH_DISPATCH_FUNC(__rd_powi_1)(vrd1_t x, vis1_t iy)
{
  vrd1_t (*fptr)(vrd1_t, vis1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_ds,frp_r);
  fptr = (vrd1_t(*)(vrd1_t, vis1_t))MTH_DISPATCH_TBL[func_powi][sv_ds][frp_r];
  return(fptr(x, iy));
}

vrd1_t
MTH_DISPATCH_FUNC(__pd_powi_1)(vrd1_t x, vis1_t iy)
{
  vrd1_t (*fptr)(vrd1_t, vis1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_ds,frp_p);
  fptr = (vrd1_t(*)(vrd1_t, vis1_t))MTH_DISPATCH_TBL[func_powi][sv_ds][frp_p];
  return(fptr(x, iy));
}

/*
 * __{frp}d_powi_2 and __{frp}d_powi_2m should technically be defined as:
 * __{frp}d_powi_2(vrd2_t x, vis4_t iy)
 * __{frp}d_powi_2m(vrd2_t x, vis4_t iy, vid2_t mask)
 *
 * But the POWER architectures needs the 32-bit integer vectors to
 * be the full 128-bits of a vector register.
 */

vrd2_t
MTH_DISPATCH_FUNC(__fd_powi_2)(vrd2_t x, vis4_t iy)
{
  vrd2_t (*fptr)(vrd2_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_dv2,frp_f);
  fptr = (vrd2_t(*)(vrd2_t, vis4_t))MTH_DISPATCH_TBL[func_powi][sv_dv2][frp_f];
  return(fptr(x, iy));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_powi_2)(vrd2_t x, vis4_t iy)
{
  vrd2_t (*fptr)(vrd2_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_dv2,frp_r);
  fptr = (vrd2_t(*)(vrd2_t, vis4_t))MTH_DISPATCH_TBL[func_powi][sv_dv2][frp_r];
  return(fptr(x, iy));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_powi_2)(vrd2_t x, vis4_t iy)
{
  vrd2_t (*fptr)(vrd2_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_dv2,frp_p);
  fptr = (vrd2_t(*)(vrd2_t, vis4_t))MTH_DISPATCH_TBL[func_powi][sv_dv2][frp_p];
  return(fptr(x, iy));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_powi_2m)(vrd2_t x, vis4_t iy, vid2_t mask)
{
  vrd2_t (*fptr)(vrd2_t, vis4_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_dv2m,frp_f);
  fptr = (vrd2_t(*)(vrd2_t, vis4_t, vid2_t))MTH_DISPATCH_TBL[func_powi][sv_dv2m][frp_f];
  return(fptr(x, iy, mask));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_powi_2m)(vrd2_t x, vis4_t iy, vid2_t mask)
{
  vrd2_t (*fptr)(vrd2_t, vis4_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_dv2m,frp_r);
  fptr = (vrd2_t(*)(vrd2_t, vis4_t, vid2_t))MTH_DISPATCH_TBL[func_powi][sv_dv2m][frp_r];
  return(fptr(x, iy, mask));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_powi_2m)(vrd2_t x, vis4_t iy, vid2_t mask)
{
  vrd2_t (*fptr)(vrd2_t, vis4_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_dv2m,frp_p);
  fptr = (vrd2_t(*)(vrd2_t, vis4_t, vid2_t))MTH_DISPATCH_TBL[func_powi][sv_dv2m][frp_p];
  return(fptr(x, iy, mask));
}

vrd1_t
MTH_DISPATCH_FUNC(__fd_powk1_1)(vrd1_t x, int64_t iy)
{
  vrd1_t (*fptr)(vrd1_t, int64_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_ds,frp_f);
  fptr = (vrd1_t(*)(vrd1_t, int64_t))MTH_DISPATCH_TBL[func_powk1][sv_ds][frp_f];
  return(fptr(x,iy));
}

vrd1_t
MTH_DISPATCH_FUNC(__rd_powk1_1)(vrd1_t x, int64_t iy)
{
  vrd1_t (*fptr)(vrd1_t, int64_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_ds,frp_r);
  fptr = (vrd1_t(*)(vrd1_t, int64_t))MTH_DISPATCH_TBL[func_powk1][sv_ds][frp_r];
  return(fptr(x,iy));
}

vrd1_t
MTH_DISPATCH_FUNC(__pd_powk1_1)(vrd1_t x, int64_t iy)
{
  vrd1_t (*fptr)(vrd1_t, int64_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_ds,frp_p);
  fptr = (vrd1_t(*)(vrd1_t, int64_t))MTH_DISPATCH_TBL[func_powk1][sv_ds][frp_p];
  return(fptr(x,iy));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_powk1_2)(vrd2_t x, long long iy)
{
  vrd2_t (*fptr)(vrd2_t, long long);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_dv2,frp_f);
  fptr = (vrd2_t(*)(vrd2_t, long long))MTH_DISPATCH_TBL[func_powk1][sv_dv2][frp_f];
  return(fptr(x, iy));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_powk1_2)(vrd2_t x, long long iy)
{
  vrd2_t (*fptr)(vrd2_t, long long);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_dv2,frp_r);
  fptr = (vrd2_t(*)(vrd2_t, long long))MTH_DISPATCH_TBL[func_powk1][sv_dv2][frp_r];
  return(fptr(x, iy));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_powk1_2)(vrd2_t x, long long iy)
{
  vrd2_t (*fptr)(vrd2_t, long long);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_dv2,frp_p);
  fptr = (vrd2_t(*)(vrd2_t, long long))MTH_DISPATCH_TBL[func_powk1][sv_dv2][frp_p];
  return(fptr(x, iy));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_powk1_2m)(vrd2_t x, long long iy, vid2_t mask)
{
  vrd2_t (*fptr)(vrd2_t, long long, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_dv2m,frp_f);
  fptr = (vrd2_t(*)(vrd2_t, long long, vid2_t))MTH_DISPATCH_TBL[func_powk1][sv_dv2m][frp_f];
  return(fptr(x, iy, mask));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_powk1_2m)(vrd2_t x, long long iy, vid2_t mask)
{
  vrd2_t (*fptr)(vrd2_t, long long, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_dv2m,frp_r);
  fptr = (vrd2_t(*)(vrd2_t, long long, vid2_t))MTH_DISPATCH_TBL[func_powk1][sv_dv2m][frp_r];
  return(fptr(x, iy, mask));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_powk1_2m)(vrd2_t x, long long iy, vid2_t mask)
{
  vrd2_t (*fptr)(vrd2_t, long long, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_dv2m,frp_p);
  fptr = (vrd2_t(*)(vrd2_t, long long, vid2_t))MTH_DISPATCH_TBL[func_powk1][sv_dv2m][frp_p];
  return(fptr(x, iy, mask));
}

vrd1_t
MTH_DISPATCH_FUNC(__fd_powk_1)(vrd1_t x, vid1_t iy)
{
  vrd1_t (*fptr)(vrd1_t, vid1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_ds,frp_f);
  fptr = (vrd1_t(*)(vrd1_t, vid1_t))MTH_DISPATCH_TBL[func_powk][sv_ds][frp_f];
  return(fptr(x, iy));
}

vrd1_t
MTH_DISPATCH_FUNC(__rd_powk_1)(vrd1_t x, vid1_t iy)
{
  vrd1_t (*fptr)(vrd1_t, vid1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_ds,frp_r);
  fptr = (vrd1_t(*)(vrd1_t, vid1_t))MTH_DISPATCH_TBL[func_powk][sv_ds][frp_r];
  return(fptr(x, iy));
}

vrd1_t
MTH_DISPATCH_FUNC(__pd_powk_1)(vrd1_t x, vid1_t iy)
{
  vrd1_t (*fptr)(vrd1_t, vid1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_ds,frp_p);
  fptr = (vrd1_t(*)(vrd1_t, vid1_t))MTH_DISPATCH_TBL[func_powk][sv_ds][frp_p];
  return(fptr(x, iy));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_powk_2)(vrd2_t x, vid2_t iy)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_dv2,frp_f);
  fptr = (vrd2_t(*)(vrd2_t, vid2_t))MTH_DISPATCH_TBL[func_powk][sv_dv2][frp_f];
  return(fptr(x, iy));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_powk_2)(vrd2_t x, vid2_t iy)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_dv2,frp_r);
  fptr = (vrd2_t(*)(vrd2_t, vid2_t))MTH_DISPATCH_TBL[func_powk][sv_dv2][frp_r];
  return(fptr(x, iy));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_powk_2)(vrd2_t x, vid2_t iy)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_dv2,frp_p);
  fptr = (vrd2_t(*)(vrd2_t, vid2_t))MTH_DISPATCH_TBL[func_powk][sv_dv2][frp_p];
  return(fptr(x, iy));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_powk_2m)(vrd2_t x, vid2_t iy, vid2_t mask)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_dv2m,frp_f);
  fptr = (vrd2_t(*)(vrd2_t, vid2_t, vid2_t))MTH_DISPATCH_TBL[func_powk][sv_dv2m][frp_f];
  return(fptr(x, iy, mask));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_powk_2m)(vrd2_t x, vid2_t iy, vid2_t mask)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_dv2m,frp_r);
  fptr = (vrd2_t(*)(vrd2_t, vid2_t, vid2_t))MTH_DISPATCH_TBL[func_powk][sv_dv2m][frp_r];
  return(fptr(x, iy, mask));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_powk_2m)(vrd2_t x, vid2_t iy, vid2_t mask)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_dv2m,frp_p);
  fptr = (vrd2_t(*)(vrd2_t, vid2_t, vid2_t))MTH_DISPATCH_TBL[func_powk][sv_dv2m][frp_p];
  return(fptr(x, iy, mask));
}

#ifdef TARGET_SUPPORTS_QUADFP
vrq1_t
MTH_DISPATCH_FUNC(__fq_powi_1)(vrq1_t x, vis1_t iy)
{
  vrq1_t (*fptr)(vrq1_t, vis1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi, sv_qs, frp_f);
  fptr = (vrq1_t(*)(vrq1_t, vis1_t))MTH_DISPATCH_TBL[func_powi][sv_qs][frp_f];
  return (fptr(x, iy));
}

vrq1_t
MTH_DISPATCH_FUNC(__rq_powi_1)(vrq1_t x, vis1_t iy)
{
  vrq1_t (*fptr)(vrq1_t, vis1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi, sv_qs, frp_r);
  fptr = (vrq1_t(*)(vrq1_t, vis1_t))MTH_DISPATCH_TBL[func_powi][sv_qs][frp_r];
  return (fptr(x, iy));
}

vrq1_t
MTH_DISPATCH_FUNC(__pq_powi_1)(vrq1_t x, vis1_t iy)
{
  vrq1_t (*fptr)(vrq1_t, vis1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi, sv_qs, frp_p);
  fptr = (vrq1_t(*)(vrq1_t, vis1_t))MTH_DISPATCH_TBL[func_powi][sv_qs][frp_p];
  return (fptr(x, iy));
}

vrq1_t
MTH_DISPATCH_FUNC(__fq_powk_1)(vrq1_t x, vid1_t iy)
{
  vrq1_t (*fptr)(vrq1_t, vid1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk, sv_qs, frp_f);
  fptr = (vrq1_t(*)(vrq1_t, vid1_t))MTH_DISPATCH_TBL[func_powk][sv_qs][frp_f];
  return (fptr(x, iy));
}

vrq1_t
MTH_DISPATCH_FUNC(__rq_powk_1)(vrq1_t x, vid1_t iy)
{
  vrq1_t (*fptr)(vrq1_t, vid1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk, sv_qs, frp_r);
  fptr = (vrq1_t(*)(vrq1_t, vid1_t))MTH_DISPATCH_TBL[func_powk][sv_qs][frp_r];
  return (fptr(x, iy));
}

vrq1_t
MTH_DISPATCH_FUNC(__pq_powk_1)(vrq1_t x, vid1_t iy)
{
  vrq1_t (*fptr)(vrq1_t, vid1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk, sv_qs, frp_p);
  fptr = (vrq1_t(*)(vrq1_t, vid1_t))MTH_DISPATCH_TBL[func_powk][sv_qs][frp_p];
  return (fptr(x, iy));
}
#endif

vrs1_t
MTH_DISPATCH_FUNC(__fs_sincos_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_ss,frp_f);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_sincos][sv_ss][frp_f];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__rs_sincos_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_ss,frp_r);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_sincos][sv_ss][frp_r];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__ps_sincos_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_ss,frp_p);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_sincos][sv_ss][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_sincos_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_sv4,frp_f);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_sincos][sv_sv4][frp_f];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_sincos_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_sv4,frp_r);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_sincos][sv_sv4][frp_r];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_sincos_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_sv4,frp_p);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_sincos][sv_sv4][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_sincos_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_sv4m,frp_f);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_sincos][sv_sv4m][frp_f];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_sincos_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_sv4m,frp_r);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_sincos][sv_sv4m][frp_r];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_sincos_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_sv4m,frp_p);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_sincos][sv_sv4m][frp_p];
  return (fptr(x, m));
}

vrd1_t
MTH_DISPATCH_FUNC(__fd_sincos_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_ds,frp_f);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_sincos][sv_ds][frp_f];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__rd_sincos_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_ds,frp_r);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_sincos][sv_ds][frp_r];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__pd_sincos_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_ds,frp_p);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_sincos][sv_ds][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_sincos_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_dv2,frp_f);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_sincos][sv_dv2][frp_f];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_sincos_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_dv2,frp_r);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_sincos][sv_dv2][frp_r];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_sincos_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_dv2,frp_p);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_sincos][sv_dv2][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_sincos_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_dv2m,frp_f);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_sincos][sv_dv2m][frp_f];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_sincos_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_dv2m,frp_r);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_sincos][sv_dv2m][frp_r];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_sincos_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_dv2m,frp_p);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_sincos][sv_dv2m][frp_p];
  return (fptr(x, m));
}

vrs1_t
MTH_DISPATCH_FUNC(__fs_aint_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_ss,frp_f);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_aint][sv_ss][frp_f];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__rs_aint_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_ss,frp_r);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_aint][sv_ss][frp_r];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__ps_aint_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_ss,frp_p);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_aint][sv_ss][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_aint_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_sv4,frp_f);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_aint][sv_sv4][frp_f];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_aint_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_sv4,frp_r);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_aint][sv_sv4][frp_r];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_aint_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_sv4,frp_p);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_aint][sv_sv4][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_aint_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_sv4m,frp_f);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_aint][sv_sv4m][frp_f];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_aint_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_sv4m,frp_r);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_aint][sv_sv4m][frp_r];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_aint_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_sv4m,frp_p);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_aint][sv_sv4m][frp_p];
  return (fptr(x, m));
}

vrd1_t
MTH_DISPATCH_FUNC(__fd_aint_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_ds,frp_f);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_aint][sv_ds][frp_f];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__rd_aint_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_ds,frp_r);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_aint][sv_ds][frp_r];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__pd_aint_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_ds,frp_p);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_aint][sv_ds][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_aint_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_dv2,frp_f);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_aint][sv_dv2][frp_f];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_aint_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_dv2,frp_r);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_aint][sv_dv2][frp_r];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_aint_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_dv2,frp_p);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_aint][sv_dv2][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_aint_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_dv2m,frp_f);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_aint][sv_dv2m][frp_f];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_aint_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_dv2m,frp_r);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_aint][sv_dv2m][frp_r];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_aint_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_dv2m,frp_p);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_aint][sv_dv2m][frp_p];
  return (fptr(x, m));
}

#ifdef TARGET_SUPPORTS_QUADFP
vrq1_t
MTH_DISPATCH_FUNC(__fq_aint_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint, sv_qs, frp_f);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_aint][sv_qs][frp_f];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__rq_aint_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint, sv_qs, frp_r);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_aint][sv_qs][frp_r];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__pq_aint_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint, sv_qs, frp_p);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_aint][sv_qs][frp_p];
  return (fptr(x));
}
#endif

vrs1_t
MTH_DISPATCH_FUNC(__fs_ceil_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_ss,frp_f);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_ceil][sv_ss][frp_f];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__rs_ceil_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_ss,frp_r);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_ceil][sv_ss][frp_r];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__ps_ceil_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_ss,frp_p);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_ceil][sv_ss][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_ceil_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_sv4,frp_f);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_ceil][sv_sv4][frp_f];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_ceil_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_sv4,frp_r);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_ceil][sv_sv4][frp_r];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_ceil_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_sv4,frp_p);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_ceil][sv_sv4][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_ceil_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_sv4m,frp_f);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_ceil][sv_sv4m][frp_f];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_ceil_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_sv4m,frp_r);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_ceil][sv_sv4m][frp_r];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_ceil_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_sv4m,frp_p);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_ceil][sv_sv4m][frp_p];
  return (fptr(x, m));
}

vrd1_t
MTH_DISPATCH_FUNC(__fd_ceil_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_ds,frp_f);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_ceil][sv_ds][frp_f];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__rd_ceil_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_ds,frp_r);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_ceil][sv_ds][frp_r];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__pd_ceil_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_ds,frp_p);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_ceil][sv_ds][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_ceil_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_dv2,frp_f);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_ceil][sv_dv2][frp_f];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_ceil_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_dv2,frp_r);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_ceil][sv_dv2][frp_r];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_ceil_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_dv2,frp_p);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_ceil][sv_dv2][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_ceil_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_dv2m,frp_f);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_ceil][sv_dv2m][frp_f];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_ceil_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_dv2m,frp_r);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_ceil][sv_dv2m][frp_r];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_ceil_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_dv2m,frp_p);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_ceil][sv_dv2m][frp_p];
  return (fptr(x, m));
}

#ifdef TARGET_SUPPORTS_QUADFP
vrq1_t
MTH_DISPATCH_FUNC(__fq_ceil_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil, sv_qs, frp_f);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_ceil][sv_qs][frp_f];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__rq_ceil_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil, sv_qs, frp_r);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_ceil][sv_qs][frp_r];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__pq_ceil_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil, sv_qs, frp_p);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_ceil][sv_qs][frp_p];
  return (fptr(x));
}
#endif

vrs1_t
MTH_DISPATCH_FUNC(__fs_floor_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_ss,frp_f);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_floor][sv_ss][frp_f];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__rs_floor_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_ss,frp_r);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_floor][sv_ss][frp_r];
  return (fptr(x));
}

vrs1_t
MTH_DISPATCH_FUNC(__ps_floor_1)(vrs1_t x)
{
  vrs1_t (*fptr)(vrs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_ss,frp_p);
  fptr = (vrs1_t(*)(vrs1_t))MTH_DISPATCH_TBL[func_floor][sv_ss][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_floor_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_sv4,frp_f);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_floor][sv_sv4][frp_f];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_floor_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_sv4,frp_r);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_floor][sv_sv4][frp_r];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_floor_4)(vrs4_t x)
{
  vrs4_t (*fptr)(vrs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_sv4,frp_p);
  fptr = (vrs4_t(*)(vrs4_t))MTH_DISPATCH_TBL[func_floor][sv_sv4][frp_p];
  return (fptr(x));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_floor_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_sv4m,frp_f);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_floor][sv_sv4m][frp_f];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_floor_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_sv4m,frp_r);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_floor][sv_sv4m][frp_r];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_floor_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_sv4m,frp_p);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_floor][sv_sv4m][frp_p];
  return (fptr(x, m));
}

vrd1_t
MTH_DISPATCH_FUNC(__fd_floor_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_ds,frp_f);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_floor][sv_ds][frp_f];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__rd_floor_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_ds,frp_r);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_floor][sv_ds][frp_r];
  return (fptr(x));
}

vrd1_t
MTH_DISPATCH_FUNC(__pd_floor_1)(vrd1_t x)
{
  vrd1_t (*fptr)(vrd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_ds,frp_p);
  fptr = (vrd1_t(*)(vrd1_t))MTH_DISPATCH_TBL[func_floor][sv_ds][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_floor_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_dv2,frp_f);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_floor][sv_dv2][frp_f];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_floor_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_dv2,frp_r);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_floor][sv_dv2][frp_r];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_floor_2)(vrd2_t x)
{
  vrd2_t (*fptr)(vrd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_dv2,frp_p);
  fptr = (vrd2_t(*)(vrd2_t))MTH_DISPATCH_TBL[func_floor][sv_dv2][frp_p];
  return (fptr(x));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_floor_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_dv2m,frp_f);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_floor][sv_dv2m][frp_f];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_floor_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_dv2m,frp_r);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_floor][sv_dv2m][frp_r];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_floor_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_dv2m,frp_p);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_floor][sv_dv2m][frp_p];
  return (fptr(x, m));
}

#ifdef TARGET_SUPPORTS_QUADFP
vrq1_t
MTH_DISPATCH_FUNC(__fq_floor_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor, sv_qs, frp_f);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_floor][sv_qs][frp_f];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__rq_floor_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor, sv_qs, frp_r);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_floor][sv_qs][frp_r];
  return (fptr(x));
}

vrq1_t
MTH_DISPATCH_FUNC(__pq_floor_1)(vrq1_t x)
{
  vrq1_t (*fptr)(vrq1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor, sv_qs, frp_p);
  fptr = (vrq1_t(*)(vrq1_t))MTH_DISPATCH_TBL[func_floor][sv_qs][frp_p];
  return (fptr(x));
}
#endif

//////////
// EXPERIMENTAL - _Complex - start
//////////
float _Complex
MTH_DISPATCH_FUNC(__fc_acos_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_cs,frp_f);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_acos][sv_cs][frp_f];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__rc_acos_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_cs,frp_r);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_acos][sv_cs][frp_r];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__pc_acos_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_cs,frp_p);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_acos][sv_cs][frp_p];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_acos_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_cv2,frp_f);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_acos][sv_cv2][frp_f];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_acos_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_cv2,frp_r);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_acos][sv_cv2][frp_r];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_acos_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_cv2,frp_p);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_acos][sv_cv2][frp_p];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_acos_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_cv2m,frp_f);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_acos][sv_cv2m][frp_f];
  return (fptr(x, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_acos_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_cv2m,frp_r);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_acos][sv_cv2m][frp_r];
  return (fptr(x, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_acos_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_cv2m,frp_p);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_acos][sv_cv2m][frp_p];
  return (fptr(x, m));
}

double _Complex
MTH_DISPATCH_FUNC(__fz_acos_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_zs,frp_f);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_acos][sv_zs][frp_f];
  return (fptr(x));
}

double _Complex
MTH_DISPATCH_FUNC(__rz_acos_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_zs,frp_r);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_acos][sv_zs][frp_r];
  return (fptr(x));
}

double _Complex
MTH_DISPATCH_FUNC(__pz_acos_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_zs,frp_p);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_acos][sv_zs][frp_p];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__fc_asin_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_cs,frp_f);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_asin][sv_cs][frp_f];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__rc_asin_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_cs,frp_r);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_asin][sv_cs][frp_r];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__pc_asin_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_cs,frp_p);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_asin][sv_cs][frp_p];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_asin_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_cv2,frp_f);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_asin][sv_cv2][frp_f];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_asin_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_cv2,frp_r);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_asin][sv_cv2][frp_r];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_asin_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_cv2,frp_p);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_asin][sv_cv2][frp_p];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_asin_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_cv2m,frp_f);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_asin][sv_cv2m][frp_f];
  return (fptr(x, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_asin_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_cv2m,frp_r);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_asin][sv_cv2m][frp_r];
  return (fptr(x, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_asin_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_cv2m,frp_p);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_asin][sv_cv2m][frp_p];
  return (fptr(x, m));
}

double _Complex
MTH_DISPATCH_FUNC(__fz_asin_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_zs,frp_f);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_asin][sv_zs][frp_f];
  return (fptr(x));
}

double _Complex
MTH_DISPATCH_FUNC(__rz_asin_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_zs,frp_r);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_asin][sv_zs][frp_r];
  return (fptr(x));
}

double _Complex
MTH_DISPATCH_FUNC(__pz_asin_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_zs,frp_p);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_asin][sv_zs][frp_p];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__fc_atan_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_cs,frp_f);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_atan][sv_cs][frp_f];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__rc_atan_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_cs,frp_r);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_atan][sv_cs][frp_r];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__pc_atan_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_cs,frp_p);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_atan][sv_cs][frp_p];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_atan_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_cv2,frp_f);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_atan][sv_cv2][frp_f];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_atan_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_cv2,frp_r);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_atan][sv_cv2][frp_r];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_atan_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_cv2,frp_p);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_atan][sv_cv2][frp_p];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_atan_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_cv2m,frp_f);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_atan][sv_cv2m][frp_f];
  return (fptr(x, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_atan_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_cv2m,frp_r);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_atan][sv_cv2m][frp_r];
  return (fptr(x, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_atan_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_cv2m,frp_p);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_atan][sv_cv2m][frp_p];
  return (fptr(x, m));
}

double _Complex
MTH_DISPATCH_FUNC(__fz_atan_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_zs,frp_f);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_atan][sv_zs][frp_f];
  return (fptr(x));
}

double _Complex
MTH_DISPATCH_FUNC(__rz_atan_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_zs,frp_r);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_atan][sv_zs][frp_r];
  return (fptr(x));
}

double _Complex
MTH_DISPATCH_FUNC(__pz_atan_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_zs,frp_p);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_atan][sv_zs][frp_p];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__fc_cos_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_cs,frp_f);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_cos][sv_cs][frp_f];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__rc_cos_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_cs,frp_r);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_cos][sv_cs][frp_r];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__pc_cos_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_cs,frp_p);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_cos][sv_cs][frp_p];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_cos_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_cv2,frp_f);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_cos][sv_cv2][frp_f];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_cos_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_cv2,frp_r);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_cos][sv_cv2][frp_r];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_cos_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_cv2,frp_p);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_cos][sv_cv2][frp_p];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_cos_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_cv2m,frp_f);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_cos][sv_cv2m][frp_f];
  return (fptr(x, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_cos_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_cv2m,frp_r);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_cos][sv_cv2m][frp_r];
  return (fptr(x, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_cos_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_cv2m,frp_p);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_cos][sv_cv2m][frp_p];
  return (fptr(x, m));
}

double _Complex
MTH_DISPATCH_FUNC(__fz_cos_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_zs,frp_f);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_cos][sv_zs][frp_f];
  return (fptr(x));
}

double _Complex
MTH_DISPATCH_FUNC(__rz_cos_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_zs,frp_r);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_cos][sv_zs][frp_r];
  return (fptr(x));
}

double _Complex
MTH_DISPATCH_FUNC(__pz_cos_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_zs,frp_p);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_cos][sv_zs][frp_p];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__fc_sin_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_cs,frp_f);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_sin][sv_cs][frp_f];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__rc_sin_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_cs,frp_r);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_sin][sv_cs][frp_r];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__pc_sin_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_cs,frp_p);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_sin][sv_cs][frp_p];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_sin_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_cv2,frp_f);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_sin][sv_cv2][frp_f];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_sin_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_cv2,frp_r);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_sin][sv_cv2][frp_r];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_sin_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_cv2,frp_p);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_sin][sv_cv2][frp_p];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_sin_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_cv2m,frp_f);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_sin][sv_cv2m][frp_f];
  return (fptr(x, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_sin_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_cv2m,frp_r);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_sin][sv_cv2m][frp_r];
  return (fptr(x, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_sin_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_cv2m,frp_p);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_sin][sv_cv2m][frp_p];
  return (fptr(x, m));
}

double _Complex
MTH_DISPATCH_FUNC(__fz_sin_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_zs,frp_f);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_sin][sv_zs][frp_f];
  return (fptr(x));
}

double _Complex
MTH_DISPATCH_FUNC(__rz_sin_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_zs,frp_r);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_sin][sv_zs][frp_r];
  return (fptr(x));
}

double _Complex
MTH_DISPATCH_FUNC(__pz_sin_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_zs,frp_p);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_sin][sv_zs][frp_p];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__fc_tan_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_cs,frp_f);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_tan][sv_cs][frp_f];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__rc_tan_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_cs,frp_r);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_tan][sv_cs][frp_r];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__pc_tan_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_cs,frp_p);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_tan][sv_cs][frp_p];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_tan_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_cv2,frp_f);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_tan][sv_cv2][frp_f];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_tan_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_cv2,frp_r);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_tan][sv_cv2][frp_r];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_tan_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_cv2,frp_p);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_tan][sv_cv2][frp_p];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_tan_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_cv2m,frp_f);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_tan][sv_cv2m][frp_f];
  return (fptr(x, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_tan_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_cv2m,frp_r);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_tan][sv_cv2m][frp_r];
  return (fptr(x, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_tan_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_cv2m,frp_p);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_tan][sv_cv2m][frp_p];
  return (fptr(x, m));
}

double _Complex
MTH_DISPATCH_FUNC(__fz_tan_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_zs,frp_f);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_tan][sv_zs][frp_f];
  return (fptr(x));
}

double _Complex
MTH_DISPATCH_FUNC(__rz_tan_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_zs,frp_r);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_tan][sv_zs][frp_r];
  return (fptr(x));
}

double _Complex
MTH_DISPATCH_FUNC(__pz_tan_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_zs,frp_p);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_tan][sv_zs][frp_p];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__fc_cosh_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_cs,frp_f);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_cosh][sv_cs][frp_f];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__rc_cosh_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_cs,frp_r);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_cosh][sv_cs][frp_r];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__pc_cosh_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_cs,frp_p);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_cosh][sv_cs][frp_p];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_cosh_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_cv2,frp_f);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_cosh][sv_cv2][frp_f];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_cosh_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_cv2,frp_r);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_cosh][sv_cv2][frp_r];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_cosh_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_cv2,frp_p);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_cosh][sv_cv2][frp_p];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_cosh_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_cv2m,frp_f);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_cosh][sv_cv2m][frp_f];
  return (fptr(x, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_cosh_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_cv2m,frp_r);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_cosh][sv_cv2m][frp_r];
  return (fptr(x, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_cosh_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_cv2m,frp_p);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_cosh][sv_cv2m][frp_p];
  return (fptr(x, m));
}

double _Complex
MTH_DISPATCH_FUNC(__fz_cosh_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_zs,frp_f);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_cosh][sv_zs][frp_f];
  return (fptr(x));
}

double _Complex
MTH_DISPATCH_FUNC(__rz_cosh_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_zs,frp_r);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_cosh][sv_zs][frp_r];
  return (fptr(x));
}

double _Complex
MTH_DISPATCH_FUNC(__pz_cosh_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_zs,frp_p);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_cosh][sv_zs][frp_p];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__fc_sinh_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_cs,frp_f);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_sinh][sv_cs][frp_f];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__rc_sinh_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_cs,frp_r);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_sinh][sv_cs][frp_r];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__pc_sinh_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_cs,frp_p);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_sinh][sv_cs][frp_p];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_sinh_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_cv2,frp_f);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_sinh][sv_cv2][frp_f];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_sinh_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_cv2,frp_r);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_sinh][sv_cv2][frp_r];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_sinh_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_cv2,frp_p);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_sinh][sv_cv2][frp_p];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_sinh_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_cv2m,frp_f);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_sinh][sv_cv2m][frp_f];
  return (fptr(x, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_sinh_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_cv2m,frp_r);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_sinh][sv_cv2m][frp_r];
  return (fptr(x, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_sinh_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_cv2m,frp_p);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_sinh][sv_cv2m][frp_p];
  return (fptr(x, m));
}

double _Complex
MTH_DISPATCH_FUNC(__fz_sinh_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_zs,frp_f);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_sinh][sv_zs][frp_f];
  return (fptr(x));
}

double _Complex
MTH_DISPATCH_FUNC(__rz_sinh_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_zs,frp_r);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_sinh][sv_zs][frp_r];
  return (fptr(x));
}

double _Complex
MTH_DISPATCH_FUNC(__pz_sinh_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_zs,frp_p);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_sinh][sv_zs][frp_p];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__fc_tanh_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_cs,frp_f);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_tanh][sv_cs][frp_f];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__rc_tanh_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_cs,frp_r);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_tanh][sv_cs][frp_r];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__pc_tanh_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_cs,frp_p);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_tanh][sv_cs][frp_p];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_tanh_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_cv2,frp_f);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_tanh][sv_cv2][frp_f];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_tanh_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_cv2,frp_r);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_tanh][sv_cv2][frp_r];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_tanh_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_cv2,frp_p);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_tanh][sv_cv2][frp_p];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_tanh_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_cv2m,frp_f);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_tanh][sv_cv2m][frp_f];
  return (fptr(x, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_tanh_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_cv2m,frp_r);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_tanh][sv_cv2m][frp_r];
  return (fptr(x, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_tanh_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_cv2m,frp_p);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_tanh][sv_cv2m][frp_p];
  return (fptr(x, m));
}

double _Complex
MTH_DISPATCH_FUNC(__fz_tanh_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_zs,frp_f);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_tanh][sv_zs][frp_f];
  return (fptr(x));
}

double _Complex
MTH_DISPATCH_FUNC(__rz_tanh_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_zs,frp_r);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_tanh][sv_zs][frp_r];
  return (fptr(x));
}

double _Complex
MTH_DISPATCH_FUNC(__pz_tanh_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_zs,frp_p);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_tanh][sv_zs][frp_p];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__fc_exp_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_cs,frp_f);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_exp][sv_cs][frp_f];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__rc_exp_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_cs,frp_r);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_exp][sv_cs][frp_r];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__pc_exp_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_cs,frp_p);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_exp][sv_cs][frp_p];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_exp_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_cv2,frp_f);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_exp][sv_cv2][frp_f];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_exp_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_cv2,frp_r);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_exp][sv_cv2][frp_r];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_exp_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_cv2,frp_p);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_exp][sv_cv2][frp_p];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_exp_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_cv2m,frp_f);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_exp][sv_cv2m][frp_f];
  return (fptr(x, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_exp_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_cv2m,frp_r);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_exp][sv_cv2m][frp_r];
  return (fptr(x, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_exp_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_cv2m,frp_p);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_exp][sv_cv2m][frp_p];
  return (fptr(x, m));
}

double _Complex
MTH_DISPATCH_FUNC(__fz_exp_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_zs,frp_f);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_exp][sv_zs][frp_f];
  return (fptr(x));
}

double _Complex
MTH_DISPATCH_FUNC(__rz_exp_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_zs,frp_r);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_exp][sv_zs][frp_r];
  return (fptr(x));
}

double _Complex
MTH_DISPATCH_FUNC(__pz_exp_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_zs,frp_p);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_exp][sv_zs][frp_p];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__fc_log_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_cs,frp_f);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_log][sv_cs][frp_f];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__rc_log_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_cs,frp_r);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_log][sv_cs][frp_r];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__pc_log_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_cs,frp_p);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_log][sv_cs][frp_p];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_log_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_cv2,frp_f);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_log][sv_cv2][frp_f];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_log_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_cv2,frp_r);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_log][sv_cv2][frp_r];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_log_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_cv2,frp_p);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_log][sv_cv2][frp_p];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_log_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_cv2m,frp_f);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_log][sv_cv2m][frp_f];
  return (fptr(x, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_log_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_cv2m,frp_r);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_log][sv_cv2m][frp_r];
  return (fptr(x, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_log_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_cv2m,frp_p);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_log][sv_cv2m][frp_p];
  return (fptr(x, m));
}

double _Complex
MTH_DISPATCH_FUNC(__fz_log_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_zs,frp_f);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_log][sv_zs][frp_f];
  return (fptr(x));
}

double _Complex
MTH_DISPATCH_FUNC(__rz_log_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_zs,frp_r);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_log][sv_zs][frp_r];
  return (fptr(x));
}

double _Complex
MTH_DISPATCH_FUNC(__pz_log_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_zs,frp_p);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_log][sv_zs][frp_p];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__fc_log10_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_cs,frp_f);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_log10][sv_cs][frp_f];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__rc_log10_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_cs,frp_r);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_log10][sv_cs][frp_r];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__pc_log10_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_cs,frp_p);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_log10][sv_cs][frp_p];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_log10_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_cv2,frp_f);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_log10][sv_cv2][frp_f];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_log10_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_cv2,frp_r);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_log10][sv_cv2][frp_r];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_log10_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_cv2,frp_p);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_log10][sv_cv2][frp_p];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_log10_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_cv2m,frp_f);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_log10][sv_cv2m][frp_f];
  return (fptr(x, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_log10_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_cv2m,frp_r);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_log10][sv_cv2m][frp_r];
  return (fptr(x, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_log10_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_cv2m,frp_p);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_log10][sv_cv2m][frp_p];
  return (fptr(x, m));
}

double _Complex
MTH_DISPATCH_FUNC(__fz_log10_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_zs,frp_f);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_log10][sv_zs][frp_f];
  return (fptr(x));
}

double _Complex
MTH_DISPATCH_FUNC(__rz_log10_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_zs,frp_r);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_log10][sv_zs][frp_r];
  return (fptr(x));
}

double _Complex
MTH_DISPATCH_FUNC(__pz_log10_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_zs,frp_p);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_log10][sv_zs][frp_p];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__fc_pow_1)(float _Complex x, float _Complex y)
{
  float _Complex (*fptr)(float _Complex, float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_cs,frp_f);
  fptr = (float _Complex(*)(float _Complex, float _Complex))MTH_DISPATCH_TBL[func_pow][sv_cs][frp_f];
  return (fptr(x, y));
}

float _Complex
MTH_DISPATCH_FUNC(__rc_pow_1)(float _Complex x, float _Complex y)
{
  float _Complex (*fptr)(float _Complex, float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_cs,frp_r);
  fptr = (float _Complex(*)(float _Complex, float _Complex))MTH_DISPATCH_TBL[func_pow][sv_cs][frp_r];
  return (fptr(x, y));
}

float _Complex
MTH_DISPATCH_FUNC(__pc_pow_1)(float _Complex x, float _Complex y)
{
  float _Complex (*fptr)(float _Complex, float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_cs,frp_p);
  fptr = (float _Complex(*)(float _Complex, float _Complex))MTH_DISPATCH_TBL[func_pow][sv_cs][frp_p];
  return (fptr(x, y));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_pow_2)(vcs2_t x, vcs2_t y)
{
  vcs2_t (*fptr)(vcs2_t, vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_cv2,frp_f);
  fptr = (vcs2_t(*)(vcs2_t, vcs2_t))MTH_DISPATCH_TBL[func_pow][sv_cv2][frp_f];
  return (fptr(x, y));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_pow_2)(vcs2_t x, vcs2_t y)
{
  vcs2_t (*fptr)(vcs2_t, vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_cv2,frp_r);
  fptr = (vcs2_t(*)(vcs2_t, vcs2_t))MTH_DISPATCH_TBL[func_pow][sv_cv2][frp_r];
  return (fptr(x, y));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_pow_2)(vcs2_t x, vcs2_t y)
{
  vcs2_t (*fptr)(vcs2_t, vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_cv2,frp_p);
  fptr = (vcs2_t(*)(vcs2_t, vcs2_t))MTH_DISPATCH_TBL[func_pow][sv_cv2][frp_p];
  return (fptr(x, y));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_pow_2m)(vcs2_t x, vcs2_t y, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_cv2m,frp_f);
  fptr = (vcs2_t(*)(vcs2_t, vcs2_t, vis2_t))MTH_DISPATCH_TBL[func_pow][sv_cv2m][frp_f];
  return (fptr(x, y, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_pow_2m)(vcs2_t x, vcs2_t y, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_cv2m,frp_r);
  fptr = (vcs2_t(*)(vcs2_t, vcs2_t, vis2_t))MTH_DISPATCH_TBL[func_pow][sv_cv2m][frp_r];
  return (fptr(x, y, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_pow_2m)(vcs2_t x, vcs2_t y, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_cv2m,frp_p);
  fptr = (vcs2_t(*)(vcs2_t, vcs2_t, vis2_t))MTH_DISPATCH_TBL[func_pow][sv_cv2m][frp_p];
  return (fptr(x, y, m));
}

double _Complex
MTH_DISPATCH_FUNC(__fz_pow_1)(double _Complex x, double _Complex y)
{
  double _Complex (*fptr)(double _Complex, double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_zs,frp_f);
  fptr = (double _Complex(*)(double _Complex, double _Complex))MTH_DISPATCH_TBL[func_pow][sv_zs][frp_f];
  return (fptr(x, y));
}

double _Complex
MTH_DISPATCH_FUNC(__rz_pow_1)(double _Complex x, double _Complex y)
{
  double _Complex (*fptr)(double _Complex, double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_zs,frp_r);
  fptr = (double _Complex(*)(double _Complex, double _Complex))MTH_DISPATCH_TBL[func_pow][sv_zs][frp_r];
  return (fptr(x, y));
}

double _Complex
MTH_DISPATCH_FUNC(__pz_pow_1)(double _Complex x, double _Complex y)
{
  double _Complex (*fptr)(double _Complex, double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_zs,frp_p);
  fptr = (double _Complex(*)(double _Complex, double _Complex))MTH_DISPATCH_TBL[func_pow][sv_zs][frp_p];
  return (fptr(x, y));
}

float _Complex
MTH_DISPATCH_FUNC(__fc_powi_1)(float _Complex x, int iy)
{
  float _Complex (*fptr)(float _Complex, int);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_cs,frp_f);
  fptr = (float _Complex(*)(float _Complex, int))MTH_DISPATCH_TBL[func_powi][sv_cs][frp_f];
  return (fptr(x, iy));
}

float _Complex
MTH_DISPATCH_FUNC(__rc_powi_1)(float _Complex x, int iy)
{
  float _Complex (*fptr)(float _Complex, int);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_cs,frp_r);
  fptr = (float _Complex(*)(float _Complex, int))MTH_DISPATCH_TBL[func_powi][sv_cs][frp_r];
  return (fptr(x, iy));
}

float _Complex
MTH_DISPATCH_FUNC(__pc_powi_1)(float _Complex x, int iy)
{
  float _Complex (*fptr)(float _Complex, int);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_cs,frp_p);
  fptr = (float _Complex(*)(float _Complex, int))MTH_DISPATCH_TBL[func_powi][sv_cs][frp_p];
  return (fptr(x, iy));
}

float _Complex
MTH_DISPATCH_FUNC(__fc_powk_1)(float _Complex x, long long iy)
{
  float _Complex (*fptr)(float _Complex, long long);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_cs,frp_f);
  fptr = (float _Complex(*)(float _Complex, long long))MTH_DISPATCH_TBL[func_powk][sv_cs][frp_f];
  return (fptr(x, iy));
}

float _Complex
MTH_DISPATCH_FUNC(__rc_powk_1)(float _Complex x, long long iy)
{
  float _Complex (*fptr)(float _Complex, long long);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_cs,frp_r);
  fptr = (float _Complex(*)(float _Complex, long long))MTH_DISPATCH_TBL[func_powk][sv_cs][frp_r];
  return (fptr(x, iy));
}

float _Complex
MTH_DISPATCH_FUNC(__pc_powk_1)(float _Complex x, long long iy)
{
  float _Complex (*fptr)(float _Complex, long long);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_cs,frp_p);
  fptr = (float _Complex(*)(float _Complex, long long))MTH_DISPATCH_TBL[func_powk][sv_cs][frp_p];
  return (fptr(x, iy));
}

double _Complex
MTH_DISPATCH_FUNC(__fz_powi_1)(double _Complex x, int iy)
{
  double _Complex (*fptr)(double _Complex, int);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_zs,frp_f);
  fptr = (double _Complex(*)(double _Complex, int))MTH_DISPATCH_TBL[func_powi][sv_zs][frp_f];
  return (fptr(x, iy));
}

double _Complex
MTH_DISPATCH_FUNC(__rz_powi_1)(double _Complex x, int iy)
{
  double _Complex (*fptr)(double _Complex, int);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_zs,frp_r);
  fptr = (double _Complex(*)(double _Complex, int))MTH_DISPATCH_TBL[func_powi][sv_zs][frp_r];
  return (fptr(x, iy));
}

double _Complex
MTH_DISPATCH_FUNC(__pz_powi_1)(double _Complex x, int iy)
{
  double _Complex (*fptr)(double _Complex, int);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_zs,frp_p);
  fptr = (double _Complex(*)(double _Complex, int))MTH_DISPATCH_TBL[func_powi][sv_zs][frp_p];
  return (fptr(x, iy));
}

double _Complex
MTH_DISPATCH_FUNC(__fz_powk_1)(double _Complex x, long long iy)
{
  double _Complex (*fptr)(double _Complex, long long);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_zs,frp_f);
  fptr = (double _Complex(*)(double _Complex, long long))MTH_DISPATCH_TBL[func_powk][sv_zs][frp_f];
  return (fptr(x, iy));
}

double _Complex
MTH_DISPATCH_FUNC(__rz_powk_1)(double _Complex x, long long iy)
{
  double _Complex (*fptr)(double _Complex, long long);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_zs,frp_r);
  fptr = (double _Complex(*)(double _Complex, long long))MTH_DISPATCH_TBL[func_powk][sv_zs][frp_r];
  return (fptr(x, iy));
}

double _Complex
MTH_DISPATCH_FUNC(__pz_powk_1)(double _Complex x, long long iy)
{
  double _Complex (*fptr)(double _Complex, long long);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_zs,frp_p);
  fptr = (double _Complex(*)(double _Complex, long long))MTH_DISPATCH_TBL[func_powk][sv_zs][frp_p];
  return (fptr(x, iy));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_div_4m)(vrs4_t x, vrs4_t y, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_sv4m,frp_f);
  fptr = (vrs4_t(*)(vrs4_t, vrs4_t, vis4_t))MTH_DISPATCH_TBL[func_div][sv_sv4m][frp_f];
  return (fptr(x, y, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_div_4m)(vrs4_t x, vrs4_t y, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_sv4m,frp_r);
  fptr = (vrs4_t(*)(vrs4_t, vrs4_t, vis4_t))MTH_DISPATCH_TBL[func_div][sv_sv4m][frp_r];
  return (fptr(x, y, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_div_4m)(vrs4_t x, vrs4_t y, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_sv4m,frp_p);
  fptr = (vrs4_t(*)(vrs4_t, vrs4_t, vis4_t))MTH_DISPATCH_TBL[func_div][sv_sv4m][frp_p];
  return (fptr(x, y, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_div_2m)(vrd2_t x, vrd2_t y, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_dv2m,frp_f);
  fptr = (vrd2_t(*)(vrd2_t, vrd2_t, vid2_t))MTH_DISPATCH_TBL[func_div][sv_dv2m][frp_f];
  return (fptr(x, y, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_div_2m)(vrd2_t x, vrd2_t y, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_dv2m,frp_r);
  fptr = (vrd2_t(*)(vrd2_t, vrd2_t, vid2_t))MTH_DISPATCH_TBL[func_div][sv_dv2m][frp_r];
  return (fptr(x, y, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_div_2m)(vrd2_t x, vrd2_t y, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_dv2m,frp_p);
  fptr = (vrd2_t(*)(vrd2_t, vrd2_t, vid2_t))MTH_DISPATCH_TBL[func_div][sv_dv2m][frp_p];
  return (fptr(x, y, m));
}

float _Complex
MTH_DISPATCH_FUNC(__fc_div_1)(float _Complex x, float _Complex y)
{
  float _Complex (*fptr)(float _Complex, float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_cs,frp_f);
  fptr = (float _Complex(*)(float _Complex, float _Complex))MTH_DISPATCH_TBL[func_div][sv_cs][frp_f];
  return (fptr(x, y));
}

float _Complex
MTH_DISPATCH_FUNC(__rc_div_1)(float _Complex x, float _Complex y)
{
  float _Complex (*fptr)(float _Complex, float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_cs,frp_r);
  fptr = (float _Complex(*)(float _Complex, float _Complex))MTH_DISPATCH_TBL[func_div][sv_cs][frp_r];
  return (fptr(x, y));
}

float _Complex
MTH_DISPATCH_FUNC(__pc_div_1)(float _Complex x, float _Complex y)
{
  float _Complex (*fptr)(float _Complex, float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_cs,frp_p);
  fptr = (float _Complex(*)(float _Complex, float _Complex))MTH_DISPATCH_TBL[func_div][sv_cs][frp_p];
  return (fptr(x, y));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_div_2)(vcs2_t x, vcs2_t y)
{
  vcs2_t (*fptr)(vcs2_t, vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_cv2,frp_f);
  fptr = (vcs2_t(*)(vcs2_t, vcs2_t))MTH_DISPATCH_TBL[func_div][sv_cv2][frp_f];
  return (fptr(x, y));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_div_2)(vcs2_t x, vcs2_t y)
{
  vcs2_t (*fptr)(vcs2_t, vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_cv2,frp_r);
  fptr = (vcs2_t(*)(vcs2_t, vcs2_t))MTH_DISPATCH_TBL[func_div][sv_cv2][frp_r];
  return (fptr(x, y));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_div_2)(vcs2_t x, vcs2_t y)
{
  vcs2_t (*fptr)(vcs2_t, vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_cv2,frp_p);
  fptr = (vcs2_t(*)(vcs2_t, vcs2_t))MTH_DISPATCH_TBL[func_div][sv_cv2][frp_p];
  return (fptr(x, y));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_div_2m)(vcs2_t x, vcs2_t y, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_cv2m,frp_f);
  fptr = (vcs2_t(*)(vcs2_t, vcs2_t, vis2_t))MTH_DISPATCH_TBL[func_div][sv_cv2m][frp_f];
  return (fptr(x, y, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_div_2m)(vcs2_t x, vcs2_t y, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_cv2m,frp_r);
  fptr = (vcs2_t(*)(vcs2_t, vcs2_t, vis2_t))MTH_DISPATCH_TBL[func_div][sv_cv2m][frp_r];
  return (fptr(x, y, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_div_2m)(vcs2_t x, vcs2_t y, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_cv2m,frp_p);
  fptr = (vcs2_t(*)(vcs2_t, vcs2_t, vis2_t))MTH_DISPATCH_TBL[func_div][sv_cv2m][frp_p];
  return (fptr(x, y, m));
}

double _Complex
MTH_DISPATCH_FUNC(__fz_div_1)(double _Complex x, double _Complex y)
{
  double _Complex (*fptr)(double _Complex, double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_zs,frp_f);
  fptr = (double _Complex(*)(double _Complex, double _Complex))MTH_DISPATCH_TBL[func_div][sv_zs][frp_f];
  return (fptr(x, y));
}

double _Complex
MTH_DISPATCH_FUNC(__rz_div_1)(double _Complex x, double _Complex y)
{
  double _Complex (*fptr)(double _Complex, double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_zs,frp_r);
  fptr = (double _Complex(*)(double _Complex, double _Complex))MTH_DISPATCH_TBL[func_div][sv_zs][frp_r];
  return (fptr(x, y));
}

double _Complex
MTH_DISPATCH_FUNC(__pz_div_1)(double _Complex x, double _Complex y)
{
  double _Complex (*fptr)(double _Complex, double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_zs,frp_p);
  fptr = (double _Complex(*)(double _Complex, double _Complex))MTH_DISPATCH_TBL[func_div][sv_zs][frp_p];
  return (fptr(x, y));
}

vrs4_t
MTH_DISPATCH_FUNC(__fs_sqrt_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_sv4m,frp_f);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_sqrt][sv_sv4m][frp_f];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__rs_sqrt_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_sv4m,frp_r);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_sqrt][sv_sv4m][frp_r];
  return (fptr(x, m));
}

vrs4_t
MTH_DISPATCH_FUNC(__ps_sqrt_4m)(vrs4_t x, vis4_t m)
{
  vrs4_t (*fptr)(vrs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_sv4m,frp_p);
  fptr = (vrs4_t(*)())MTH_DISPATCH_TBL[func_sqrt][sv_sv4m][frp_p];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__fd_sqrt_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_dv2m,frp_f);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_sqrt][sv_dv2m][frp_f];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__rd_sqrt_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_dv2m,frp_r);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_sqrt][sv_dv2m][frp_r];
  return (fptr(x, m));
}

vrd2_t
MTH_DISPATCH_FUNC(__pd_sqrt_2m)(vrd2_t x, vid2_t m)
{
  vrd2_t (*fptr)(vrd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_dv2m,frp_p);
  fptr = (vrd2_t(*)())MTH_DISPATCH_TBL[func_sqrt][sv_dv2m][frp_p];
  return (fptr(x, m));
}

float _Complex
MTH_DISPATCH_FUNC(__fc_sqrt_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_cs,frp_f);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_sqrt][sv_cs][frp_f];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__rc_sqrt_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_cs,frp_r);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_sqrt][sv_cs][frp_r];
  return (fptr(x));
}

float _Complex
MTH_DISPATCH_FUNC(__pc_sqrt_1)(float _Complex x)
{
  float _Complex (*fptr)(float _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_cs,frp_p);
  fptr = (float _Complex(*)(float _Complex))MTH_DISPATCH_TBL[func_sqrt][sv_cs][frp_p];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_sqrt_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_cv2,frp_f);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_sqrt][sv_cv2][frp_f];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_sqrt_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_cv2,frp_r);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_sqrt][sv_cv2][frp_r];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_sqrt_2)(vcs2_t x)
{
  vcs2_t (*fptr)(vcs2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_cv2,frp_p);
  fptr = (vcs2_t(*)(vcs2_t))MTH_DISPATCH_TBL[func_sqrt][sv_cv2][frp_p];
  return (fptr(x));
}

vcs2_t
MTH_DISPATCH_FUNC(__fc_sqrt_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_cv2m,frp_f);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_sqrt][sv_cv2m][frp_f];
  return (fptr(x, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__rc_sqrt_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_cv2m,frp_r);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_sqrt][sv_cv2m][frp_r];
  return (fptr(x, m));
}

vcs2_t
MTH_DISPATCH_FUNC(__pc_sqrt_2m)(vcs2_t x, vis2_t m)
{
  vcs2_t (*fptr)(vcs2_t, vis2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_cv2m,frp_p);
  fptr = (vcs2_t(*)())MTH_DISPATCH_TBL[func_sqrt][sv_cv2m][frp_p];
  return (fptr(x, m));
}

double _Complex
MTH_DISPATCH_FUNC(__fz_sqrt_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_zs,frp_f);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_sqrt][sv_zs][frp_f];
  return (fptr(x));
}

double _Complex
MTH_DISPATCH_FUNC(__rz_sqrt_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_zs,frp_r);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_sqrt][sv_zs][frp_r];
  return (fptr(x));
}

double _Complex
MTH_DISPATCH_FUNC(__pz_sqrt_1)(double _Complex x)
{
  double _Complex (*fptr)(double _Complex);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_zs,frp_p);
  fptr = (double _Complex(*)(double _Complex))MTH_DISPATCH_TBL[func_sqrt][sv_zs][frp_p];
  return (fptr(x));
}

/*
 * Real/_Complex passed as vectors of length 1.
 */

vcs1_t
MTH_DISPATCH_FUNC(__fc_acos_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_cv1,frp_f);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_acos][sv_cv1][frp_f];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__rc_acos_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_cv1,frp_r);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_acos][sv_cv1][frp_r];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__pc_acos_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_cv1,frp_p);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_acos][sv_cv1][frp_p];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__fz_acos_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_zv1,frp_f);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_acos][sv_zv1][frp_f];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__rz_acos_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_zv1,frp_r);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_acos][sv_zv1][frp_r];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__pz_acos_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_zv1,frp_p);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_acos][sv_zv1][frp_p];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__fc_asin_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_cv1,frp_f);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_asin][sv_cv1][frp_f];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__rc_asin_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_cv1,frp_r);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_asin][sv_cv1][frp_r];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__pc_asin_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_cv1,frp_p);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_asin][sv_cv1][frp_p];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__fz_asin_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_zv1,frp_f);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_asin][sv_zv1][frp_f];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__rz_asin_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_zv1,frp_r);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_asin][sv_zv1][frp_r];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__pz_asin_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_zv1,frp_p);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_asin][sv_zv1][frp_p];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__fc_atan_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_cv1,frp_f);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_atan][sv_cv1][frp_f];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__rc_atan_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_cv1,frp_r);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_atan][sv_cv1][frp_r];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__pc_atan_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_cv1,frp_p);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_atan][sv_cv1][frp_p];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__fz_atan_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_zv1,frp_f);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_atan][sv_zv1][frp_f];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__rz_atan_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_zv1,frp_r);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_atan][sv_zv1][frp_r];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__pz_atan_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_zv1,frp_p);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_atan][sv_zv1][frp_p];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__fc_cos_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_cv1,frp_f);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_cos][sv_cv1][frp_f];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__rc_cos_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_cv1,frp_r);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_cos][sv_cv1][frp_r];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__pc_cos_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_cv1,frp_p);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_cos][sv_cv1][frp_p];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__fz_cos_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_zv1,frp_f);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_cos][sv_zv1][frp_f];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__rz_cos_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_zv1,frp_r);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_cos][sv_zv1][frp_r];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__pz_cos_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_zv1,frp_p);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_cos][sv_zv1][frp_p];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__fc_sin_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_cv1,frp_f);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_sin][sv_cv1][frp_f];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__rc_sin_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_cv1,frp_r);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_sin][sv_cv1][frp_r];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__pc_sin_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_cv1,frp_p);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_sin][sv_cv1][frp_p];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__fz_sin_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_zv1,frp_f);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_sin][sv_zv1][frp_f];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__rz_sin_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_zv1,frp_r);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_sin][sv_zv1][frp_r];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__pz_sin_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_zv1,frp_p);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_sin][sv_zv1][frp_p];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__fc_tan_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_cv1,frp_f);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_tan][sv_cv1][frp_f];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__rc_tan_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_cv1,frp_r);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_tan][sv_cv1][frp_r];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__pc_tan_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_cv1,frp_p);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_tan][sv_cv1][frp_p];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__fz_tan_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_zv1,frp_f);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_tan][sv_zv1][frp_f];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__rz_tan_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_zv1,frp_r);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_tan][sv_zv1][frp_r];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__pz_tan_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_zv1,frp_p);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_tan][sv_zv1][frp_p];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__fc_cosh_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_cv1,frp_f);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_cosh][sv_cv1][frp_f];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__rc_cosh_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_cv1,frp_r);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_cosh][sv_cv1][frp_r];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__pc_cosh_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_cv1,frp_p);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_cosh][sv_cv1][frp_p];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__fz_cosh_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_zv1,frp_f);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_cosh][sv_zv1][frp_f];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__rz_cosh_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_zv1,frp_r);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_cosh][sv_zv1][frp_r];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__pz_cosh_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_zv1,frp_p);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_cosh][sv_zv1][frp_p];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__fc_sinh_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_cv1,frp_f);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_sinh][sv_cv1][frp_f];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__rc_sinh_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_cv1,frp_r);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_sinh][sv_cv1][frp_r];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__pc_sinh_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_cv1,frp_p);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_sinh][sv_cv1][frp_p];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__fz_sinh_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_zv1,frp_f);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_sinh][sv_zv1][frp_f];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__rz_sinh_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_zv1,frp_r);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_sinh][sv_zv1][frp_r];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__pz_sinh_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_zv1,frp_p);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_sinh][sv_zv1][frp_p];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__fc_tanh_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_cv1,frp_f);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_tanh][sv_cv1][frp_f];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__rc_tanh_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_cv1,frp_r);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_tanh][sv_cv1][frp_r];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__pc_tanh_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_cv1,frp_p);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_tanh][sv_cv1][frp_p];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__fz_tanh_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_zv1,frp_f);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_tanh][sv_zv1][frp_f];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__rz_tanh_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_zv1,frp_r);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_tanh][sv_zv1][frp_r];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__pz_tanh_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_zv1,frp_p);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_tanh][sv_zv1][frp_p];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__fc_exp_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_cv1,frp_f);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_exp][sv_cv1][frp_f];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__rc_exp_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_cv1,frp_r);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_exp][sv_cv1][frp_r];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__pc_exp_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_cv1,frp_p);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_exp][sv_cv1][frp_p];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__fz_exp_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_zv1,frp_f);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_exp][sv_zv1][frp_f];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__rz_exp_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_zv1,frp_r);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_exp][sv_zv1][frp_r];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__pz_exp_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_zv1,frp_p);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_exp][sv_zv1][frp_p];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__fc_log_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_cv1,frp_f);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_log][sv_cv1][frp_f];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__rc_log_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_cv1,frp_r);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_log][sv_cv1][frp_r];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__pc_log_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_cv1,frp_p);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_log][sv_cv1][frp_p];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__fz_log_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_zv1,frp_f);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_log][sv_zv1][frp_f];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__rz_log_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_zv1,frp_r);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_log][sv_zv1][frp_r];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__pz_log_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_zv1,frp_p);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_log][sv_zv1][frp_p];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__fc_log10_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_cv1,frp_f);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_log10][sv_cv1][frp_f];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__rc_log10_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_cv1,frp_r);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_log10][sv_cv1][frp_r];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__pc_log10_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_cv1,frp_p);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_log10][sv_cv1][frp_p];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__fz_log10_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_zv1,frp_f);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_log10][sv_zv1][frp_f];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__rz_log10_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_zv1,frp_r);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_log10][sv_zv1][frp_r];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__pz_log10_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_zv1,frp_p);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_log10][sv_zv1][frp_p];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__fc_pow_1v)(vcs1_t x, vcs1_t y)
{
  vcs1_t (*fptr)(vcs1_t, vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_cv1,frp_f);
  fptr = (vcs1_t(*)(vcs1_t, vcs1_t))MTH_DISPATCH_TBL[func_pow][sv_cv1][frp_f];
  return (fptr(x, y));
}

vcs1_t
MTH_DISPATCH_FUNC(__rc_pow_1v)(vcs1_t x, vcs1_t y)
{
  vcs1_t (*fptr)(vcs1_t, vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_cv1,frp_r);
  fptr = (vcs1_t(*)(vcs1_t, vcs1_t))MTH_DISPATCH_TBL[func_pow][sv_cv1][frp_r];
  return (fptr(x, y));
}

vcs1_t
MTH_DISPATCH_FUNC(__pc_pow_1v)(vcs1_t x, vcs1_t y)
{
  vcs1_t (*fptr)(vcs1_t, vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_cv1,frp_p);
  fptr = (vcs1_t(*)(vcs1_t, vcs1_t))MTH_DISPATCH_TBL[func_pow][sv_cv1][frp_p];
  return (fptr(x, y));
}

vcd1_t
MTH_DISPATCH_FUNC(__fz_pow_1v)(vcd1_t x, vcd1_t y)
{
  vcd1_t (*fptr)(vcd1_t, vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_zv1,frp_f);
  fptr = (vcd1_t(*)(vcd1_t, vcd1_t))MTH_DISPATCH_TBL[func_pow][sv_zv1][frp_f];
  return (fptr(x, y));
}

vcd1_t
MTH_DISPATCH_FUNC(__rz_pow_1v)(vcd1_t x, vcd1_t y)
{
  vcd1_t (*fptr)(vcd1_t, vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_zv1,frp_r);
  fptr = (vcd1_t(*)(vcd1_t, vcd1_t))MTH_DISPATCH_TBL[func_pow][sv_zv1][frp_r];
  return (fptr(x, y));
}

vcd1_t
MTH_DISPATCH_FUNC(__pz_pow_1v)(vcd1_t x, vcd1_t y)
{
  vcd1_t (*fptr)(vcd1_t, vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_zv1,frp_p);
  fptr = (vcd1_t(*)(vcd1_t, vcd1_t))MTH_DISPATCH_TBL[func_pow][sv_zv1][frp_p];
  return (fptr(x, y));
}

vcs1_t
MTH_DISPATCH_FUNC(__fc_powi_1v)(vcs1_t x, int iy)
{
  vcs1_t (*fptr)(vcs1_t, int);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_cv1,frp_f);
  fptr = (vcs1_t(*)(vcs1_t, int))MTH_DISPATCH_TBL[func_powi][sv_cv1][frp_f];
  return (fptr(x, iy));
}

vcs1_t
MTH_DISPATCH_FUNC(__rc_powi_1v)(vcs1_t x, int iy)
{
  vcs1_t (*fptr)(vcs1_t, int);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_cv1,frp_r);
  fptr = (vcs1_t(*)(vcs1_t, int))MTH_DISPATCH_TBL[func_powi][sv_cv1][frp_r];
  return (fptr(x, iy));
}

vcs1_t
MTH_DISPATCH_FUNC(__pc_powi_1v)(vcs1_t x, int iy)
{
  vcs1_t (*fptr)(vcs1_t, int);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_cv1,frp_p);
  fptr = (vcs1_t(*)(vcs1_t, int))MTH_DISPATCH_TBL[func_powi][sv_cv1][frp_p];
  return (fptr(x, iy));
}

vcs1_t
MTH_DISPATCH_FUNC(__fc_powk_1v)(vcs1_t x, long long iy)
{
  vcs1_t (*fptr)(vcs1_t, long long);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_cv1,frp_f);
  fptr = (vcs1_t(*)(vcs1_t, long long))MTH_DISPATCH_TBL[func_powk][sv_cv1][frp_f];
  return (fptr(x, iy));
}

vcs1_t
MTH_DISPATCH_FUNC(__rc_powk_1v)(vcs1_t x, long long iy)
{
  vcs1_t (*fptr)(vcs1_t, long long);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_cv1,frp_r);
  fptr = (vcs1_t(*)(vcs1_t, long long))MTH_DISPATCH_TBL[func_powk][sv_cv1][frp_r];
  return (fptr(x, iy));
}

vcs1_t
MTH_DISPATCH_FUNC(__pc_powk_1v)(vcs1_t x, long long iy)
{
  vcs1_t (*fptr)(vcs1_t, long long);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_cv1,frp_p);
  fptr = (vcs1_t(*)(vcs1_t, long long))MTH_DISPATCH_TBL[func_powk][sv_cv1][frp_p];
  return (fptr(x, iy));
}

vcd1_t
MTH_DISPATCH_FUNC(__fz_powi_1v)(vcd1_t x, int iy)
{
  vcd1_t (*fptr)(vcd1_t, int);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_zv1,frp_f);
  fptr = (vcd1_t(*)(vcd1_t, int))MTH_DISPATCH_TBL[func_powi][sv_zv1][frp_f];
  return (fptr(x, iy));
}

vcd1_t
MTH_DISPATCH_FUNC(__rz_powi_1v)(vcd1_t x, int iy)
{
  vcd1_t (*fptr)(vcd1_t, int);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_zv1,frp_r);
  fptr = (vcd1_t(*)(vcd1_t, int))MTH_DISPATCH_TBL[func_powi][sv_zv1][frp_r];
  return (fptr(x, iy));
}

vcd1_t
MTH_DISPATCH_FUNC(__pz_powi_1v)(vcd1_t x, int iy)
{
  vcd1_t (*fptr)(vcd1_t, int);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_zv1,frp_p);
  fptr = (vcd1_t(*)(vcd1_t, int))MTH_DISPATCH_TBL[func_powi][sv_zv1][frp_p];
  return (fptr(x, iy));
}

vcd1_t
MTH_DISPATCH_FUNC(__fz_powk_1v)(vcd1_t x, long long iy)
{
  vcd1_t (*fptr)(vcd1_t, long long);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_zv1,frp_f);
  fptr = (vcd1_t(*)(vcd1_t, long long))MTH_DISPATCH_TBL[func_powk][sv_zv1][frp_f];
  return (fptr(x, iy));
}

vcd1_t
MTH_DISPATCH_FUNC(__rz_powk_1v)(vcd1_t x, long long iy)
{
  vcd1_t (*fptr)(vcd1_t, long long);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_zv1,frp_r);
  fptr = (vcd1_t(*)(vcd1_t, long long))MTH_DISPATCH_TBL[func_powk][sv_zv1][frp_r];
  return (fptr(x, iy));
}

vcd1_t
MTH_DISPATCH_FUNC(__pz_powk_1v)(vcd1_t x, long long iy)
{
  vcd1_t (*fptr)(vcd1_t, long long);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_zv1,frp_p);
  fptr = (vcd1_t(*)(vcd1_t, long long))MTH_DISPATCH_TBL[func_powk][sv_zv1][frp_p];
  return (fptr(x, iy));
}

vcs1_t
MTH_DISPATCH_FUNC(__fc_div_1v)(vcs1_t x, vcs1_t y)
{
  vcs1_t (*fptr)(vcs1_t, vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_cv1,frp_f);
  fptr = (vcs1_t(*)(vcs1_t, vcs1_t))MTH_DISPATCH_TBL[func_div][sv_cv1][frp_f];
  return (fptr(x, y));
}

vcs1_t
MTH_DISPATCH_FUNC(__rc_div_1v)(vcs1_t x, vcs1_t y)
{
  vcs1_t (*fptr)(vcs1_t, vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_cv1,frp_r);
  fptr = (vcs1_t(*)(vcs1_t, vcs1_t))MTH_DISPATCH_TBL[func_div][sv_cv1][frp_r];
  return (fptr(x, y));
}

vcs1_t
MTH_DISPATCH_FUNC(__pc_div_1v)(vcs1_t x, vcs1_t y)
{
  vcs1_t (*fptr)(vcs1_t, vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_cv1,frp_p);
  fptr = (vcs1_t(*)(vcs1_t, vcs1_t))MTH_DISPATCH_TBL[func_div][sv_cv1][frp_p];
  return (fptr(x, y));
}

vcd1_t
MTH_DISPATCH_FUNC(__fz_div_1v)(vcd1_t x, vcd1_t y)
{
  vcd1_t (*fptr)(vcd1_t, vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_zv1,frp_f);
  fptr = (vcd1_t(*)(vcd1_t, vcd1_t))MTH_DISPATCH_TBL[func_div][sv_zv1][frp_f];
  return (fptr(x, y));
}

vcd1_t
MTH_DISPATCH_FUNC(__rz_div_1v)(vcd1_t x, vcd1_t y)
{
  vcd1_t (*fptr)(vcd1_t, vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_zv1,frp_r);
  fptr = (vcd1_t(*)(vcd1_t, vcd1_t))MTH_DISPATCH_TBL[func_div][sv_zv1][frp_r];
  return (fptr(x, y));
}

vcd1_t
MTH_DISPATCH_FUNC(__pz_div_1v)(vcd1_t x, vcd1_t y)
{
  vcd1_t (*fptr)(vcd1_t, vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_zv1,frp_p);
  fptr = (vcd1_t(*)(vcd1_t, vcd1_t))MTH_DISPATCH_TBL[func_div][sv_zv1][frp_p];
  return (fptr(x, y));
}

vcs1_t
MTH_DISPATCH_FUNC(__fc_sqrt_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_cv1,frp_f);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_sqrt][sv_cv1][frp_f];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__rc_sqrt_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_cv1,frp_r);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_sqrt][sv_cv1][frp_r];
  return (fptr(x));
}

vcs1_t
MTH_DISPATCH_FUNC(__pc_sqrt_1v)(vcs1_t x)
{
  vcs1_t (*fptr)(vcs1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_cv1,frp_p);
  fptr = (vcs1_t(*)(vcs1_t))MTH_DISPATCH_TBL[func_sqrt][sv_cv1][frp_p];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__fz_sqrt_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_zv1,frp_f);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_sqrt][sv_zv1][frp_f];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__rz_sqrt_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_zv1,frp_r);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_sqrt][sv_zv1][frp_r];
  return (fptr(x));
}

vcd1_t
MTH_DISPATCH_FUNC(__pz_sqrt_1v)(vcd1_t x)
{
  vcd1_t (*fptr)(vcd1_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_zv1,frp_p);
  fptr = (vcd1_t(*)(vcd1_t))MTH_DISPATCH_TBL[func_sqrt][sv_zv1][frp_p];
  return (fptr(x));
}

//////////
// EXPERIMENTAL - _Complex - end
//////////

#if     defined(TARGET_LINUX_X8664) && ! defined(MTH_I_INTRIN_STATS) && ! defined(MTH_I_INTRIN_INIT)
vrd1_t __gsd_atan(vrd1_t) __attribute__ ((weak, alias ("__fd_atan_1")));
vrd2_t __gvd_atan2(vrd2_t) __attribute__ ((weak, alias ("__fd_atan_2")));
vrd2_t __gvd_atan2_mask(vrd2_t,vid2_t) __attribute__ ((weak, alias ("__fd_atan_2m")));
vrs1_t __gss_atan(vrs1_t) __attribute__ ((weak, alias ("__fs_atan_1")));
vrs4_t __gvs_atan4(vrs4_t) __attribute__ ((weak, alias ("__fs_atan_4")));
vrs4_t __gvs_atan4_mask(vrs4_t,vis4_t) __attribute__ ((weak, alias ("__fs_atan_4m")));
vrd1_t __gsd_exp(vrd1_t) __attribute__ ((weak, alias ("__fd_exp_1")));
vrd2_t __gvd_exp2(vrd2_t) __attribute__ ((weak, alias ("__fd_exp_2")));
vrd2_t __gvd_exp2_mask(vrd2_t,vid2_t) __attribute__ ((weak, alias ("__fd_exp_2m")));
vrs1_t __gss_exp(vrs1_t) __attribute__ ((weak, alias ("__fs_exp_1")));
vrs4_t __gvs_exp4(vrs4_t) __attribute__ ((weak, alias ("__fs_exp_4")));
vrs4_t __gvs_exp4_mask(vrs4_t,vis4_t) __attribute__ ((weak, alias ("__fs_exp_4m")));
vrd1_t __gsd_log(vrd1_t) __attribute__ ((weak, alias ("__fd_log_1")));
vrd2_t __gvd_log2(vrd2_t) __attribute__ ((weak, alias ("__fd_log_2")));
vrd2_t __gvd_log2_mask(vrd2_t,vid2_t) __attribute__ ((weak, alias ("__fd_log_2m")));
vrs1_t __gss_log(vrs1_t) __attribute__ ((weak, alias ("__fs_log_1")));
vrs4_t __gvs_log4(vrs4_t) __attribute__ ((weak, alias ("__fs_log_4")));
vrs4_t __gvs_log4_mask(vrs4_t,vis4_t) __attribute__ ((weak, alias ("__fs_log_4m")));
vrd1_t __gsd_pow(vrd1_t,vrd1_t) __attribute__ ((weak, alias ("__fd_pow_1")));
vrd2_t __gvd_pow2(vrd2_t,vrd2_t) __attribute__ ((weak, alias ("__fd_pow_2")));
vrd2_t __gvd_pow2_mask(vrd2_t,vrd2_t,vid2_t) __attribute__ ((weak, alias ("__fd_pow_2m")));
vrs1_t __gss_pow(vrs1_t,vrs1_t) __attribute__ ((weak, alias ("__fs_pow_1")));
vrs4_t __gvs_pow4(vrs4_t,vrs4_t) __attribute__ ((weak, alias ("__fs_pow_4")));
vrs4_t __gvs_pow4_mask(vrs4_t,vrs4_t,vis4_t) __attribute__ ((weak, alias ("__fs_pow_4m")));
#endif

#if     defined(TARGET_LINUX_POWER) && ! defined(MTH_I_INTRIN_STATS) && ! defined(MTH_I_INTRIN_INIT)
vrs1_t __gss_atan(vrs1_t) __attribute__ ((weak, alias ("__fs_atan_1")));
vrd1_t __gsd_atan(vrd1_t) __attribute__ ((weak, alias ("__fd_atan_1")));
vrs1_t __gss_cos(vrs1_t) __attribute__ ((weak, alias ("__fs_cos_1")));
vrd1_t __gsd_cos(vrd1_t) __attribute__ ((weak, alias ("__fd_cos_1")));
vrs1_t __gss_sin(vrs1_t) __attribute__ ((weak, alias ("__fs_sin_1")));
vrd1_t __gsd_sin(vrd1_t) __attribute__ ((weak, alias ("__fd_sin_1")));
vrs1_t __gss_tan(vrs1_t) __attribute__ ((weak, alias ("__fs_tan_1")));
vrd1_t __gsd_tan(vrd1_t) __attribute__ ((weak, alias ("__fd_tan_1")));
vrs1_t __gss_exp(vrs1_t) __attribute__ ((weak, alias ("__fs_exp_1")));
vrd1_t __gsd_exp(vrd1_t) __attribute__ ((weak, alias ("__fd_exp_1")));
vrs1_t __gss_log(vrs1_t) __attribute__ ((weak, alias ("__fs_log_1")));
vrd1_t __gsd_log(vrd1_t) __attribute__ ((weak, alias ("__fd_log_1")));
vrs1_t __gss_pow(vrs1_t,vrs1_t) __attribute__ ((weak, alias ("__fs_pow_1")));
vrd1_t __gsd_pow(vrd1_t,vrd1_t) __attribute__ ((weak, alias ("__fd_pow_1")));

vrs4_t __gvs_atan4(vrs4_t) __attribute__ ((weak, alias ("__fs_atan_4")));
vrd2_t __gvd_atan2(vrd2_t) __attribute__ ((weak, alias ("__fd_atan_2")));
vrs4_t __gvs_cos4(vrs4_t) __attribute__ ((weak, alias ("__fs_cos_4")));
vrd2_t __gvd_cos2(vrd2_t) __attribute__ ((weak, alias ("__fd_cos_2")));
vrs4_t __gvs_sin4(vrs4_t) __attribute__ ((weak, alias ("__fs_sin_4")));
vrd2_t __gvd_sin2(vrd2_t) __attribute__ ((weak, alias ("__fd_sin_2")));
vrs4_t __gvs_tan4(vrs4_t) __attribute__ ((weak, alias ("__fs_tan_4")));
vrd2_t __gvd_tan2(vrd2_t) __attribute__ ((weak, alias ("__fd_tan_2")));
vrs4_t __gvs_exp4(vrs4_t) __attribute__ ((weak, alias ("__fs_exp_4")));
vrd2_t __gvd_exp2(vrd2_t) __attribute__ ((weak, alias ("__fd_exp_2")));
vrs4_t __gvs_log4(vrs4_t) __attribute__ ((weak, alias ("__fs_log_4")));
vrd2_t __gvd_log2(vrd2_t) __attribute__ ((weak, alias ("__fd_log_2")));
vrs4_t __gvs_pow4(vrs4_t,vrs4_t) __attribute__ ((weak, alias ("__fs_pow_4")));
vrd2_t __gvd_pow2(vrd2_t,vrd2_t) __attribute__ ((weak, alias ("__fd_pow_2")));

#endif

#if defined(TARGET_LINUX_GENERIC) && !defined(MTH_I_INTRIN_STATS) && ! defined(MTH_I_INTRIN_INIT)
vrs1_t __gss_atan(vrs1_t) __attribute__ ((weak, alias ("__fs_atan_1")));
vrd1_t __gsd_atan(vrd1_t) __attribute__ ((weak, alias ("__fd_atan_1")));
vrs1_t __gss_cos(vrs1_t) __attribute__ ((weak, alias ("__fs_cos_1")));
vrd1_t __gsd_cos(vrd1_t) __attribute__ ((weak, alias ("__fd_cos_1")));
vrs1_t __gss_sin(vrs1_t) __attribute__ ((weak, alias ("__fs_sin_1")));
vrd1_t __gsd_sin(vrd1_t) __attribute__ ((weak, alias ("__fd_sin_1")));
vrs1_t __gss_tan(vrs1_t) __attribute__ ((weak, alias ("__fs_tan_1")));
vrd1_t __gsd_tan(vrd1_t) __attribute__ ((weak, alias ("__fd_tan_1")));
vrs1_t __gss_exp(vrs1_t) __attribute__ ((weak, alias ("__fs_exp_1")));
vrd1_t __gsd_exp(vrd1_t) __attribute__ ((weak, alias ("__fd_exp_1")));
vrs1_t __gss_log(vrs1_t) __attribute__ ((weak, alias ("__fs_log_1")));
vrd1_t __gsd_log(vrd1_t) __attribute__ ((weak, alias ("__fd_log_1")));
vrs1_t __gss_pow(vrs1_t,vrs1_t) __attribute__ ((weak, alias ("__fs_pow_1")));
vrd1_t __gsd_pow(vrd1_t,vrd1_t) __attribute__ ((weak, alias ("__fd_pow_1")));

vrs4_t __gvs_atan4(vrs4_t) __attribute__ ((weak, alias ("__fs_atan_4")));
vrd2_t __gvd_atan2(vrd2_t) __attribute__ ((weak, alias ("__fd_atan_2")));
vrs4_t __gvs_cos4(vrs4_t) __attribute__ ((weak, alias ("__fs_cos_4")));
vrd2_t __gvd_cos2(vrd2_t) __attribute__ ((weak, alias ("__fd_cos_2")));
vrs4_t __gvs_sin4(vrs4_t) __attribute__ ((weak, alias ("__fs_sin_4")));
vrd2_t __gvd_sin2(vrd2_t) __attribute__ ((weak, alias ("__fd_sin_2")));
vrs4_t __gvs_tan4(vrs4_t) __attribute__ ((weak, alias ("__fs_tan_4")));
vrd2_t __gvd_tan2(vrd2_t) __attribute__ ((weak, alias ("__fd_tan_2")));
vrs4_t __gvs_exp4(vrs4_t) __attribute__ ((weak, alias ("__fs_exp_4")));
vrd2_t __gvd_exp2(vrd2_t) __attribute__ ((weak, alias ("__fd_exp_2")));
vrs4_t __gvs_log4(vrs4_t) __attribute__ ((weak, alias ("__fs_log_4")));
vrd2_t __gvd_log2(vrd2_t) __attribute__ ((weak, alias ("__fd_log_2")));
vrs4_t __gvs_pow4(vrs4_t,vrs4_t) __attribute__ ((weak, alias ("__fs_pow_4")));
vrd2_t __gvd_pow2(vrd2_t,vrd2_t) __attribute__ ((weak, alias ("__fd_pow_2")));
#endif
