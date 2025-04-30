/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#if defined(TARGET_ARM64)
#error "gsincos.c is not used on AArch64."
#endif

#include "mth_intrinsics.h"

#if defined(TARGET_LINUX_POWER)
#include <altivec.h>
#elif   defined(LINUX8664) || defined(TARGET_OSX_X8664) || defined(TARGET_WIN_X8664)
#include <immintrin.h>
#endif


#if PRECSIZE == 4
#define PREC    s
#define FLOAT   float
#else
#define PREC    d
#define FLOAT   double
#endif

#define CONCAT4_(a,b,c,d) a##b##c##d
#define CONCAT4(a,b,c,d) CONCAT4_(a,b,c,d)
#define CONCAT5_(a,b,c,d,e) a##b##c##d##e
#define CONCAT5(a,b,c,d,e) CONCAT5_(a,b,c,d,e)
#define EXPAND(a) a
#define VFLOAT  CONCAT4(vr,PREC,VLEN,_t)
#define VINT    CONCAT4(vi,PREC,VLEN,_t)

#if     ! defined(TARGET_OSX_X8664) && ! defined(TARGET_WIN_X8664)
#define VFLOATRETURN    CONCAT4(__mth_return2,VFLOAT,,)
#else
/*
 * OSX and Windows do not support weak aliases - so just use the generic
 * for all vector types.
 */
#define VFLOATRETURN    __mth_return2vectors
#endif
#define GENERICNAME     CONCAT4(__g,PREC,_sincos_,VLEN)
#define GENERICNAMEMASK CONCAT5(__g,PREC,_sincos_,VLEN,m)

#if     defined(LINUX8664) || defined(TARGET_OSX_X8664) || defined(TARGET_WIN_X8664)
  #define _s_VL_4 
  #define _d_VL_2 
  #define _s_VL_8 256
  #define _d_VL_4 256
  #define _s_VL_16 512
  #define _d_VL_8 512
  #define __VLSIZE(_prec,_vlen) _##_prec##_VL_##_vlen
  #define _VLSIZE(_prec,_vlen) __VLSIZE(_prec,_vlen)
  #define VEC_LOAD(_a)  (VFLOAT)CONCAT4(_mm,_VLSIZE(PREC,VLEN),_load_p,PREC)((FLOAT *)_a)
#elif   defined(TARGET_LINUX_POWER)
  /*
   * POWER intrinsic does note seems to accept (double *) as an address in vec_ld().
   * Thus make the argument always look like a (float *).
   */
  #define VEC_LOAD(_a)    (VFLOAT)vec_ld(0, (float *)_a)
#else
  #define VEC_LOAD(_a)    *((VFLOAT *)(_a))
#endif

#if     ! defined(TARGET_WIN) && ! defined(TARGET_OSX_X8664)
extern  void   SINCOS(FLOAT, FLOAT *, FLOAT *);
#endif      /* #if     ! defined(TARGET_WIN_X8664) */
extern  VFLOAT  VFLOATRETURN(VFLOAT, VFLOAT);

VFLOAT
GENERICNAME(VFLOAT x)
{
  int i;
  FLOAT ts[VLEN] __attribute__((aligned((VLEN*sizeof(FLOAT)))));
  FLOAT tc[VLEN] __attribute__((aligned((VLEN*sizeof(FLOAT)))));

  for (i = 0 ; i < VLEN; i++)
    SINCOS(x[i], &ts[i], &tc[i]);

  return VFLOATRETURN(VEC_LOAD(&ts), VEC_LOAD(&tc));
}

VFLOAT
GENERICNAMEMASK(VFLOAT x, VINT mask)
{
  int i;
  FLOAT ts[VLEN] __attribute__((aligned((VLEN*sizeof(FLOAT)))));
  FLOAT tc[VLEN] __attribute__((aligned((VLEN*sizeof(FLOAT)))));

  for (i = 0 ; i < VLEN; i++) {
    if (mask[i] != 0) {
        SINCOS(x[i], &ts[i], &tc[i]);
    }
  }
  return VFLOATRETURN(VEC_LOAD(&ts), VEC_LOAD(&tc));
}
