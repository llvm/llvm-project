/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "stdioInterf.h"
#include "fioMacros.h"
#include "scatter.h"

/* local scatter functions */

static void
local_scatter_INT1(int n, void *dstp, int *sv, void *srcp)
{
  __INT1_T *dst = (__INT1_T *)dstp;
  __INT1_T *src = (__INT1_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[i];
}

static void
local_scatter_INT2(int n, void *dstp, int *sv, void *srcp)
{
  __INT2_T *dst = (__INT2_T *)dstp;
  __INT2_T *src = (__INT2_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[i];
}

static void
local_scatter_INT4(int n, void *dstp, int *sv, void *srcp)
{
  __INT4_T *dst = (__INT4_T *)dstp;
  __INT4_T *src = (__INT4_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[i];
}

static void
local_scatter_INT8(int n, void *dstp, int *sv, void *srcp)
{
  __INT8_T *dst = (__INT8_T *)dstp;
  __INT8_T *src = (__INT8_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[i];
}

static void
local_scatter_LOG1(int n, void *dstp, int *sv, void *srcp)
{
  __LOG1_T *dst = (__LOG1_T *)dstp;
  __LOG1_T *src = (__LOG1_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[i];
}

static void
local_scatter_LOG2(int n, void *dstp, int *sv, void *srcp)
{
  __LOG2_T *dst = (__LOG2_T *)dstp;
  __LOG2_T *src = (__LOG2_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[i];
}

static void
local_scatter_LOG4(int n, void *dstp, int *sv, void *srcp)
{
  __LOG4_T *dst = (__LOG4_T *)dstp;
  __LOG4_T *src = (__LOG4_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[i];
}

static void
local_scatter_LOG8(int n, void *dstp, int *sv, void *srcp)
{
  __LOG8_T *dst = (__LOG8_T *)dstp;
  __LOG8_T *src = (__LOG8_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[i];
}

static void
local_scatter_REAL4(int n, void *dstp, int *sv, void *srcp)
{
  __REAL4_T *dst = (__REAL4_T *)dstp;
  __REAL4_T *src = (__REAL4_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[i];
}

static void
local_scatter_REAL8(int n, void *dstp, int *sv, void *srcp)
{
  __REAL8_T *dst = (__REAL8_T *)dstp;
  __REAL8_T *src = (__REAL8_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[i];
}

static void
local_scatter_REAL16(int n, void *dstp, int *sv, void *srcp)
{
  __REAL16_T *dst = (__REAL16_T *)dstp;
  __REAL16_T *src = (__REAL16_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[i];
}

static void
local_scatter_CPLX8(int n, void *dstp, int *sv, void *srcp)
{
  __CPLX8_T *dst = (__CPLX8_T *)dstp;
  __CPLX8_T *src = (__CPLX8_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[i];
}

static void
local_scatter_CPLX16(int n, void *dstp, int *sv, void *srcp)
{
  __CPLX16_T *dst = (__CPLX16_T *)dstp;
  __CPLX16_T *src = (__CPLX16_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[i];
}

static void
local_scatter_CPLX32(int n, void *dstp, int *sv, void *srcp)
{
  __CPLX32_T *dst = (__CPLX32_T *)dstp;
  __CPLX32_T *src = (__CPLX32_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[i];
}

static scatterfn_t __fort_local_scatter[__NTYPES] = {
    NULL,                 /*     no type (absent optional argument) */
    NULL,                 /* C   signed short */
    NULL,                 /* C   unsigned short */
    NULL,                 /* C   signed int */
    NULL,                 /* C   unsigned int */
    NULL,                 /* C   signed long int */
    NULL,                 /* C   unsigned long int */
    NULL,                 /* C   float */
    NULL,                 /* C   double */
    local_scatter_CPLX8,  /*   F complex*8 (2x real*4) */
    local_scatter_CPLX16, /*   F complex*16 (2x real*8) */
    NULL,                 /* C   signed char */
    NULL,                 /* C   unsigned char */
    NULL,                 /* C   long double */
    NULL,                 /*   F character */
    NULL,                 /* C   long long */
    NULL,                 /* C   unsigned long long */
    local_scatter_LOG1,   /*   F logical*1 */
    local_scatter_LOG2,   /*   F logical*2 */
    local_scatter_LOG4,   /*   F logical*4 */
    local_scatter_LOG8,   /*   F logical*8 */
    NULL,                 /*   F typeless */
    NULL,                 /*   F double typeless */
    NULL,                 /*   F ncharacter - kanji */
    local_scatter_INT2,   /*   F integer*2 */
    local_scatter_INT4,   /*   F integer*4, integer */
    local_scatter_INT8,   /*   F integer*8 */
    local_scatter_REAL4,  /*   F real*4, real */
    local_scatter_REAL8,  /*   F real*8, double precision */
    local_scatter_REAL16, /*   F real*16 */
    local_scatter_CPLX32, /*   F complex*32 (2x real*16) */
    NULL,                 /*   F quad typeless */
    local_scatter_INT1,   /*   F integer*1 */
    NULL                  /*   F derived type */
};

void
local_scatter_WRAPPER(int n, void *dst, int *sv, void *src, __INT_T kind)
{
  __fort_local_scatter[kind](n, dst, sv, src);
}

/* local gather-scatter functions */

static void
local_gathscat_INT1(int n, void *dstp, int *sv, void *srcp, int *gv)
{
  __INT1_T *dst = (__INT1_T *)dstp;
  __INT1_T *src = (__INT1_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[gv[i]];
}

static void
local_gathscat_INT2(int n, void *dstp, int *sv, void *srcp, int *gv)
{
  __INT2_T *dst = (__INT2_T *)dstp;
  __INT2_T *src = (__INT2_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[gv[i]];
}

static void
local_gathscat_INT4(int n, void *dstp, int *sv, void *srcp, int *gv)
{
  __INT4_T *dst = (__INT4_T *)dstp;
  __INT4_T *src = (__INT4_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[gv[i]];
}

static void
local_gathscat_INT8(int n, void *dstp, int *sv, void *srcp, int *gv)
{
  __INT8_T *dst = (__INT8_T *)dstp;
  __INT8_T *src = (__INT8_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[gv[i]];
}

static void
local_gathscat_LOG1(int n, void *dstp, int *sv, void *srcp, int *gv)
{
  __LOG1_T *dst = (__LOG1_T *)dstp;
  __LOG1_T *src = (__LOG1_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[gv[i]];
}

static void
local_gathscat_LOG2(int n, void *dstp, int *sv, void *srcp, int *gv)
{
  __LOG2_T *dst = (__LOG2_T *)dstp;
  __LOG2_T *src = (__LOG2_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[gv[i]];
}

static void
local_gathscat_LOG4(int n, void *dstp, int *sv, void *srcp, int *gv)
{
  __LOG4_T *dst = (__LOG4_T *)dstp;
  __LOG4_T *src = (__LOG4_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[gv[i]];
}

static void
local_gathscat_LOG8(int n, void *dstp, int *sv, void *srcp, int *gv)
{
  __LOG8_T *dst = (__LOG8_T *)dstp;
  __LOG8_T *src = (__LOG8_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[gv[i]];
}

static void
local_gathscat_REAL4(int n, void *dstp, int *sv, void *srcp, int *gv)
{
  __REAL4_T *dst = (__REAL4_T *)dstp;
  __REAL4_T *src = (__REAL4_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[gv[i]];
}

static void
local_gathscat_REAL8(int n, void *dstp, int *sv, void *srcp, int *gv)
{
  __REAL8_T *dst = (__REAL8_T *)dstp;
  __REAL8_T *src = (__REAL8_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[gv[i]];
}

static void
local_gathscat_REAL16(int n, void *dstp, int *sv, void *srcp, int *gv)
{
  __REAL16_T *dst = (__REAL16_T *)dstp;
  __REAL16_T *src = (__REAL16_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[gv[i]];
}

static void
local_gathscat_CPLX8(int n, void *dstp, int *sv, void *srcp, int *gv)
{
  __CPLX8_T *dst = (__CPLX8_T *)dstp;
  __CPLX8_T *src = (__CPLX8_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[gv[i]];
}

static void
local_gathscat_CPLX16(int n, void *dstp, int *sv, void *srcp, int *gv)
{
  __CPLX16_T *dst = (__CPLX16_T *)dstp;
  __CPLX16_T *src = (__CPLX16_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[gv[i]];
}

static void
local_gathscat_CPLX32(int n, void *dstp, int *sv, void *srcp, int *gv)
{
  __CPLX32_T *dst = (__CPLX32_T *)dstp;
  __CPLX32_T *src = (__CPLX32_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[sv[i]] = src[gv[i]];
}

static gathscatfn_t __fort_local_gathscat[__NTYPES] = {
    NULL,                  /*     no type (absent optional argument) */
    NULL,                  /* C   signed short */
    NULL,                  /* C   unsigned short */
    NULL,                  /* C   signed int */
    NULL,                  /* C   unsigned int */
    NULL,                  /* C   signed long int */
    NULL,                  /* C   unsigned long int */
    NULL,                  /* C   float */
    NULL,                  /* C   double */
    local_gathscat_CPLX8,  /*   F complex*8 (2x real*4) */
    local_gathscat_CPLX16, /*   F complex*16 (2x real*8) */
    NULL,                  /* C   signed char */
    NULL,                  /* C   unsigned char */
    NULL,                  /* C   long double */
    NULL,                  /*   F character */
    NULL,                  /* C   long long */
    NULL,                  /* C   unsigned long long */
    local_gathscat_LOG1,   /*   F logical*1 */
    local_gathscat_LOG2,   /*   F logical*2 */
    local_gathscat_LOG4,   /*   F logical*4 */
    local_gathscat_LOG8,   /*   F logical*8 */
    NULL,                  /*   F typeless */
    NULL,                  /*   F double typeless */
    NULL,                  /*   F ncharacter - kanji */
    local_gathscat_INT2,   /*   F integer*2 */
    local_gathscat_INT4,   /*   F integer*4, integer */
    local_gathscat_INT8,   /*   F integer*8 */
    local_gathscat_REAL4,  /*   F real*4, real */
    local_gathscat_REAL8,  /*   F real*8, double precision */
    local_gathscat_REAL16, /*   F real*16 */
    local_gathscat_CPLX32, /*   F complex*32 (2x real*16) */
    NULL,                  /*   F quad typeless */
    local_gathscat_INT1,   /*   F integer*1 */
    NULL                   /*   F derived type */
};

void
local_gathscat_WRAPPER(int n, void *dst, int *sv, void *src, int *gv,
                       __INT_T kind)
{
  __fort_local_gathscat[kind](n, dst, sv, src, gv);
}
