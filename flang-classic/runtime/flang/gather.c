/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "stdioInterf.h"
#include "fioMacros.h"
#include "scatter.h"

/* local gather functions */

static void
local_gather_INT1(int n, void *dstp, void *srcp, int *gv)
{
  __INT1_T *dst = (__INT1_T *)dstp;
  __INT1_T *src = (__INT1_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[i] = src[gv[i]];
}

static void
local_gather_INT2(int n, void *dstp, void *srcp, int *gv)
{
  __INT2_T *dst = (__INT2_T *)dstp;
  __INT2_T *src = (__INT2_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[i] = src[gv[i]];
}

static void
local_gather_INT4(int n, void *dstp, void *srcp, int *gv)
{
  __INT4_T *dst = (__INT4_T *)dstp;
  __INT4_T *src = (__INT4_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[i] = src[gv[i]];
}

static void
local_gather_INT8(int n, void *dstp, void *srcp, int *gv)
{
  __INT8_T *dst = (__INT8_T *)dstp;
  __INT8_T *src = (__INT8_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[i] = src[gv[i]];
}

static void
local_gather_LOG1(int n, void *dstp, void *srcp, int *gv)
{
  __LOG1_T *dst = (__LOG1_T *)dstp;
  __LOG1_T *src = (__LOG1_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[i] = src[gv[i]];
}

static void
local_gather_LOG2(int n, void *dstp, void *srcp, int *gv)
{
  __LOG2_T *dst = (__LOG2_T *)dstp;
  __LOG2_T *src = (__LOG2_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[i] = src[gv[i]];
}

static void
local_gather_LOG4(int n, void *dstp, void *srcp, int *gv)
{
  __LOG4_T *dst = (__LOG4_T *)dstp;
  __LOG4_T *src = (__LOG4_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[i] = src[gv[i]];
}

static void
local_gather_LOG8(int n, void *dstp, void *srcp, int *gv)
{
  __LOG8_T *dst = (__LOG8_T *)dstp;
  __LOG8_T *src = (__LOG8_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[i] = src[gv[i]];
}

static void
local_gather_REAL4(int n, void *dstp, void *srcp, int *gv)
{
  __REAL4_T *dst = (__REAL4_T *)dstp;
  __REAL4_T *src = (__REAL4_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[i] = src[gv[i]];
}

static void
local_gather_REAL8(int n, void *dstp, void *srcp, int *gv)
{
  __REAL8_T *dst = (__REAL8_T *)dstp;
  __REAL8_T *src = (__REAL8_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[i] = src[gv[i]];
}

static void
local_gather_REAL16(int n, void *dstp, void *srcp, int *gv)
{
  __REAL16_T *dst = (__REAL16_T *)dstp;
  __REAL16_T *src = (__REAL16_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[i] = src[gv[i]];
}

static void
local_gather_CPLX8(int n, void *dstp, void *srcp, int *gv)
{
  __CPLX8_T *dst = (__CPLX8_T *)dstp;
  __CPLX8_T *src = (__CPLX8_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[i] = src[gv[i]];
}

static void
local_gather_CPLX16(int n, void *dstp, void *srcp, int *gv)
{
  __CPLX16_T *dst = (__CPLX16_T *)dstp;
  __CPLX16_T *src = (__CPLX16_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[i] = src[gv[i]];
}

static void
local_gather_CPLX32(int n, void *dstp, void *srcp, int *gv)
{
  __CPLX32_T *dst = (__CPLX32_T *)dstp;
  __CPLX32_T *src = (__CPLX32_T *)srcp;
  int i;
  for (i = 0; i < n; ++i)
    dst[i] = src[gv[i]];
}

static gatherfn_t __fort_local_gather[__NTYPES] = {
    NULL,                /*     no type (absent optional argument) */
    NULL,                /* C   signed short */
    NULL,                /* C   unsigned short */
    NULL,                /* C   signed int */
    NULL,                /* C   unsigned int */
    NULL,                /* C   signed long int */
    NULL,                /* C   unsigned long int */
    NULL,                /* C   float */
    NULL,                /* C   double */
    local_gather_CPLX8,  /*   F complex*8 (2x real*4) */
    local_gather_CPLX16, /*   F complex*16 (2x real*8) */
    NULL,                /* C   signed char */
    NULL,                /* C   unsigned char */
    NULL,                /* C   long double */
    NULL,                /*   F character */
    NULL,                /* C   long long */
    NULL,                /* C   unsigned long long */
    local_gather_LOG1,   /*   F logical*1 */
    local_gather_LOG2,   /*   F logical*2 */
    local_gather_LOG4,   /*   F logical*4 */
    local_gather_LOG8,   /*   F logical*8 */
    NULL,                /*   F typeless */
    NULL,                /*   F double typeless */
    NULL,                /*   F ncharacter - kanji */
    local_gather_INT2,   /*   F integer*2 */
    local_gather_INT4,   /*   F integer*4, integer */
    local_gather_INT8,   /*   F integer*8 */
    local_gather_REAL4,  /*   F real*4, real */
    local_gather_REAL8,  /*   F real*8, double precision */
    local_gather_REAL16, /*   F real*16 */
    local_gather_CPLX32, /*   F complex*32 (2x real*16) */
    NULL,                /*   F quad typeless */
    local_gather_INT1,   /*   F integer*1 */
    NULL                 /*   F derived type */
};

void
local_gather_WRAPPER(int n, void *dst, void *src, int *gv, __INT_T kind)
{
  __fort_local_gather[kind](n, dst, src, gv);
}
