/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* scalar_copy.c - scalar element copy routines */

#include "stdioInterf.h"
#include "fioMacros.h"

static void
copy_none(__SHORT_T *rp, const __SHORT_T *sp, int size)
{
  __fort_abort("scalar_copy: undefined type");
}
static void
copy_short(__SHORT_T *rp, const __SHORT_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_ushort(__USHORT_T *rp, const __USHORT_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_cint(__CINT_T *rp, const __CINT_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_uint(__UINT_T *rp, const __UINT_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_long(__LONG_T *rp, const __LONG_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_ulong(__ULONG_T *rp, const __ULONG_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_float(__FLOAT_T *rp, const __FLOAT_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_double(__DOUBLE_T *rp, const __DOUBLE_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_cplx8(__CPLX8_T *rp, const __CPLX8_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_cplx16(__CPLX16_T *rp, const __CPLX16_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_char(__CHAR_T *rp, const __CHAR_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_uchar(__UCHAR_T *rp, const __UCHAR_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_longdouble(__LONGDOUBLE_T *rp, const __LONGDOUBLE_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_longlong(__LONGLONG_T *rp, const __LONGLONG_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_ulonglong(__ULONGLONG_T *rp, const __ULONGLONG_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_log1(__LOG1_T *rp, const __LOG1_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_log2(__LOG2_T *rp, const __LOG2_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_log4(__LOG4_T *rp, const __LOG4_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_log8(__LOG8_T *rp, const __LOG8_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_word4(__WORD4_T *rp, const __WORD4_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_word8(__WORD8_T *rp, const __WORD8_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_nchar(__NCHAR_T *rp, const __NCHAR_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_int2(__INT2_T *rp, const __INT2_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_int4(__INT4_T *rp, const __INT4_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_int8(__INT8_T *rp, const __INT8_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_real4(__REAL4_T *rp, const __REAL4_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_real8(__REAL8_T *rp, const __REAL8_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_real16(__REAL16_T *rp, const __REAL16_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_cplx32(__CPLX32_T *rp, const __CPLX32_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_word16(__WORD16_T *rp, const __WORD16_T *sp, int size)
{
  *rp = *sp;
}
static void
copy_int1(__INT1_T *rp, const __INT1_T *sp, int size)
{
  *rp = *sp;
}

static void copy_bytes(char *, const char *, int);

void (*__fort_scalar_copy[__NTYPES])() = {
    copy_none,       /*     no type (absent optional argument) */
    copy_short,      /* C   signed short */
    copy_ushort,     /* C   unsigned short */
    copy_cint,       /* C   signed int */
    copy_uint,       /* C   unsigned int */
    copy_long,       /* C   signed long int */
    copy_ulong,      /* C   unsigned long int */
    copy_float,      /* C   float */
    copy_double,     /* C   double */
    copy_cplx8,      /*   F complex*8 (2x real*4) */
    copy_cplx16,     /*   F complex*16 (2x real*8) */
    copy_char,       /* C   signed char */
    copy_uchar,      /* C   unsigned char */
    copy_longdouble, /* C   long double */
    copy_bytes,      /*   F character */
    copy_longlong,   /* C   long long */
    copy_ulonglong,  /* C   unsigned long long */
    copy_log1,       /*   F logical*1 */
    copy_log2,       /*   F logical*2 */
    copy_log4,       /*   F logical*4 */
    copy_log8,       /*   F logical*8 */
    copy_word4,      /*   F typeless */
    copy_word8,      /*   F double typeless */
    copy_nchar,      /*   F ncharacter - kanji */
    copy_int2,       /*   F integer*2 */
    copy_int4,       /*   F integer*4, integer */
    copy_int8,       /*   F integer*8 */
    copy_real4,      /*   F real*4, real */
    copy_real8,      /*   F real*8, double precision */
    copy_real16,     /*   F real*16 */
    copy_cplx32,     /*   F complex*32 (2x real*16) */
    copy_word16,     /*   F quad typeless */
    copy_int1,       /*   F integer*1 */
    copy_bytes       /*   F derived type */
};

static void
copy_bytes(char *to, const char *fr, int n)
{
  memmove(to, fr, n);
}
