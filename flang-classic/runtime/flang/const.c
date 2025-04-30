/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* const.c -- constants */

#include "fioMacros.h"
#include "cnfg.h"
#include "float.h"

#ifdef TARGET_SUPPORTS_QUADFP
#define MAXLONGDOUBLE (LDBL_MAX)
#endif
#define MAXDOUBLE ((double)1.797693134862315708e+308)
#define MAXFLOAT ((float)3.40282346638528860e+38)

/* shift values for each data type (used by other modules) */
int __fort_shifts[__NTYPES]; /* initialized by __fort_init_consts */

/* size of data type */

int __fort_size_of[__NTYPES] = {
    0,                      /*     no type (absent optional argument) */
    sizeof(__SHORT_T),      /* C   signed short */
    sizeof(__USHORT_T),     /* C   unsigned short */
    sizeof(__CINT_T),       /* C   signed int */
    sizeof(__UINT_T),       /* C   unsigned int */
    sizeof(__LONG_T),       /* C   signed long int */
    sizeof(__ULONG_T),      /* C   unsigned long int */
    sizeof(__FLOAT_T),      /* C   float */
    sizeof(__DOUBLE_T),     /* C   double */
    sizeof(__CPLX8_T),      /*   F complex*8 (2x real*4) */
    sizeof(__CPLX16_T),     /*   F complex*16 (2x real*8) */
    sizeof(__CHAR_T),       /* C   signed char */
    sizeof(__UCHAR_T),      /* C   unsigned char */
    sizeof(__LONGDOUBLE_T), /* C   long double */
    sizeof(__STR_T),        /*   F character */
    sizeof(__LONGLONG_T),   /* C   long long */
    sizeof(__ULONGLONG_T),  /* C   unsigned long long */
    sizeof(__LOG1_T),       /*   F logical*1 */
    sizeof(__LOG2_T),       /*   F logical*2 */
    sizeof(__LOG4_T),       /*   F logical*4 */
    sizeof(__LOG8_T),       /*   F logical*8 */
    sizeof(__WORD4_T),      /*   F typeless */
    sizeof(__WORD8_T),      /*   F double typeless */
    sizeof(__NCHAR_T),      /*   F ncharacter - kanji */
    sizeof(__INT2_T),       /*   F integer*2 */
    sizeof(__INT4_T),       /*   F integer*4 */
    sizeof(__INT8_T),       /*   F integer*8 */
    sizeof(__REAL4_T),      /*   F real*4 */
    sizeof(__REAL8_T),      /*   F real*8 */
    sizeof(__REAL16_T),     /*   F real*16 */
    sizeof(__CPLX32_T),     /*   F complex*32 (2x real*16) */
    sizeof(__WORD16_T),     /*   F quad typeless */
    sizeof(__INT1_T),       /*   F integer*1 */
    sizeof(__DERIVED_T),    /*   F derived type */
    sizeof(__PROC_T),       /*     __PROC */
    sizeof(__DESC_T),       /*     __DESC */
    sizeof(__SKED_T),       /*     __SKED */
    16,                     /*     __M128 */
    32,                     /*     __M256 */
    16,                     /*   F integer*16 */
    16,                     /*   F logical*16 */
    16,                     /*   F real*16    */
    32,                     /*   F complex*32 */
    sizeof(__POLY_T),       /*   F polymorphic derived type */
    sizeof(__PROCPTR_T),    /*   F procedure pointer */
};

const char *__fort_typenames[__NTYPES] = {
    "none",               /*     no type (absent optional argument) */
    "short",              /* C   signed short */
    "unsigned short",     /* C   unsigned short */
    "int",                /* C   signed int */
    "unsigned int",       /* C   unsigned int */
    "long",               /* C   signed long int */
    "unsigned long",      /* C   unsigned long int */
    "float",              /* C   float */
    "double",             /* C   double */
    "complex*8",          /*   F complex*8 (2x real*4) */
    "complex*16",         /*   F complex*16 (2x real*8) */
    "char",               /* C   signed char */
    "unsigned char",      /* C   unsigned char */
    "long double",        /* C   long double */
    "character*(*)",      /*   F character */
    "long long",          /* C   long long */
    "unsigned long long", /* C   unsigned long long */
    "logical*1",          /*   F logical*1 */
    "logical*2",          /*   F logical*2 */
    "logical*4",          /*   F logical*4 */
    "logical*8",          /*   F logical*8 */
    "word*4",             /*   F typeless */
    "word*8",             /*   F double typeless */
    "nchar*2",            /*   F ncharacter - kanji */
    "integer*2",          /*   F integer*2 */
    "integer*4",          /*   F integer*4 */
    "integer*8",          /*   F integer*8 */
    "real*4",             /*   F real*4 */
    "real*8",             /*   F real*8 */
    "real*16",            /*   F real*16 */
    "complex*32",         /*   F complex*32 (2x real*16) */
    "word*16",            /*   F quad typeless */
    "integer*1",          /*   F integer*1 */
    "type()",             /*   F derived type */
    "rte34",              /*     __PROC */
    "rte35",              /*     __DESC */
    "rte36",              /*     __SKED */
    "m128",               /*     __M128 */
    "m256",               /*     __M256 */
    "integer*16",         /*   F integer*16 */
    "logical*16",         /*   F logical*16 */
    "real*16",            /*   F real*16    */
    "complex*32",         /*   F complex*32 */
    "class()",            /*   F polymorphic variable */
    "procedure ptr",      /*   F procedure pointer */
};

/* internal datatype array, -42:42
 *  These values should be the same as what's in rest.c
 */
__INT_T ENTCOMN(TYPE, type)[] = {
    -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29,
    -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14,
    -13, -12, -11, -10, -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,  0,   1,
    2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,
    17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
    32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43};

/* universal constants */

long long int __fort_one[4] = {~0, ~0, ~0, ~0};
long long int __fort_zed[4] = {0, 0, 0, 0};

/* maximum values */

static __INT1_T max_int1 = 0; /* initialized */
static __INT2_T max_int2 = 0; /* initialized */
static __INT4_T max_int4 = 0; /* initialized */
static __INT8_T max_int8 = 0; /* initialized */
static __STR_T max_str = (__STR_T) 255; /* initialized */
static __REAL4_T max_real4 = MAXFLOAT;
static __REAL8_T max_real8 = MAXDOUBLE;
#ifdef TARGET_SUPPORTS_QUADFP
static __REAL16_T max_real16 = MAXLONGDOUBLE;
#else
static __REAL16_T max_real16 = MAXDOUBLE;
#endif

void *__fort_maxs[__NTYPES] = {
    (void *)0,          /*  0 __NONE       no type */
    (void *)0,          /*  1 __SHORT      short */
    (void *)0,          /*  2 __USHORT     unsigned short */
    (void *)0,          /*  3 __CINT       int */
    (void *)0,          /*  4 __UINT       unsigned int */
    (void *)0,          /*  5 __LONG       long */
    (void *)0,          /*  6 __ULONG      unsigned long */
    (void *)0,          /*  7 __FLOAT      float */
    (void *)0,          /*  8 __DOUBLE     double */
    (void *)0,          /*  9 __CPLX8      float complex */
    (void *)0,          /* 10 __CPLX16     double complex */
    (void *)0,          /* 11 __CHAR       char */
    (void *)0,          /* 12 __UCHAR      unsigned char */
    (void *)0,          /* 13 __LONGDOUBLE long double */
    (char *) & max_str, /* 14 __STR        string */
    (void *)0,          /* 15 __LONGLONG   long long */
    (void *)0,          /* 16 __ULONGLONG  unsigned long long */
    __fort_zed,          /* 17 __LOG1       logical*1 */
    __fort_zed,          /* 18 __LOG2       logical*2 */
    __fort_zed,          /* 19 __LOG4       logical*4*/
    __fort_zed,          /* 20 __LOG8       logical*8 */
    (void *)0,          /* 21 __WORD4      typeless */
    (void *)0,          /* 22 __WORD8      double typeless */
    (void *)0,          /* 23 __NCHAR      ncharacter - kanji */
    &max_int2,          /* 24 __INT2       integer*2 */
    &max_int4,          /* 25 __INT4       integer*4 */
    &max_int8,          /* 26 __INT8       integer*8 */
    &max_real4,         /* 27 __REAL4      real*4 */
    &max_real8,         /* 28 __REAL8      real*8 */
    &max_real16,        /* 29 __REAL16     real*16 */
    (void *)0,          /* 30 __CPLX32     complex*32 */
    (void *)0,          /* 31 __WORD16     quad typeless */
    &max_int1,          /* 32 __INT1       integer*1 */
    (void *)0,          /* 33 __DERIVED    derived type */
    (void *)0,          /* 34 __PROC       processors descriptor */
    (void *)0,          /* 35 __DESC       section descriptor */
    (void *)0,          /* 36 __SKED       communication schedule */
    (void *)0,          /* 37 __M128       128-bit type */
    (void *)0,          /* 38 __M256       256-bit type */
    (void *)0,          /* 39 __INT16      integer(16) */
    (void *)0,          /* 40 __LOG16      logical(16) */
    (void *)0,          /* 41 __QREAL16    real(16) */
    (void *)0,          /* 42 __QCPLX32    complex(32) */
    (void *)0,          /* 43 __POLY       polymorphic derived type */
    (void *)0,          /* 44 __PROCPTR    procedure pointer */
};

/* minimum values */

static __INT1_T min_int1 = 0; /* initialized */
static __INT2_T min_int2 = 0; /* initialized */
static __INT4_T min_int4 = 0; /* initialized */
static __INT8_T min_int8 = 0; /* initialized */
static __STR_T min_str = 0;   /* initialized */
static __REAL4_T min_real4 = -MAXFLOAT;
static __REAL8_T min_real8 = -MAXDOUBLE;
#ifdef TARGET_SUPPORTS_QUADFP
static __REAL16_T min_real16 = -MAXLONGDOUBLE;
#else
static __REAL16_T min_real16 = -MAXDOUBLE;
#endif

void *__fort_mins[__NTYPES] = {
    (void *)0,          /*  0 __NONE       no type */
    (void *)0,          /*  1 __SHORT      short */
    (void *)0,          /*  2 __USHORT     unsigned short */
    (void *)0,          /*  3 __CINT       int */
    (void *)0,          /*  4 __UINT       unsigned int */
    (void *)0,          /*  5 __LONG       long */
    (void *)0,          /*  6 __ULONG      unsigned long */
    (void *)0,          /*  7 __FLOAT      float */
    (void *)0,          /*  8 __DOUBLE     double */
    (void *)0,          /*  9 __CPLX8      float complex */
    (void *)0,          /* 10 __CPLX16     double complex */
    (void *)0,          /* 11 __CHAR       char */
    (void *)0,          /* 12 __UCHAR      unsigned char */
    (void *)0,          /* 13 __LONGDOUBLE long double */
    (char *) & min_str, /* 14 __STR        string */
    (void *)0,          /* 15 __LONGLONG   long long */
    (void *)0,          /* 16 __ULONGLONG  unsigned long long */
    __fort_zed,          /* 17 __LOG1       logical*1 */
    __fort_zed,          /* 18 __LOG2       logical*2 */
    __fort_zed,          /* 19 __LOG4       logical*4*/
    __fort_zed,          /* 20 __LOG8       logical*8 */
    (void *)0,          /* 21 __WORD4      typeless */
    (void *)0,          /* 22 __WORD8      double typeless */
    (void *)0,          /* 23 __NCHAR      ncharacter - kanji */
    &min_int2,          /* 24 __INT2       integer*2 */
    &min_int4,          /* 25 __INT4       integer*4 */
    &min_int8,          /* 26 __INT8       integer*8 */
    &min_real4,         /* 27 __REAL4      real*4 */
    &min_real8,         /* 28 __REAL8      real*8 */
    &min_real16,        /* 29 __REAL16     real*16 */
    (void *)0,          /* 30 __CPLX32     complex*32 */
    (void *)0,          /* 31 __WORD16     quad typeless */
    &min_int1,          /* 32 __INT1       integer*1 */
    (void *)0,          /* 33 __DERIVED    derived type */
    (void *)0,          /* 34 __PROC       processors descriptor */
    (void *)0,          /* 35 __DESC       section descriptor */
    (void *)0,          /* 36 __SKED       communication schedule */
    (void *)0,          /* 37 __M128       128-bit type */
    (void *)0,          /* 38 __M256       256-bit type */
    (void *)0,          /* 39 __INT16      integer(16) */
    (void *)0,          /* 40 __LOG16      logical(16) */
    (void *)0,          /* 41 __QREAL16    real(16) */
    (void *)0,          /* 42 __QCPLX32    complex(32) */
    (void *)0,          /* 43 __POLY       polymorphic derived type */
    (void *)0,          /* 44 __PROCPTR    procedure pointer */
};

/* units */

static __INT1_T unit_int1 = 1;
static __INT2_T unit_int2 = 1;
static __INT4_T unit_int4 = 1;
static __INT8_T unit_int8 = 1;
static __REAL4_T unit_real4 = 1.0;
static __REAL8_T unit_real8 = 1.0;
static __REAL16_T unit_real16 = 1.0;
static __CPLX8_T unit_cplx8 = {1.0, 0.0};
static __CPLX16_T unit_cplx16 = {1.0, 0.0};

void *__fort_units[__NTYPES] = {
    (void *)0,    /*  0 __NONE       no type */
    (void *)0,    /*  1 __SHORT      short */
    (void *)0,    /*  2 __USHORT     unsigned short */
    (void *)0,    /*  3 __CINT       int */
    (void *)0,    /*  4 __UINT       unsigned int */
    (void *)0,    /*  5 __LONG       long */
    (void *)0,    /*  6 __ULONG      unsigned long */
    (void *)0,    /*  7 __FLOAT      float */
    (void *)0,    /*  8 __DOUBLE     double */
    &unit_cplx8,  /*  9 __CPLX8      float complex */
    &unit_cplx16, /* 10 __CPLX16     double complex */
    (void *)0,    /* 11 __CHAR       char */
    (void *)0,    /* 12 __UCHAR      unsigned char */
    (void *)0,    /* 13 __LONGDOUBLE long double */
    (void *)0,    /* 14 __STR        string */
    (void *)0,    /* 15 __LONGLONG   long long */
    (void *)0,    /* 16 __ULONGLONG  unsigned long long */
    __fort_one,    /* 17 __LOG1       logical*1 */
    __fort_one,    /* 18 __LOG2       logical*2 */
    __fort_one,    /* 19 __LOG4       logical*4*/
    __fort_one,    /* 20 __LOG8       logical*8 */
    (void *)0,    /* 21 __WORD4      typeless */
    (void *)0,    /* 22 __WORD8      double typeless */
    (void *)0,    /* 23 __NCHAR      ncharacter - kanji */
    &unit_int2,   /* 24 __INT2       integer*2 */
    &unit_int4,   /* 25 __INT4       integer*4 */
    &unit_int8,   /* 26 __INT8       integer*8 */
    &unit_real4,  /* 27 __REAL4      real*4 */
    &unit_real8,  /* 28 __REAL8      real*8 */
    &unit_real16, /* 29 __REAL16     real*16 */
    (void *)0,    /* 30 __CPLX32     complex*32 */
    (void *)0,    /* 31 __WORD16     quad typeless */
    &unit_int1,   /* 32 __INT1       integer*1 */
    (void *)0     /* 33 __DERIVED    derived type */
};

/* logical trues - initialized from __fort_cnfg_.ftn_true */

__LOG_T __fort_true_log = 1;
__LOG1_T __fort_true_log1;
__LOG2_T __fort_true_log2;
__LOG4_T __fort_true_log4;
__LOG8_T __fort_true_log8;
static __INT1_T __fort_true_int1;
static __INT2_T __fort_true_int2;
static __INT4_T __fort_true_int4;
static __INT8_T __fort_true_int8;
static __REAL4_T __fort_true_real4;
static __REAL8_T __fort_true_real8;
static __REAL16_T __fort_true_real16;
static __CPLX8_T __fort_true_cplx8;
static __CPLX16_T __fort_true_cplx16;
static __CPLX32_T __fort_true_cplx32;

void *__fort_trues[__NTYPES] = {
    (void *)0,          /*  0 __NONE       no type */
    (void *)0,          /*  1 __SHORT      short */
    (void *)0,          /*  2 __USHORT     unsigned short */
    (void *)0,          /*  3 __CINT       int */
    (void *)0,          /*  4 __UINT       unsigned int */
    (void *)0,          /*  5 __LONG       long */
    (void *)0,          /*  6 __ULONG      unsigned long */
    (void *)0,          /*  7 __FLOAT      float */
    (void *)0,          /*  8 __DOUBLE     double */
    &__fort_true_cplx8,  /*  9 __CPLX8      float complex */
    &__fort_true_cplx16, /* 10 __CPLX16     double complex */
    (void *)0,          /* 11 __CHAR       char */
    (void *)0,          /* 12 __UCHAR      unsigned char */
    (void *)0,          /* 13 __LONGDOUBLE long double */
    (void *)0,          /* 14 __STR        string */
    (void *)0,          /* 15 __LONGLONG   long long */
    (void *)0,          /* 16 __ULONGLONG  unsigned long long */
    &__fort_true_log1,   /* 17 __LOG1       logical*1 */
    &__fort_true_log2,   /* 18 __LOG2       logical*2 */
    &__fort_true_log4,   /* 19 __LOG4       logical*4 */
    &__fort_true_log8,   /* 20 __LOG8       logical*8 */
    (void *)0,          /* 21 __WORD4      typeless */
    (void *)0,          /* 22 __WORD8      double typeless */
    (void *)0,          /* 23 __NCHAR      ncharacter - kanji */
    &__fort_true_int2,   /* 24 __INT2       integer*2 */
    &__fort_true_int4,   /* 25 __INT4       integer*4 */
    &__fort_true_int8,   /* 26 __INT8       integer*8 */
    &__fort_true_real4,  /* 27 __REAL4      real*4 */
    &__fort_true_real8,  /* 28 __REAL8      real*8 */
    &__fort_true_real16, /* 29 __REAL16     real*16 */
    &__fort_true_cplx32, /* 30 __CPLX32     complex*32 */
    (void *)0,          /* 31 __WORD16     quad typeless */
    &__fort_true_int1,   /* 32 __INT1       integer*1 */
    (void *)0           /* 33 __DERIVED    derived type */
};

/* logical masks - initialized from __fort_cnfg_.true_mask */
__LOG_T __fort_mask_log;

__LOG1_T __fort_mask_log1;
__LOG2_T __fort_mask_log2;
__LOG4_T __fort_mask_log4;
__LOG8_T __fort_mask_log8;
__INT1_T __fort_mask_int1;
__INT2_T __fort_mask_int2;
__INT4_T __fort_mask_int4;
__INT8_T __fort_mask_int8;
static __REAL4_T __fort_mask_real4;
static __REAL8_T __fort_mask_real8;
static __REAL16_T __fort_mask_real16;
static __CPLX8_T __fort_mask_cplx8;
static __CPLX16_T __fort_mask_cplx16;
static __CPLX32_T __fort_mask_cplx32;
static __STR_T __fort_mask_str;

void *__fort_masks[__NTYPES] = {
    (void *)0,          /*  0 __NONE       no type */
    (void *)0,          /*  1 __SHORT      short */
    (void *)0,          /*  2 __USHORT     unsigned short */
    (void *)0,          /*  3 __CINT       int */
    (void *)0,          /*  4 __UINT       unsigned int */
    (void *)0,          /*  5 __LONG       long */
    (void *)0,          /*  6 __ULONG      unsigned long */
    (void *)0,          /*  7 __FLOAT      float */
    (void *)0,          /*  8 __DOUBLE     double */
    &__fort_mask_cplx8,  /*  9 __CPLX8      float complex */
    &__fort_mask_cplx16, /* 10 __CPLX16     double complex */
    (void *)0,          /* 11 __CHAR       char */
    (void *)0,          /* 12 __UCHAR      unsigned char */
    (void *)0,          /* 13 __LONGDOUBLE long double */
    &__fort_mask_str,    /* 14 __STR        string */
    (void *)0,          /* 15 __LONGLONG   long long */
    (void *)0,          /* 16 __ULONGLONG  unsigned long long */
    &__fort_mask_log1,   /* 17 __LOG1       logical*1 */
    &__fort_mask_log2,   /* 18 __LOG2       logical*2 */
    &__fort_mask_log4,   /* 19 __LOG4       logical*4*/
    &__fort_mask_log8,   /* 20 __LOG8       logical*8 */
    (void *)0,          /* 21 __WORD4      typeless */
    (void *)0,          /* 22 __WORD8      double typeless */
    (void *)0,          /* 23 __NCHAR      ncharacter - kanji */
    &__fort_mask_int2,   /* 24 __INT2       integer*2 */
    &__fort_mask_int4,   /* 25 __INT4       integer*4 */
    &__fort_mask_int8,   /* 26 __INT8       integer*8 */
    &__fort_mask_real4,  /* 27 __REAL4      real*4 */
    &__fort_mask_real8,  /* 28 __REAL8      real*8 */
    &__fort_mask_real16, /* 29 __REAL16     real*16 */
    &__fort_mask_cplx32, /* 30 __CPLX32     complex*32 */
    (void *)0,          /* 31 __WORD16     quad typeless */
    &__fort_mask_log1,   /* 32 __INT1       integer*1 */
    (void *)0           /* 33 __DERIVED    derived type */
};

int
__get_size_of(int* idx)
{
  return __fort_size_of[*idx];
}

#if defined(_WIN64)

/* pg access routines for data shared between windows dlls */

__LOG_T
__get_fort_true_log(void) { return __fort_true_log; }

__LOG_T *
__get_fort_true_log_addr(void)
{
  return &__fort_true_log;
}

__LOG1_T
__get_fort_true_log1(void) { return __fort_true_log1; }

__LOG2_T
__get_fort_true_log2(void) { return __fort_true_log2; }

__LOG4_T
__get_fort_true_log4(void) { return __fort_true_log4; }

__LOG8_T
__get_fort_true_log8(void) { return __fort_true_log8; }

void
__set_fort_true_log(__LOG_T t)
{
  __fort_true_log = t;
}

void
__set_fort_true_log1(__LOG1_T t)
{
  __fort_true_log1 = t;
}

void
__set_fort_true_log2(__LOG2_T t)
{
  __fort_true_log2 = t;
}

void
__set_fort_true_log4(__LOG4_T t)
{
  __fort_true_log4 = t;
}

void
__set_fort_true_log8(__LOG8_T t)
{
  __fort_true_log8 = t;
}

__LOG_T
__get_fort_mask_log(void) { return __fort_mask_log; }

__LOG1_T
__get_fort_mask_log1(void) { return __fort_mask_log1; }

__LOG2_T
__get_fort_mask_log2(void) { return __fort_mask_log2; }

__LOG4_T
__get_fort_mask_log4(void) { return __fort_mask_log4; }

__LOG8_T
__get_fort_mask_log8(void) { return __fort_mask_log8; }

__INT1_T
__get_fort_mask_int1(void) { return __fort_mask_int1; }

__INT2_T
__get_fort_mask_int2(void) { return __fort_mask_int2; }

__INT4_T
__get_fort_mask_int4(void) { return __fort_mask_int4; }

__INT8_T
__get_fort_mask_int8(void) { return __fort_mask_int8; }

__STR_T
__get_fort_mask_str(void) { return __fort_mask_str; }

void
__set_fort_mask_log(__LOG_T m)
{
  __fort_mask_log = m;
}

void
__set_fort_mask_log1(__LOG1_T m)
{
  __fort_mask_log1 = m;
}

void
__set_fort_mask_log2(__LOG2_T m)
{
  __fort_mask_log2 = m;
}

void
__set_fort_mask_log4(__LOG4_T m)
{
  __fort_mask_log4 = m;
}

void
__set_fort_mask_log8(__LOG8_T m)
{
  __fort_mask_log8 = m;
}

void
__set_fort_mask_int1(__INT1_T m)
{
  __fort_mask_int1 = m;
}

void
__set_fort_mask_int2(__INT2_T m)
{
  __fort_mask_int2 = m;
}

void
__set_fort_mask_int4(__INT4_T m)
{
  __fort_mask_int4 = m;
}

void
__set_fort_mask_int8(__INT8_T m)
{
  __fort_mask_int8 = m;
}

void *
__get_fort_maxs(int idx)
{
  return __fort_maxs[idx];
}

void *
__get_fort_mins(int idx)
{
  return __fort_mins[idx];
}

int
__get_fort_shifts(int idx)
{
  return __fort_shifts[idx];
}

int
__get_fort_size_of(int idx)
{
  return __fort_size_of[idx];
}

void *
__get_fort_trues(int idx)
{
  return __fort_trues[idx];
}

const char *
__get_fort_typenames(int idx)
{
  return __fort_typenames[idx];
}

void *
__get_fort_units(int idx)
{
  return __fort_units[idx];
}

void
__set_fort_maxs(int idx, void *val)
{
  __fort_maxs[idx] = val;
}

void
__set_fort_mins(int idx, void *val)
{
  __fort_mins[idx] = val;
}

void
__set_fort_shifts(int idx, int val)
{
  __fort_shifts[idx] = val;
}

void
__set_fort_size_of(int idx, int val)
{
  __fort_size_of[idx] = val;
}

void
__set_fort_trues(int idx, void *val)
{
  __fort_trues[idx] = val;
}

void
__set_fort_typenames(int idx, const char *val)
{
  __fort_typenames[idx] = val;
}

void
__set_fort_units(int idx, void *val)
{
  __fort_units[idx] = val;
}

long long int *
__get_fort_one(void)
{
  return __fort_one;
}

long long int *
__get_fort_zed(void)
{
  return __fort_zed;
}

#endif /* _WIN64 */

void
__fort_init_consts()
{
  int i, j, k;
  char *m, *t;

/* Compute max value for N bits: 2**(N-1)-1 can overflow so use 2**(N-2) - 1 + 2**(N-2) */
#define MAX_FOR_INT_TYPE(type) \
  ((type)1 << (8*sizeof(type) - 2)) - 1 + ((type)1 << (8*sizeof(type) - 2));
  max_int1 = MAX_FOR_INT_TYPE(__INT1_T);
  max_int2 = MAX_FOR_INT_TYPE(__INT2_T);
  max_int4 = MAX_FOR_INT_TYPE(__INT4_T);
  max_int8 = MAX_FOR_INT_TYPE(__INT8_T);
#undef MAX_FOR_INT_TYPE

  max_str = (__STR_T) 255;

  min_int1 = -max_int1 - 1;
  min_int2 = -max_int2 - 1;
  min_int4 = -max_int4 - 1;
  min_int8 = -max_int8 - 1;
  min_str = 0;

  __fort_shifts[__NONE] = 0;

  for (i = __NONE + 1; i < __NTYPES; ++i) {

    /* initialize __fort_shifts */

    for (j = 0, k = 1; k < __fort_size_of[i]; ++j, k <<= 1)
      ;
#if defined(DEBUG)
    if (k != __fort_size_of[i])
      __fort_abort("init_consts: type size not a power of two");
#endif
    __fort_shifts[i] = j;

    /* initialize logical trues */

    m = (char *)GET_FIO_CNFG_FTN_TRUE_ADDR;
    t = __fort_trues[i];
    if (t != (void *)0) {
      for (j = 0; j < k; ++j)
        t[j] = m[1];
      t[0] |= m[0];
      t[k - 1] |= m[sizeof(GET_FIO_CNFG_FTN_TRUE) - 1];
    }

    /* initialize logical masks */

    m = (char *)GET_FIO_CNFG_TRUE_MASK_ADDR;
    t = __fort_masks[i];
    if (t != (void *)0) {
      for (j = 0; j < k; ++j)
        t[j] = m[1];
      t[0] |= m[0];
      t[k - 1] |= m[sizeof(GET_FIO_CNFG_TRUE_MASK) - 1];
    }
  }
  __fort_true_log = *(__LOG_T *)__fort_trues[__LOG];
  __fort_mask_log = *(__LOG_T *)__fort_masks[__LOG];

#if defined(DEBUG)

/* check compiler-runtime descriptor interface constants */

  if (sizeof(__POINT_T) != sizeof(char *))
    __fort_abort("init_consts: __POINT_T is not pointer size");

  if (sizeof(F90_Desc) !=
      (F90_DESC_HDR_INT_LEN * sizeof(__INT_T) +
       F90_DESC_HDR_PTR_LEN * sizeof(__POINT_T) +
       MAXDIMS * (F90_DESC_DIM_INT_LEN * sizeof(__INT_T) +
                  F90_DESC_DIM_PTR_LEN * sizeof(__POINT_T))))
    __fort_abort("init_consts: F90_DESC_HDR INT/PTR_LEN incorrect");

  if (sizeof(DIST_Desc) !=
      (DIST_DESC_HDR_INT_LEN * sizeof(__INT_T) +
       DIST_DESC_HDR_PTR_LEN * sizeof(__POINT_T) +
       MAXDIMS * (DIST_DESC_DIM_INT_LEN * sizeof(__INT_T) +
                  DIST_DESC_DIM_PTR_LEN * sizeof(__POINT_T))))
    __fort_abort("init_consts: DIST_DESC_HDR INT/PTR_LEN incorrect");

  /* check reciprocal operations */

  for (j = 1; j <= 10; ++j) {
    __INT_T j_recip = RECIP(j);
    for (i = 0; i < 100; ++i) {
      int quo, rem;
      RECIP_DIV(&quo, i, j);
      if (quo != i / j)
        __fort_abort("init_consts: RECIP_DIV failed");
      RECIP_MOD(&rem, i, j);
      if (rem != i % j)
        __fort_abort("init_consts: RECIP_MOD failed");
      RECIP_DIVMOD(&quo, &rem, i, j);
      if (quo != i / j || rem != i % j)
        __fort_abort("init_consts: RECIP_DIVMOD failed");
    }
  }
#endif
}

/*
 * Always emit the comms for non-windows systems.
 */
#if defined(_WIN64)
/*
 * Emit the comms for win if pg.dll is not used -- PGDLL is defined
 * if we need to revert to pg.dll.
 */
#endif
__INT_T ENTCOMN(0, 0)[4];
__STR_T ENTCOMN(0C, 0c)[1];
__INT_T ENTCOMN(LOCAL_MODE, local_mode)[1];
__INT_T ENTCOMN(NP, np)[1];
__INT_T ENTCOMN(ME, me)[1];
__INT_T LINENO[1];

