/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
 * \file
 * \brief - runtime Data type code definitions
 */

#ifndef _PGHPF_TYPES_H_
#define _PGHPF_TYPES_H_

#include "float128.h"

/** \typedef _DIST_TYPE
 *
 *  \brief Data type codes.
 *
 * These must correspond to the compiler's codes.
 *
 * Any changes to the intrinsic types must get reflected in the
 * F2003 type descriptors defined in type.c (see I8(__f03_ty_to_id)[] array).
 * (see also \ref __NTYPES below).
 */
typedef enum {
  __NONE = 0,        /**< no type (absent optional argument) */
  __SHORT = 1,       /**< C   signed short */
  __USHORT = 2,      /**< C   unsigned short */
  __CINT = 3,        /**< C   signed int */
  __UINT = 4,        /**< C   unsigned int */
  __LONG = 5,        /**< C   signed long int */
  __ULONG = 6,       /**< C   unsigned long int */
  __FLOAT = 7,       /**< C   float */
  __DOUBLE = 8,      /**< C   double */
  __CPLX8 = 9,       /**< Fortran complex*8 (2x real*4) */
  __CPLX16 = 10,     /**< Fortran complex*16 (2x real*8) */
  __CHAR = 11,       /**< C   signed char */
  __UCHAR = 12,      /**< C   unsigned char */
  __LONGDOUBLE = 13, /**< C   long double */
  __STR = 14,        /**< Fortran character */
  __LONGLONG = 15,   /**< C   long long */
  __ULONGLONG = 16,  /**< C   unsigned long long */
  __LOG1 = 17,       /**< Fortran logical*1 */
  __LOG2 = 18,       /**< Fortran logical*2 */
  __LOG4 = 19,       /**< Fortran logical*4 */
  __LOG8 = 20,       /**< Fortran logical*8 */
  __WORD4 = 21,      /**< Fortran typeless */
  __WORD8 = 22,      /**< Fortran double typeless */
  __NCHAR = 23,      /**< Fortran ncharacter - kanji */

  __INT2 = 24,       /**< Fortran integer*2 */
  __INT4 = 25,       /**< Fortran integer*4, integer */
  __INT8 = 26,       /**< Fortran integer*8 */
  __REAL2 = 45,      /**< Fortran real*2, half */
  __REAL4 = 27,      /**< Fortran real*4, real */
  __REAL8 = 28,      /**< Fortran real*8, double precision */
  __REAL16 = 29,     /**< Fortran real*16 */
  __CPLX32 = 30,     /**< Fortran complex*32 (2x real*16) */
  __WORD16 = 31,     /**< Fortran quad typeless */
  __INT1 = 32,       /**< Fortran integer*1 */
  __DERIVED = 33,    /**< Fortran derived-type */

  /* runtime descriptor types (not scalar data types) */
  __PROC = 34,       /**< processors descriptor */
  __DESC = 35,       /**< template/array/section descriptor */
  __SKED = 36,       /**< communication schedule */

  __M128 = 37,       /**< 128-bit type */
  __M256 = 38,       /**< 256-bit type */
  __INT16 = 39,      /**< Fortran integer(16) */
  __LOG16 = 40,      /**< Fortran logical(16) */
  __QREAL16 = 41,    /**< Fortran real(16) */
  __QCPLX32 = 42,    /**< Fortran complex(32) */
  __POLY = 43,       /**< Fortran polymorphic variable */
  __PROCPTR = 44,    /**< Fortran Procedure Ptr Descriptor */

/** \def __NTYPES
 *
 * Number of data types (for sizing arrays).  This used to be the
 * number of scalar data types for. Unfortunately, the values of the
 * runtime descriptor types cannot change.  Therefore, new values will
 * be added after any current values.
 */
#define __NTYPES 46

} _DIST_TYPE;

/* typedefs for all of the scalar data types.  Arbitrary substitutions
   are made where the C compiler doesn't support an equivalent type.
   The compiler promises not to generate those types unless the C
   compiler also supports them.  typedefs ending in _UT are unsigned
   versions of the corresponding Fortran _T type (they don't have a
   separate enumeration.)  */

typedef short __SHORT_T; /*  1 __SHORT      signed short */

typedef unsigned short __USHORT_T; /*  2 __USHORT     unsigned short */

typedef int __CINT_T; /*  3 __CINT       C signed int */

typedef unsigned int __UINT_T; /*  4 __UINT       unsigned int */

typedef long __LONG_T; /*  5 __LONG       signed long int */

typedef unsigned long __ULONG_T; /*  6 __ULONG      unsigned long int */

typedef float __FLOAT_T; /*  7 __FLOAT      float */

typedef double __DOUBLE_T; /*  8 __DOUBLE     double */

typedef char __CHAR_T; /* 11 __CHAR       signed char */

typedef unsigned char __UCHAR_T; /* 12 __UCHAR      unsigned char */

typedef double __LONGDOUBLE_T;       /* 13 __LONGDOUBLE long double */

typedef char __STR_T; /* 14 __STR        character */

typedef long long __LONGLONG_T; /* 15 __LONGLONG   long long */
typedef unsigned long long
    __ULONGLONG_T; /* 16 __ULONGLONG  unsigned long long */

typedef signed char __LOG1_T; /* 17 __LOG1       logical*1 */
typedef short __LOG2_T;       /* 18 __LOG2       logical*2 */

typedef int __LOG4_T;                /* 19 __LOG4       logical*4 */

typedef long long __LOG8_T;          /* 20 __LOG8       logical*8 */

/* lfm -- these two are wrong, hopefully not used */
typedef int __WORD4_T; /* 21 __WORD4      typeless */

typedef double __WORD8_T; /* 22 __WORD8      double typeless */

typedef short __NCHAR_T;      /* 23 __NCHAR      ncharacter - kanji */
typedef short __INT2_T;       /* 24 __INT2       integer*2 */
typedef unsigned short __INT2_UT;

typedef int __INT4_T; /* 25 __INT4       integer*4 */
typedef unsigned int __INT4_UT;

typedef long __INT8_T; /* 26 __INT8       integer*8 */
typedef unsigned long __INT8_UT;

typedef unsigned short __REAL2_T; /* 45 __REAL2      real*2 */

typedef float __REAL4_T; /* 27 __REAL4      real*4 */

typedef double __REAL8_T; /* 28 __REAL8      real*8 */
#ifdef TARGET_SUPPORTS_QUADFP
typedef float128_t __REAL16_T; /* 29 __REAL16     real*16 */
#else
typedef double __REAL16_T; /* 29 __REAL16     real*16 */
#endif
typedef struct {
  __REAL4_T r, i;
} __CPLX8_T; /*  9 __CPLX8       complex*8 */

typedef struct {
  __REAL8_T r, i;
} __CPLX16_T; /* 10 __CPLX16      complex*16 */

typedef struct {
  __REAL16_T r, i;
} __CPLX32_T; /* 30 __CPLX32     complex*32 */

typedef double __WORD16_T; /* 31 __WORD16     quad typeless */

typedef signed char __INT1_T; /* 32 __INT1       integer*1 */
typedef unsigned char __INT1_UT;

typedef char __DERIVED_T; /* 33 __DERIVED    derived type */

typedef char __PROC_T; /* 34 __PROC */

typedef char __DESC_T; /* 35 __DESC */

typedef char __SKED_T; /* 36 __SKED */

typedef char __POLY_T; /* 43 __POLY    polymorphic derived type */

typedef char __PROCPTR_T; /* 44 __PROCPTR */

/* double and quad type component views */

typedef union {
  __REAL8_T d;
  __INT8_T ll;
  struct {
    __INT4_T l, h;
  } i;
} __REAL8_SPLIT;

typedef union {
  __REAL16_T q;
  struct {
    __INT8_T l, h;
  } ll;
  struct {
    __INT4_T l, k, j, h;
  } i;
} __REAL16_SPLIT;

/* default fortran types (type you get if you just say integer, real,
   etc.  */

#if defined(C90) || defined(T3D) || defined(T3E64)
#define __INT __INT8
#define __LOG __LOG8
#define __REAL __REAL8
#define __DBLE __REAL16
#define __CPLX __CPLX16
#define __DCPLX __CPLX32
typedef __INT8_T __INT_T;
typedef __INT8_T __STAT_T;
typedef __INT8_UT __INT_UT;
typedef __LOG8_T __LOG_T;
typedef __REAL8_T __REAL_T;
typedef __REAL16_T __DBLE_T;
typedef __CPLX16_T __CPLX_T;
typedef __CPLX32_T __DCPLX_T;

#else

#define __INT __INT4
#define __LOG __LOG4
#if defined(DESC_I8)
typedef __INT8_T __INT_T;
#else
typedef __INT4_T __INT_T;
#endif
typedef __INT4_T __STAT_T;
typedef __INT4_UT __INT_UT;
typedef __LOG4_T __LOG_T;

#define __REAL __REAL4
#define __DBLE __REAL8
#define __CPLX __CPLX8
#define __DCPLX __CPLX16
typedef __REAL4_T __REAL_T;
typedef __REAL8_T __DBLE_T;
typedef __REAL16_T __QUAD_T;
typedef __CPLX8_T __CPLX_T;
typedef __CPLX16_T __DCPLX_T;
#endif

/* __BIG's are defined to hold the biggest integer or floating point value
 * likely to be seen during list-directed/namelist/fmt read
 *
 * NOTE: changes here may require changes in format.h for BIGREALs
 */

#define __BIGINT __INT4
#define __BIGLOG __LOG4
typedef __INT4_T __BIGINT_T;
typedef __LOG4_T __BIGLOG_T;

#ifdef TARGET_SUPPORTS_QUADFP
#define __BIGREAL __REAL16
#define __BIGCPLX __CPLX32
typedef __REAL16_T __BIGREAL_T;
typedef __CPLX32_T __BIGCPLX_T;
#else
#define __BIGREAL __REAL8
#define __BIGCPLX __CPLX16
typedef __REAL8_T __BIGREAL_T;
typedef __CPLX16_T __BIGCPLX_T;
#endif

/* pointer-sized integer */

#if   defined(_WIN64)

typedef long long __POINT_T;

/** \def POINT(type, name)
 * \brief Pointer macro
 */
#define POINT(type, name) type *name

#else

typedef __LONG_T __POINT_T;
/** \def POINT(type, name)
 * \brief Pointer macro
 */
#define POINT(type, name) type *name

#endif

#define Is_complex(parm) ((parm) == __CPLX8 || (parm) == __CPLX16)
#define Is_real(parm) ((parm) == __REAL8 || (parm) == __REAL16)

#define REAL_ALLOWED(param) ((Is_complex(param)) || Is_real(param))

/* data type code */

typedef __INT_T dtype;


/*
 * data type representing the number of elements passed to
 * ENTF90(ALLOC04, alloc04), etc.
 */

#define __NELEM_T __INT8_T

#endif /*_PGHPF_TYPES_H_*/
