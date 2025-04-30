/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "dblint64.h"

/*
 * define a C type for long long so that the routines using this type
 * will always compile.  For those systems where long long isn't
 * supported, TM_I8 will not be defined, but at least the run-time routines
 * will compile.
 */

#define __HAVE_LONGLONG_T

#if defined(OSX8664) || defined(TARGET_LLVM_ARM64)
typedef long _LONGLONG_T;
typedef unsigned long _ULONGLONG_T;
#else
typedef long long _LONGLONG_T;
typedef unsigned long long _ULONGLONG_T;
#endif

#define I64_MSH(t) t[1]
#define I64_LSH(t) t[0]

extern int __ftn_32in64_; // Declared in utilsi64.c

#define VOID void

typedef union {
  DBLINT64 i;
  double d;
  _LONGLONG_T lv;
} INT64D;

#if defined(OSX8664) || defined(TARGET_LLVM_ARM64)
#define __I8RET_T long
#define UTL_I_I64RET(m, l)                                                     \
  {                                                                            \
    INT64D int64d;                                                             \
    I64_MSH(int64d.i) = m;                                                     \
    I64_LSH(int64d.i) = l;                                                     \
    return int64d.lv;                                                          \
  }
#elif defined(TARGET_X8664) || defined(_WIN64)
/* Someday, should only care if TM_I8 is defined */
#define __I8RET_T long long
#define UTL_I_I64RET(m, l)                                                     \
  {                                                                            \
    INT64D int64d;                                                             \
    I64_MSH(int64d.i) = m;                                                     \
    I64_LSH(int64d.i) = l;                                                     \
    return int64d.lv;                                                          \
  }
#else
#define __I8RET_T void
#define UTL_I_I64RET __utl_i_i64ret
extern VOID UTL_I_I64RET();
#endif
