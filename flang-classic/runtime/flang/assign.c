/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "global.h"
#include <string.h>

/** \file
 *  RunTime routines to handle namelist and list directed reads
 */

#define PP_INT1(i) (*(__INT1_T *)(i))
#define PP_INT2(i) (*(__INT2_T *)(i))
#define PP_INT4(i) (*(__INT4_T *)(i))
#define PP_INT8(i) (*(__INT8_T *)(i))
#define PP_LOG1(i) (*(__LOG1_T *)(i))
#define PP_LOG2(i) (*(__LOG2_T *)(i))
#define PP_LOG4(i) (*(__LOG4_T *)(i))
#define PP_LOG8(i) (*(__LOG8_T *)(i))
#define PP_REAL4(i) (*(__REAL4_T *)(i))
#define PP_REAL8(i) (*(__REAL8_T *)(i))
#define PP_REAL16(i) (*(__REAL16_T *)(i))

static __BIGINT_T to_bigint(AVAL *);
static __BIGREAL_T to_bigreal(AVAL *);
static __INT8_T to_int8(AVAL *);
static __LOG8_T to_log8(AVAL *);

/* ---------------------------------------------------------------- */
/** \brief
 * __fortio_assign  -  store a value found during list-directed or
 *		namelist read.  return 0 if successful, otherwise an
 *		error code (typically, a conversion error).
 */
int
__fortio_assign(char *item,           /* where to store */
                int type,             /* data type of item (as in pghpft.h) */
                __CLEN_T item_length, /* number of chars if type == __STR */
                AVAL *valp            /* value to store */
)
{
  __CLEN_T len;
  AVAL *lp;

  switch (valp->dtype) {
  case __BIGCPLX:
    lp = valp->val.cmplx;
    goto assn_shared;
  case __INT8:
    if (__ftn_32in64_)
      I64_MSH(valp->val.i8) = 0;
    FLANG_FALLTHROUGH;
  case __BIGINT:
  case __BIGREAL:
    lp = valp;
  assn_shared:
    switch (type) {
    default:
      assert(0);
      break;
    case __INT1:
      PP_INT1(item) = to_bigint(lp);
      break;
    case __INT2:
      PP_INT2(item) = to_bigint(lp);
      break;
    case __INT4:
      PP_INT4(item) = to_bigint(lp);
      break;
    case __INT8:
      PP_INT8(item) = to_int8(lp);
      break;
    case __REAL4:
      PP_REAL4(item) = (__REAL4_T)to_bigreal(lp);
      break;
    case __REAL8:
      PP_REAL8(item) = (__REAL8_T)to_bigreal(lp);
      break;
    case __REAL16:
      PP_REAL16(item) = (__REAL16_T)to_bigreal(lp);
      break;
    case __WORD16:
      return FIO_EQUAD;
    case __CPLX8:
      PP_REAL4(item) = (__REAL4_T)to_bigreal(lp);
      if (valp->dtype == __BIGCPLX)
        PP_REAL4(item + (sizeof(__CPLX8_T) >> 1)) =
            (__REAL4_T)to_bigreal(lp + 1);
      else
        PP_REAL4(item + (sizeof(__CPLX8_T) >> 1)) = 0;
      break;
    case __CPLX16:
      PP_REAL8(item) = (__REAL8_T)to_bigreal(lp);
      if (valp->dtype == __BIGCPLX)
        PP_REAL8(item + (sizeof(__CPLX16_T) >> 1)) =
            (__REAL8_T)to_bigreal(lp + 1);
      else
        PP_REAL8(item + (sizeof(__CPLX16_T) >> 1)) = 0;
      break;
    case __CPLX32:
      PP_REAL16(item) = (__REAL16_T)to_bigreal(lp);
      if (valp->dtype == __BIGCPLX)
        PP_REAL16(item + (sizeof(__CPLX32_T) >> 1)) =
            (__REAL16_T)to_bigreal(lp + 1);
      else
        PP_REAL16(item + (sizeof(__CPLX32_T) >> 1)) = 0;
      break;
    case __LOG1:
      PP_LOG1(item) = to_bigint(lp);
      break;
    case __LOG2:
      PP_LOG2(item) = to_bigint(lp);
      break;
    case __LOG4:
      PP_LOG4(item) = to_bigint(lp);
      break;
    case __LOG8:
      PP_LOG8(item) = to_log8(lp);
      break;
    case __STR:
    case __NCHAR:
      goto assn_err;
    }
    break;

  case __BIGLOG:
    switch (type) {
    default:
      assert(0);
      break;
    case __INT1:
      PP_INT1(item) = valp->val.i;
      break;
    case __INT2:
      PP_INT2(item) = valp->val.i;
      break;
    case __INT4:
      PP_INT4(item) = valp->val.i;
      break;
    case __INT8:
      PP_INT8(item) = valp->val.i;
      break;
    case __REAL4:
      PP_REAL4(item) = (__REAL4_T)valp->val.i;
      break;
    case __REAL8:
      PP_REAL8(item) = (__REAL8_T)valp->val.i;
      break;
    case __REAL16:
      PP_REAL16(item) = (__REAL16_T)valp->val.i;
      break;
    case __WORD16:
      return FIO_EQUAD;
    case __CPLX8:
      PP_REAL4(item) = (__REAL4_T)valp->val.i;
      PP_REAL4(item + (sizeof(__CPLX8_T) >> 1)) = 0;
      break;
    case __CPLX16:
      PP_REAL8(item) = valp->val.i;
      PP_REAL8(item + (sizeof(__CPLX16_T) >> 1)) = 0;
      break;
    case __CPLX32:
      PP_REAL16(item) = valp->val.i;
      PP_REAL16(item + (sizeof(__CPLX32_T) >> 1)) = 0;
      break;
    case __LOG1:
      PP_LOG1(item) = valp->val.i;
      break;
    case __LOG2:
      PP_LOG2(item) = valp->val.i;
      break;
    case __LOG4:
      PP_LOG4(item) = valp->val.i;
      break;
    case __LOG8:
      PP_LOG8(item) = to_log8(valp);
      break;
    case __STR:
    case __NCHAR:
      goto assn_err;
    }
    break;

  case __LOG8:
    switch (type) {
    default:
      assert(0);
      break;
    case __INT1:
      PP_INT1(item) = I64_LSH(valp->val.i8);
      break;
    case __INT2:
      PP_INT2(item) = I64_LSH(valp->val.i8);
      break;
    case __INT4:
      PP_INT4(item) = I64_LSH(valp->val.i8);
      break;
    case __INT8:
      if (__ftn_32in64_)
        I64_MSH(valp->val.i8) = 0;
      PP_INT4(item) = valp->val.i8[0];
      PP_INT4(item + 4) = valp->val.i8[1];
      break;
    case __REAL4:
      PP_REAL4(item) = (float)I64_LSH(valp->val.i8);
      break;
    case __REAL8:
      PP_REAL8(item) = I64_LSH(valp->val.i8);
      break;
    case __REAL16:
    case __WORD16:
      return FIO_EQUAD;
    case __CPLX8:
      PP_REAL4(item) = (float)I64_LSH(valp->val.i8);
      PP_REAL4(item + 4) = 0;
      break;
    case __CPLX16:
      PP_REAL8(item) = I64_LSH(valp->val.i8);
      PP_REAL8(item + 8) = 0;
      break;
    case __CPLX32:
      return FIO_EQUAD;
    case __LOG1:
      PP_LOG1(item) = I64_LSH(valp->val.i8);
      break;
    case __LOG2:
      PP_LOG2(item) = I64_LSH(valp->val.i8);
      break;
    case __LOG4:
      PP_LOG4(item) = I64_LSH(valp->val.i8);
      break;
    case __LOG8:
      PP_LOG8(item) = to_log8(valp);
      break;
    case __STR:
    case __NCHAR:
      goto assn_err;
    }
    break;

  case __STR:
    if (type != __STR)
      goto assn_err;
    len = valp->val.c.len;
    if (len > item_length)
      len = item_length;
    (void)memcpy(item, valp->val.c.str, len);
    if (len < item_length)
      (void)memset(item + len, ' ', item_length - len);
    break;

  default:
    assert(0);
    break;
  }

  return 0;
assn_err:
  return FIO_EERR_DATA_CONVERSION;
}

/* ------------------------------------------------------------------ */

static __BIGINT_T
to_bigint(AVAL *valp)
{
  if (valp->dtype == __BIGREAL)
    return (__BIGINT_T)valp->val.d;
  assert(valp->dtype == __BIGINT || valp->dtype == __BIGLOG);
  return (__BIGINT_T)valp->val.i;
}

static __BIGREAL_T
to_bigreal(AVAL *valp)
{
  if (valp->dtype == __BIGREAL)
    return valp->val.d;
  if (valp->dtype == __INT8 || valp->dtype == __LOG8) {
    return (__BIGREAL_T)valp->val.i8v;
  }
  assert(valp->dtype == __BIGINT || valp->dtype == __BIGLOG);
  return (__BIGREAL_T)valp->val.i;
}

static __INT8_T
to_int8(AVAL *valp)
{
  if (valp->dtype == __INT4 || valp->dtype == __LOG4) {
    I64_LSH(valp->val.i8) = valp->val.i;
    if (valp->val.i < 0)
      I64_MSH(valp->val.i8) = 0xFFFFFFFF;
    else
      I64_MSH(valp->val.i8) = 0;
  } else if (valp->dtype == __BIGREAL) {
    return (__INT8_T)valp->val.d;
  }
  return valp->val.i8v;
}

static __LOG8_T
to_log8(AVAL *valp)
{
  if (valp->dtype == __INT4 || valp->dtype == __LOG4) {
    I64_LSH(valp->val.ui8) = valp->val.i;
    I64_MSH(valp->val.ui8) = 0;
  }
  return (__LOG8_T)valp->val.ui8v;
}
