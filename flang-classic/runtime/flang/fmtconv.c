/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Utility module for converting the internal representation
 * of a data item to string form.
 *
 * - __fortio_default_convert() - conversion routine which applies the
 *       default editing rules with respect to a data type.
 * - __fortio_fmt_i() - Iw, Iw.m edit descrs.
 * - __fortio_fmt_g() - Gw.d, Gw.dEe
 * - __fortio_fmt_e() - Ew.d, Ew.dEe, ENw.d, ENw.dEe, ESw.d, ESw.dEe,
 * - __fortio_fmt_d() - Dw.d
 * - __fortio_fmt_f() - Fw.d
 */

#include <string.h>
#include "global.h"
#include "format.h"
#include "feddesc.h"
#include "stdioInterf.h"

/* * define a few things for run-time tracing ***/
static int dbgflag;
#undef DBGBIT
#define DBGBIT(v) (LOCAL_DEBUG && (dbgflag & v))

static char *conv_int(__BIGINT_T, int *, int *);
static char *conv_int8(DBLINT64, int *, int *);
static void put_buf(int, const char *, int, int);

static void conv_e(int, int, int, bool, int);
static void conv_en(int, int, bool);
static void conv_es(int, int, bool);
static void conv_f(int, int);
#ifdef FLANG_FMTCONV_UNUSED
static void fp_canon(__BIGREAL_T, int, int);
static void cvtp_round(int);
#endif
static void cvtp_cp(int);
static void cvtp_set(int, int);
static void alloc_fpbuf(int);

static char *strip_blnk(char *, char *);
static char *__fortio_fmt_z(unsigned int);

#define PP_INT1(i) (*(__INT1_T *)(i))
#define PP_INT2(i) (*(__INT2_T *)(i))
#define PP_INT4(i) (*(__INT4_T *)(i))
#define PP_INT8(i) (*(__INT8_T *)(i))
#define PP_LOG1(i) (*(__LOG1_T *)(i))
#define PP_LOG2(i) (*(__LOG2_T *)(i))
#define PP_LOG4(i) (*(__LOG4_T *)(i))
#define PP_LOG8(i) (*(__LOG8_T *)(i))
#define PP_REAL4(i) (__fortio_chk_f((__REAL4_T *)(i)))
#define PP_REAL8(i) (*(__REAL8_T *)(i))
#define PP_REAL16(i) (*(__REAL16_T *)(i))

static int field_overflow;

char __f90io_conv_buf[96] = {0}; /* sufficient size for non-char types - must
                                  * be init'd for 32-bit OSX
                                  */
static char *conv_bufp = __f90io_conv_buf;
static size_t conv_bufsize = sizeof(__f90io_conv_buf);

static char cmplx_buf[64]; /* just for list-directed and nml io */
static char exp_letter = 'E';
static char *buff_pos;

/* ----------------------------------------------------------------- */
void
__fortio_printbigreal(__BIGREAL_T val)
{
#ifdef TARGET_SUPPORTS_QUADFP
  __io_printf("(%Lf)", val);
#else
  __io_printf("(%f)", val);
#endif
}

/* ----------------------------------------------------------------- */

char *
__fortio_default_convert(char *item, int type,
                        int item_length, /* only if character */
                        int *lenp, bool dc_flag, bool plus_flag, int round)
{
  int width;
  char *p;
  DBLINT64 i8val;

  switch (type) {
  default:
    assert(0);
    width = 1;
    strcpy(conv_bufp, " ");
    break;
  case __INT1:
    width = 5;
    (void) __fortio_fmt_i((__BIGINT_T)PP_INT1(item), width, 1, plus_flag);
    break;
  case __INT2:
    width = 7;
    (void) __fortio_fmt_i((__BIGINT_T)PP_INT2(item), width, 1, plus_flag);
    break;
  case __INT8:
    width = 24;
    (void) __fortio_fmt_i8(*(DBLINT64 *)(item), width, 1, plus_flag);
    break;
  case __WORD4:
    width = 8;
    buff_pos = conv_bufp + 8 - 1; /* start at right end of buffer */
    (void) __fortio_fmt_z(*item);
    (void) __fortio_fmt_z(*(item + 1));
    (void) __fortio_fmt_z(*(item + 2));
    (void) __fortio_fmt_z(*(item + 3));
    break;
  case __INT4:
    width = 12;
    (void) __fortio_fmt_i((__BIGINT_T)PP_INT4(item), width, 1, plus_flag);
    break;
  case __REAL4:
    width = REAL4_W;
    (void)__fortio_fmt_g((__BIGREAL_T)PP_REAL4(item), width, REAL4_D, REAL4_E,
                         1, __REAL4, plus_flag, TRUE, dc_flag, round, FALSE);
    break;
  case __WORD8:
    width = 16;
    buff_pos = conv_bufp + 16 - 1; /* start at right end of buffer */
    (void) __fortio_fmt_z(*item);
    (void) __fortio_fmt_z(*(item + 1));
    (void) __fortio_fmt_z(*(item + 2));
    (void) __fortio_fmt_z(*(item + 3));
    (void) __fortio_fmt_z(*(item + 4));
    (void) __fortio_fmt_z(*(item + 5));
    (void) __fortio_fmt_z(*(item + 6));
    (void) __fortio_fmt_z(*(item + 7));
    break;
  case __REAL8:
    width = REAL8_W;
    (void)__fortio_fmt_g((__BIGREAL_T)PP_REAL8(item), width, REAL8_D, REAL8_E,
                         1, __REAL8, plus_flag, TRUE, dc_flag, round, FALSE);
    break;
  case __REAL16:
    width = G_REAL16_W;
    (void)__fortio_fmt_g((__BIGREAL_T)PP_REAL16(item), width, G_REAL16_D,
                         REAL16_E, 1, __REAL16, plus_flag, TRUE, dc_flag, round,
                         TRUE);
    break;
  case __WORD16:
    assert(0);
    width = 0;
    strcpy(conv_bufp, "");
    break;
  case __CPLX8:
    p = cmplx_buf;
    *p++ = '(';
    width = REAL4_W;
    (void)__fortio_fmt_g((__BIGREAL_T)PP_REAL4(item), width, REAL4_D, REAL4_E,
                         1, __REAL4, plus_flag, TRUE, dc_flag, round, FALSE);
    p = strip_blnk(p, conv_bufp);
    if (dc_flag == TRUE)
      *p++ = ';';
    else
      *p++ = ',';
    (void)__fortio_fmt_g((__BIGREAL_T)PP_REAL4(item + 4), width, REAL4_D,
                         REAL4_E, 1, __REAL4, plus_flag, TRUE, dc_flag, round,
                         FALSE);
    p = strip_blnk(p, conv_bufp);
    *p++ = ')';
    *p++ = '\0';
    *lenp = strlen(cmplx_buf);
    return cmplx_buf;
  case __CPLX16:
    p = cmplx_buf;
    *p++ = '(';
    width = REAL8_W;
    (void)__fortio_fmt_g((__BIGREAL_T)PP_REAL8(item), width, REAL8_D, REAL8_E,
                         1, __REAL8, plus_flag, TRUE, dc_flag, round, FALSE);
    p = strip_blnk(p, conv_bufp);
    if (dc_flag == TRUE)
      *p++ = ';';
    else
      *p++ = ',';
    (void)__fortio_fmt_g((__BIGREAL_T)PP_REAL8(item + 8), width, REAL8_D,
                         REAL8_E, 1, __REAL8, plus_flag, TRUE, dc_flag, round,
                         FALSE);
    p = strip_blnk(p, conv_bufp);
    *p++ = ')';
    *p++ = '\0';
    *lenp = strlen(cmplx_buf);
    return cmplx_buf;
  case __CPLX32:
    p = cmplx_buf;
    *p++ = '(';
    width = G_REAL16_W;
    (void)__fortio_fmt_g((__BIGREAL_T)PP_REAL16(item), width, G_REAL16_D,
                         REAL16_E, 1, __REAL16, plus_flag, TRUE, dc_flag, round,
                         TRUE);
    p = strip_blnk(p, conv_bufp);
    if (dc_flag == TRUE)
      *p++ = ';';
    else
      *p++ = ',';
    (void)__fortio_fmt_g((__BIGREAL_T)PP_REAL16(item + 16), width, G_REAL16_D,
                         REAL16_E, 1, __REAL16, plus_flag, TRUE, dc_flag, round,
                         TRUE);
    p = strip_blnk(p, conv_bufp);
    *p++ = ')';
    *p++ = '\0';
    *lenp = strlen(cmplx_buf);
    return cmplx_buf;
  case __LOG1:
    width = 2;
    if (PP_LOG1(item) & GET_FIO_CNFG_TRUE_MASK)
      put_buf(width, "T", 1, 0);
    else
      put_buf(width, "F", 1, 0);
    break;
  case __LOG2:
    width = 2;
    if (PP_LOG2(item) & GET_FIO_CNFG_TRUE_MASK)
      put_buf(width, "T", 1, 0);
    else
      put_buf(width, "F", 1, 0);
    break;
  case __LOG4:
    width = 2;
    if (PP_LOG4(item) & GET_FIO_CNFG_TRUE_MASK)
      put_buf(width, "T", 1, 0);
    else
      put_buf(width, "F", 1, 0);
    break;
  case __LOG8:
    width = 2;
    i8val[0] = PP_LOG4(item);
    i8val[1] = PP_LOG4(item + 4);
    if (I64_LSH(i8val) & GET_FIO_CNFG_TRUE_MASK)
      put_buf(width, "T", 1, 0);
    else
      put_buf(width, "F", 1, 0);
    break;
  case __STR:
    if (item_length >= conv_bufsize) {
      conv_bufsize = item_length + 128;
      if (conv_bufp != __f90io_conv_buf)
        free(conv_bufp);
      conv_bufp = (char *)malloc(conv_bufsize);
    }
    (void) memcpy(conv_bufp, item, item_length);
    conv_bufp[item_length] = '\0';
    width = item_length;
    break;
  } /* end switch */

  *lenp = width;
  return conv_bufp;
}

char *
__fortio_fmt_i(__BIGINT_T val, int width,
              int mn, /* minimum # of digits (Iw.m) */
              bool plus_flag)
{
  char *p;
  int len;
  int neg;  /* flag word set by conv_int, becomes sign character */
  int olen; /* output length of integer value */

  field_overflow = FALSE;
  p = conv_int(val, &len, &neg);

  if (neg)
    neg = '-';
  else if (plus_flag)
    neg = '+';
  olen = len >= mn ? len : mn;
  if (neg)
    olen++;
  if (olen > width) {
    field_overflow = TRUE;
    put_buf(width, p, len, neg);
  } else {
    if (mn == 0 && val == 0) /* Iw.0 gen's blanks if value is 0 */
      neg = 0;
    put_buf(width, p, len, neg);
    if (mn > len) {
      int i;
      i = mn - len;
      p = conv_bufp + (width - len);
      while (i-- > 0)
        *--p = '0';
      if (neg)
        *--p = neg;
    }
  }

  return conv_bufp;
}

static char *
conv_int(__BIGINT_T val, int *lenp, int *negp)
{
#define MAX_CONV_INT 32
#define MAX_HX 0x80000000
#define MAX_STR "2147483648"

  static char tmp[MAX_CONV_INT];
  char *p;
  int len;
  int neg;
  unsigned int n;

  if (val < 0) {
    if (val == MAX_HX) {
      *lenp = 10;
      *negp = 1;
      strcpy(tmp, MAX_STR);
      return tmp;
    }
    neg = 1;
    val = -val;
  } else
    neg = 0;

  p = tmp + MAX_CONV_INT;
  len = 0;
  while (val > 0) {
    n = val / 10;
    *--p = (val - (n * 10)) + '0';
    val = n;
    len++;
  }

  *lenp = len;
  *negp = neg;
  return p;
}

char *
__fortio_fmt_i8(DBLINT64 val,
               int width,
               int mn, /* minimum # of digits (Iw.m) */
               bool plus_flag)
{
  char *p;
  int len;
  int neg;  /* flag word set by conv_int, becomes sign character */
  int olen; /* output length of integer value */

  field_overflow = FALSE;
  p = conv_int8(val, &len, &neg);

  if (neg)
    neg = '-';
  else if (plus_flag)
    neg = '+';
  olen = len >= mn ? len : mn;
  if (neg)
    olen++;
  if (olen > width) {
    field_overflow = TRUE;
    put_buf(width, p, len, neg);
  } else {
    /* Iw.0 gen's blanks if value is 0 */
    if (mn == 0 && val[0] == 0 && val[1] == 0)
      neg = 0;
    put_buf(width, p, len, neg);
    if (mn > len) {
      int i;
      i = mn - len;
      p = conv_bufp + (width - len);
      while (i--)
        *--p = '0';
      if (neg)
        *--p = neg;
    }
  }

  return conv_bufp;
}

static char *
conv_int8(DBLINT64 val, int *lenp, int *negp)
{
#define MAX_CONV_INT8 32

  static char tmp[MAX_CONV_INT8];
  char *p;
  int len;
  DBLINT64 value;

  *negp = 0;
  value[0] = val[0];
  value[1] = val[1];
  if (__ftn_32in64_) {
    if (I64_LSH(value) & 0x80000000)
      I64_MSH(value) = -1;
    else
      I64_MSH(value) = 0;
  } else if (I64_MSH(value) < 0) {
    if (I64_MSH(value) == 0x80000000 && I64_LSH(value) == 0) {
      *lenp = 19;
      *negp = 1;
      strcpy(tmp, "9223372036854775808");
      return tmp;
    }
    *negp = 1;
    /* now negate the value */
    I64_MSH(value) = ~I64_MSH(value);
    I64_LSH(value) = (~I64_LSH(value)) + 1;
    if ((((unsigned)I64_LSH(val) >> 31) == 0) && I64_LSH(value) >= 0)
      I64_MSH(value)++;
  }

  p = tmp;
  len = 24;
  __fort_i64toax(value, p, len, 0, 10);

  *lenp = strlen(p);
  return p;
}

static char fpbuf[64];
static struct {
  int exp;  /* initially set by ecvt/fcvt. adjusted by the
             * scale factor.  WARNING: may be set to zero if
             * value to be printed represents 0.
             */
  int sign; /* non-zero if value is negative (initially set by
             * ecvt/fcvt).  WARNING:  conv_e/conv_f may set to
             * zero if value to be printed represents zero.
             */
  int ndigits;
  int decimal_char;
  bool iszero; /* TRUE if after conversions with respect to the
                * edit descriptor, the string to be printed represents
                * zero. conv_e/conv_f initializes to TRUE; cvtp_cp
                * sets FALSE.
                */
  char *cvtp;
  char *curp;
  char *buf;
  int bufsize;
  __BIGREAL_T zero; /* hide 0.0 from the optimizer here */
} fpdat = {0, 0, 0, '.', 0, 0, 0, fpbuf, sizeof(fpbuf), 0.0};

static void put_buf(int width,        /* where width (# bytes) */
                    const char *valp, /* value in string form */
                    int len,          /* length of value */
                    int sign_char)    /* '-', '+', or 0 (nothing to prepend) */
{
  char *bufp;
  int cnt;
  int sign_cnt;

  if (DBGBIT(0x1))
    __io_printf("put_buf: width=%d, len=%d, val=%.*s#, sign_char=%d\n", width,
                 len, len, valp, sign_char);
  if (width >= conv_bufsize) {
    conv_bufsize = width + 128;
    if (conv_bufp != __f90io_conv_buf)
      free(conv_bufp);
    conv_bufp = (char *)malloc(conv_bufsize);
  }
  bufp = conv_bufp;
  if (width == 0) {
    /* short circuit for if no transfer - do we need error message */
    *bufp = '\0';
    return;
  }
  if (field_overflow)
    goto fill_asterisks;
  sign_cnt = sign_char != 0;
  if (len + sign_cnt > width) {
    /*
     * special floating point case: if value begins with "0.",
     * try ignoring the '0'.
     */
    if (*valp != '0' || *(valp + 1) != fpdat.decimal_char ||
        (len + sign_cnt - 1) > width)
      goto fill_asterisks;
    len--;
    valp++;
  }
  cnt = width - len - sign_cnt;
  while (cnt-- > 0)
    *bufp++ = ' ';
  if (sign_char)
    *bufp++ = sign_char;
  cnt = len;
  while (cnt-- > 0)
    *bufp++ = *valp++;
  *bufp = '\0';
  return;

fill_asterisks:
  (void) memset(bufp, '*', width);
  bufp += width;
  *bufp = '\0';
  field_overflow = FALSE;
}

/* ------------------------------------------------------------------- */

extern char *
__fortio_fmt_d(__BIGREAL_T val, int w, int d, int sf, int type, bool plus_flag,
              int round)
{
  int sign_char;
  int newd;

  exp_letter = 'D';
  field_overflow = FALSE;
  /*
      fp_canon(val, type, round);
  */
  if ((sf < 0) && (-d) >= sf) {
    field_overflow = TRUE;
    put_buf(w, (char *)0, 0, 0);
    exp_letter = 'E'; /* Make sure to change this back! */
    return conv_bufp;
  }
  newd = d + ((sf > 0) ? 1 : sf);
  fpdat.cvtp = __io_ecvt(val, w, newd, &fpdat.exp, &fpdat.sign, round, FALSE);
  fpdat.ndigits = strlen(fpdat.cvtp);
  fpdat.curp = fpdat.buf;

  if (DBGBIT(0x4)) {
    __io_printf("fio_fmt_d ");
    __fortio_printbigreal(val);
    __io_printf(" = #%s#, exp=%d, sign=%d, sf=%d, pf=%d\n", fpdat.cvtp,
                 fpdat.exp, fpdat.sign, sf, plus_flag);
    __io_printf("          w=%d, d=%d\n", w, d);
  }
  if (*fpdat.cvtp < '0' || *fpdat.cvtp > '9') {
    if (fpdat.sign)
      sign_char = '-';
    else if (plus_flag)
      sign_char = '+';
    else
      sign_char = 0;
    put_buf(w, fpdat.cvtp, fpdat.ndigits, sign_char);
  } else {
    conv_e(d, 2, sf, FALSE, FALSE);
    if (fpdat.sign) /* must check after conv_e */
      sign_char = '-';
    else if (plus_flag)
      sign_char = '+';
    else
      sign_char = 0;
    put_buf(w, fpdat.buf, (int)(fpdat.curp - fpdat.buf), sign_char);
  }
  exp_letter = 'E';
  return conv_bufp;
}

extern char *
__fortio_fmt_g(__BIGREAL_T val, int w, int d, int e, int sf, int type,
               bool plus_flag, bool e_flag, bool dc_flag, int round,
               int is_quad) /* TRUE, if the value is quad precision. */
{
  int sign_char;
  int newd;
#if defined(TARGET_X8664)
  /*
   * the following guarded IF may look like a no-op, but is
   * needed when val is a denorm and DAZ is enabled.  In this case, the
   * comparison will say val is identical to 0, but the bits of val will
   * indicate otherwise and the ensuing code may go down the wrong path.
   */
  if (val == fpdat.zero && !is_quad) {
    union {
      __BIGREAL_T vv;
      int ii[2];
    } u;
    u.vv = val;
    val = fpdat.zero;
    if (u.ii[1] < 0) {
      ((int *)&val)[1] |= 0x80000000;
    }
  }
#endif
  field_overflow = FALSE;
  /*
      fp_canon(val, type, round);
  */
  if ((sf < 0) && (-d) >= sf) {
    field_overflow = TRUE;
    put_buf(w, (char *)0, 0, 0);
    return conv_bufp;
  }
  if (!is_quad)
    newd = d + ((sf > 0) ? 1 : sf);
  else
    newd = d + ((sf > 0) ? 0 : sf - 1);
  fpdat.cvtp = __io_ecvt(val, w, newd, &fpdat.exp, &fpdat.sign, round, is_quad);
  fpdat.ndigits = strlen(fpdat.cvtp);
  fpdat.curp = fpdat.buf;

  if (DBGBIT(0x4)) {
    __io_printf("fio_fmt_g ");
    __fortio_printbigreal(val);
    __io_printf(" = #%s#, exp=%d, sign=%d, sf=%d, pf=%d, ef=%d\n", fpdat.cvtp,
                 fpdat.exp, fpdat.sign, sf, plus_flag, e_flag);
    __io_printf("          w=%d, d=%d\n", w, d);
  }
  if (dc_flag == TRUE)
    fpdat.decimal_char = ',';
  else
    fpdat.decimal_char = '.';
  if (*fpdat.cvtp < '0' || *fpdat.cvtp > '9') {
    if (fpdat.sign)
      sign_char = '-';
    else if (plus_flag)
      sign_char = '+';
    else
      sign_char = 0;
    put_buf(w, fpdat.cvtp, fpdat.ndigits, sign_char);
  } else if ((val != (__BIGREAL_T)0.0) &&
             (*fpdat.cvtp == '0' || fpdat.exp < 0 || fpdat.exp >= d + 1)) {
    /*  m  .lt. 0.1  or  m .ge. 10**d  */
    /*  Ew.dEe */
    conv_e(d, e, sf, e_flag, is_quad);
    if (fpdat.sign) /* must check after conv_e */
      sign_char = '-';
    else if (plus_flag)
      sign_char = '+';
    else
      sign_char = 0;
    put_buf(w, fpdat.buf, (int)(fpdat.curp - fpdat.buf), sign_char);
  } else {
    char *p;
    int m, n;
    int texp;
    int ww, dd;
    /*  Fm.<d-exp><n blanks>, where  n = e + 2, m = w - n  */
    n = e + 2;
    m = w - n;
    if (*fpdat.cvtp == '0') {
      fpdat.exp = 1;
    }
    /* switching to F, scale factor is suppressed */
    /*
     * use fcvt to convert value correctly rounded to a certain number of
     * digits right or left of the decimal point.  The last digit rounded
     * is the sum of the value of 'd'.
     * If this value is positive, then the digit is to the right of '.';
     * negative => to the left of '.'.
     */
    fpdat.cvtp =
        __io_fcvt(val, w, d - fpdat.exp, 0, &texp, &fpdat.sign, round, is_quad);
    if (val == (__BIGREAL_T)0.0) {
      texp = fpdat.exp;
    } else if (texp != fpdat.exp) {
      fpdat.exp = texp;
      fpdat.cvtp =
          __io_fcvt(val, w, d - texp, 0, &texp, &fpdat.sign, round, is_quad);
    }
    fpdat.ndigits = strlen(fpdat.cvtp);
    ww = m;
    dd = d - texp;
    if (DBGBIT(0x4)) {
      __io_printf("gfmt_f ");
      __fortio_printbigreal(val);
      __io_printf(" = #%s#, exp=%d, sign=%d, sf=%d, pf=%d\n", fpdat.cvtp,
                   fpdat.exp, fpdat.sign, 0, plus_flag);
      __io_printf("          w=%d, d=%d\n", ww, dd);
    }
    if (*fpdat.cvtp < '0' || *fpdat.cvtp > '9') {
      if (fpdat.sign)
        sign_char = '-';
      else if (plus_flag)
        sign_char = '+';
      else
        sign_char = 0;
      put_buf(ww, fpdat.cvtp, fpdat.ndigits, sign_char);
    } else {
      conv_f(ww, dd);
      if (fpdat.sign) /* must be checked after conv_f */
        sign_char = '-';
      else if (plus_flag)
        sign_char = '+';
      else
        sign_char = 0;
      put_buf(ww, fpdat.buf, fpdat.curp - fpdat.buf, sign_char);
    }

    p = conv_bufp;
    p += m;
    while (n-- > 0)
      *p++ = ' ';
    *p = '\0';
  }
  return conv_bufp;
}

extern char *
__fortio_fmt_e(__BIGREAL_T val, int w, int d, int e, int sf, int type,
              bool plus_flag, bool e_flag, bool dc_flag, int code, int round)
{
  int sign_char;
  int newd, newrnd;

  field_overflow = FALSE;

  /* Replace this call
      fp_canon(val, type, round);
  */
  if (code == FED_ENw_d) {
    /* Need to add nc digits, but nc is unknown now */
    newd = d + 3;
    newrnd = round + 256;
  } else if (code == FED_ESw_d) {
    newd = d + 1;
    newrnd = round;
  } else {
    if ((sf < 0) && (-d) >= sf) {
      field_overflow = TRUE;
      put_buf(w, (char *)0, 0, 0);
      return conv_bufp;
    }
    newd = d + ((sf > 0) ? 1 : sf);
    newrnd = round;
  }

  fpdat.cvtp = __io_ecvt(val, w, newd, &fpdat.exp, &fpdat.sign, newrnd, FALSE);
  fpdat.ndigits = strlen(fpdat.cvtp);
  fpdat.curp = fpdat.buf;

  if (dc_flag == TRUE)
    fpdat.decimal_char = ',';
  else
    fpdat.decimal_char = '.';
  if (DBGBIT(0x4)) {
    __io_printf("fio_fmt_e ");
    __fortio_printbigreal(val);
    __io_printf(" = #%s#, exp=%d, sign=%d, sf=%d, pf=%d, ef=%d\n", fpdat.cvtp,
                 fpdat.exp, fpdat.sign, sf, plus_flag, e_flag);
    __io_printf("          w:%d, d:%d, e:%d\n", w, d, e);
  }
  if (*fpdat.cvtp < '0' || *fpdat.cvtp > '9') {
    if (fpdat.sign)
      sign_char = '-';
    else if (plus_flag)
      sign_char = '+';
    else
      sign_char = 0;
    put_buf(w, fpdat.cvtp, fpdat.ndigits, sign_char);
  } else {
    if (code == FED_ENw_d) {
      conv_en(d, e, e_flag);
    } else if (code == FED_ESw_d) {
      conv_es(d, e, e_flag);
    } else {
      conv_e(d, e, sf, e_flag, FALSE);
    }
    if (fpdat.sign) /* must check after conv_e */
      sign_char = '-';
    else if (plus_flag)
      sign_char = '+';
    else
      sign_char = 0;
    put_buf(w, fpdat.buf, (int)(fpdat.curp - fpdat.buf), sign_char);
  }
  return conv_bufp;
}

extern char *
__fortio_fmt_f(__BIGREAL_T val, int w, int d,
              int sf, /* printed value is val * 10**sf */
              bool plus_flag, bool dc_flag, int round)
{
  int sign_char;
  void *p;

  field_overflow = FALSE;
  /*
   * use fcvt to convert value correctly rounded to a certain number of
   * digits right or left of the decimal point.  The last digit rounded
   * is the sum of the value of 'd' and the current scale factor in effect.
   * If this value is positive, then the digit is to the right of '.';
   * negative => to the left of '.'.
   */
  fpdat.cvtp = __io_fcvt(val, w, d, sf, &fpdat.exp, &fpdat.sign, round, FALSE);

  if (dc_flag == TRUE)
    fpdat.decimal_char = ',';
  else
    fpdat.decimal_char = '.';
  p = fpdat.cvtp; /* conf_f walks on cvtp */
  fpdat.ndigits = strlen(fpdat.cvtp);
  if (DBGBIT(0x4)) {
    __io_printf("fio_fmt_f ");
    __fortio_printbigreal(val);
    __io_printf(" = #%s#, exp=%d, sign=%d, sf=%d, pf=%d\n", fpdat.cvtp,
                 fpdat.exp, fpdat.sign, sf, plus_flag);
    __io_printf("          w=%d, d=%d\n", w, d);
  }
  if (*fpdat.cvtp < '0' || *fpdat.cvtp > '9') {
    if (fpdat.sign)
      sign_char = '-';
    else if (plus_flag)
      sign_char = '+';
    else
      sign_char = 0;
    put_buf(w, fpdat.cvtp, fpdat.ndigits, sign_char);
  } else {
    fpdat.exp += sf; /* exp changed to adjust printed value */
    conv_f(w, d);
    if (fpdat.sign) /* must be checked after conv_f */
      sign_char = '-';
    else if (plus_flag)
      sign_char = '+';
    else
      sign_char = 0;
    put_buf(w, fpdat.buf, (int)(fpdat.curp - fpdat.buf), sign_char);
  }
  return conv_bufp;
}

/* ------------------------------------------------------------------- */

static void
conv_e(int d, int e, int sf, bool e_flag, /* TRUE, if Ee was specified */
       int is_quad)                       /* TRUE, if the value is quad */
{
  char *p;
  int len, neg;

  fpdat.iszero = TRUE;

  /* Maximum space required for fpdat.buf:
   *     "O." + <d>digits + "E+/-" <e>digits + "\0"
   */
  alloc_fpbuf(d + e + 5);

  if (sf == 0) {
    /*  0 . <d digits>  */
    /* cvtp_round(d);  */
    *fpdat.curp++ = '0';
    *fpdat.curp++ = fpdat.decimal_char;
    cvtp_cp(d);
  } else if (sf > 0 && sf < (d + 2)) {
    /*  <sf digits> . <(d - sf + 1) digits>  */
    /* cvtp_round(d+1); */
    cvtp_cp(sf);
    *fpdat.curp++ = fpdat.decimal_char;
    if (!is_quad)
      cvtp_cp(d - sf + 1);
    else
      cvtp_cp(d - sf);
  } else if (sf < 0 && (-d) < sf) {
    /*  0 . <|sf| 0's> <(d - |sf|) digits>  */
    /* cvtp_round(d + sf); */
    *fpdat.curp++ = '0';
    *fpdat.curp++ = fpdat.decimal_char;
    cvtp_set(-sf, '0');
    cvtp_cp(d + sf); /* reduce # of digits to copy */
  } else {
    /* scale factor error */
    /* BDL: I don't think we should be printing this */
    /* __io_printf("conv_e: illegal scale factor\n"); */
    field_overflow = TRUE;
    *fpdat.curp = '\0';
    return;
  }

  if (fpdat.iszero) {
    fpdat.exp = 0;
    if (__fortio_no_minus_zero()) {
      fpdat.sign = 0;
    }
  } else {
    fpdat.exp -= sf;
  }

  p = conv_int((__BIGINT_T)fpdat.exp, &len, &neg);
  if (e) {
    if (!e_flag && len == e + 1) /* don't check if Ee specified */
      e++; /* don't store exp letter, allow for another digit */
    else
      *fpdat.curp++ = exp_letter;
  } else if (len <= 2)
    *fpdat.curp++ = exp_letter;
  if (neg)
    *fpdat.curp++ = '-';
  else
    *fpdat.curp++ = '+';
  if (len > e)
    field_overflow = TRUE;
  else {
    cvtp_set(e - len, '0');
    while (len--)
      *fpdat.curp++ = *p++;
  }
  *fpdat.curp = '\0';
}

static void
conv_en(int d, int e, bool e_flag) /* TRUE, if Ee was specified */
{
  char *p;
  int len, neg;
  int ne, nc;
  int old_exp;
  char old_buf[64]; /* must be >= sizeof(buf) in __io_ecvt (ECVTSIZE) */

  strcpy(old_buf, fpdat.cvtp);

  /* Maximum required for fpdat.buf:
   *     "yyy." + <d>digits + "E+/-" <e>digits + "\0"
   */
  alloc_fpbuf(d + e + 7);

  fpdat.iszero = (*fpdat.cvtp == '0') ? TRUE : FALSE;
  if (fpdat.exp > 0) {
    ne = (fpdat.exp - 1) / 3 * 3;
    nc = fpdat.exp - ne;
  } else if (fpdat.exp < 0 || !fpdat.iszero) {
    ne = (fpdat.exp / 3 - 1) * 3;
    nc = ne - fpdat.exp;
    if (nc < 0)
      nc = -nc; /* need abs value */
  } else
    nc = 1;
  old_exp = fpdat.exp;
  while (nc-- > 0)
    *fpdat.curp++ = *fpdat.cvtp++;
  *fpdat.curp++ = fpdat.decimal_char;
  cvtp_cp(d);

  if (fpdat.iszero) {
    fpdat.exp = 0;
    if (__fortio_no_minus_zero()) {
      fpdat.sign = 0;
    }
  } else {
    fpdat.exp = ne;
  }
  p = conv_int((__BIGINT_T)fpdat.exp, &len, &neg);
  if (e) {
    if (!e_flag && len == e + 1) /* don't check if Ee specified */
      e++; /* don't store exp letter, allow for another digit */
    else
      *fpdat.curp++ = exp_letter;
  } else if (len <= 2)
    *fpdat.curp++ = exp_letter;
  if (neg)
    *fpdat.curp++ = '-';
  else
    *fpdat.curp++ = '+';
  if (len > e)
    field_overflow = TRUE;
  else {
    cvtp_set(e - len, '0');
    while (len--)
      *fpdat.curp++ = *p++;
  }
  *fpdat.curp = '\0';
}

static void
conv_es(int d, int e, bool e_flag) /* TRUE, if Ee was specified */
{
  char *p;
  int len, neg;

  /* Maximum space required for fpdat.buf:
   *     "y." + <d>digits + "E+/-" <e>digits + "\0"
   */
  alloc_fpbuf(d + e + 5);

  fpdat.iszero = (*fpdat.cvtp == '0') ? TRUE : FALSE;

  /* ABR.  Already Been Rounded */
  /* cvtp_round(d+1); */
  *fpdat.curp++ = *fpdat.cvtp++;
  *fpdat.curp++ = fpdat.decimal_char;
  cvtp_cp(d);

  if (fpdat.iszero) {
    fpdat.exp = 0;
    if (__fortio_no_minus_zero()) {
      fpdat.sign = 0;
    }
  } else {
    fpdat.exp -= 1;
  }
  p = conv_int((__BIGINT_T)fpdat.exp, &len, &neg);
  if (e) {
    if (!e_flag && len == e + 1) /* don't check if Ee specified */
      e++; /* don't store exp letter, allow for another digit */
    else
      *fpdat.curp++ = exp_letter;
  } else if (len <= 2)
    *fpdat.curp++ = exp_letter;
  if (neg)
    *fpdat.curp++ = '-';
  else
    *fpdat.curp++ = '+';
  if (len > e)
    field_overflow = TRUE;
  else {
    cvtp_set(e - len, '0');
    while (len--)
      *fpdat.curp++ = *p++;
  }
  *fpdat.curp = '\0';
}

static void
conv_f(int w, int d)
{
  int lh;
  int i;

  fpdat.iszero = TRUE;

  /* Maximum space required for fpdat.buf:
   *    <w> + "\0"
   */
  alloc_fpbuf(w + 1);

  lh = w - d - 1; /* space left of . */
  if (DBGBIT(0x2))
    __io_printf("conv_f: w=%d, d=%d, exp=%d, lh=%d\n", w, d, fpdat.exp, lh);
  if (fpdat.exp > 0) {
    /* remove any leading zeros */
    while (*fpdat.cvtp == '0') {
      fpdat.cvtp++;
      fpdat.exp--;
      fpdat.ndigits--;
    }
    if (*fpdat.cvtp == '\0') {
      fpdat.exp = 0; /* value is zero */
      if (__fortio_no_minus_zero()) {
        fpdat.sign = 0;
      }
    }
    if (DBGBIT(0x2))
      __io_printf("conv_f: #%s# exp=%d\n", fpdat.cvtp, fpdat.exp);
  }
  if (fpdat.exp > lh)
    field_overflow = TRUE;
  else if (fpdat.exp <= 0) {
    *fpdat.curp++ = '0';
    *fpdat.curp++ = fpdat.decimal_char;
    i = -fpdat.exp;
    if (i > d)
      i = d;
    cvtp_set(i, '0');
    cvtp_cp(d - i);
  } else {
    cvtp_cp(fpdat.exp);
    *fpdat.curp++ = fpdat.decimal_char;
    cvtp_cp(d);
  }
  *fpdat.curp = '\0';
  /*  fcvs fm111 test 1 behaves differently when setting sign to 0;
   *  -0.0044 written as f2.1 is now '**' instead of '.0' -- I fixed
   *  the test to write +0.0044
   *  fcvs fm506 test 3 behaves differently for -0.0001 written as f4.1;
   *  added '-0.0' as an allowed output
   */
  if (__fortio_no_minus_zero()) {
    if (fpdat.iszero) {
      fpdat.sign = 0;
    }
  }
}

static char hextab[17] = "0123456789ABCDEF";

static char *
__fortio_fmt_z(unsigned int c)
{
  *buff_pos = hextab[c & 0xF];
  *(buff_pos - 1) = hextab[(c >> 4) & 0xF];
  buff_pos -= 2;
  return buff_pos;
}

#ifdef FLANG_FMTCONV_UNUSED
static void
fp_canon(__BIGREAL_T val, int type, int round)
{
  if (type == __REAL4)
    fpdat.cvtp = __io_ecvt(val, REAL4_W, REAL4_D + 1, &fpdat.exp, &fpdat.sign,
                           round, FALSE);
  else if (type == __REAL8)
    fpdat.cvtp = __io_ecvt(val, REAL8_W, REAL8_D + 1, &fpdat.exp, &fpdat.sign,
                           round, FALSE);
  else
    fpdat.cvtp = __io_ecvt(val, G_REAL16_W, G_REAL16_D + 1, &fpdat.exp,
                           &fpdat.sign, round, FALSE);
  if (DBGBIT(0x2)) {
    __io_printf("fp_canon ");
    __fortio_printbigreal(val);
    __io_printf(" = #%s#, exp=%d, sign=%d\n", fpdat.cvtp, fpdat.exp,
                 fpdat.sign);
  }
  fpdat.ndigits = strlen(fpdat.cvtp);
  fpdat.curp = fpdat.buf;
}

/*
 * d digits are going to be copied from the canonical (ecvt) form.
 * need to round the digits before the (d+1)th digit.  If all digits
 * are rounded (i.e., .9999... becomes 1.0000..., the first digit of
 * the cvtp buffer is changed to a '1' and the exponent is incremented.
 */
static void
cvtp_round(int d)
{
  char *p;
  char digit;
  int r;

  if (d <= 0 || fpdat.ndigits <= d)
    return;
  p = fpdat.cvtp + d;
  if (*p >= '5') { /* begin rounding */
    r = d;
    do {
      digit = *--p;
      if (digit == '9')
        *p = '0';
      else {
        *p = (char)(digit + 1);
        break;
      }
    } while (--r);
    if (r == 0) { /* all digits were rounded -- all 9's */
      fpdat.cvtp[0] = '1';
      fpdat.exp++;
    }
    if (DBGBIT(0x2))
      __io_printf("cvtp_round: #%s#  exp: %d \n", fpdat.cvtp, fpdat.exp);
  }

}
#endif

static void
cvtp_cp(int n)
{
  while (n) {
    char ch;
    if (*fpdat.cvtp == '\0') {
      break;
    }
    ch = *fpdat.cvtp++;
    if (ch != '0')
      fpdat.iszero = FALSE;
    *fpdat.curp++ = ch;
    n--;
  }
  while (n-- > 0)
    *fpdat.curp++ = '0';
}

static void
cvtp_set(int n, int ch)
{
  while (n-- > 0)
    *fpdat.curp++ = ch;
}

static void
alloc_fpbuf(int n)
{
  if (n > fpdat.bufsize) {
    fpdat.bufsize = n + 32;
    if (fpdat.buf != fpbuf)
      free(fpdat.buf);
    fpdat.buf = malloc(fpdat.bufsize);
  }
  fpdat.curp = fpdat.buf;
}

__BIGREAL_T
__fortio_chk_f(__REAL4_T *f)
{
  union {
    int i;
    float f;
  } uf;
  union {
    int i[2];
    double d;
  } u;
  static int chk = 0x01020304; /* endian test at run-time */

  uf.f = *f;

  if ((uf.i & 0x7F800000) == 0x7F800000) {
    if (*(char *)&chk == 0x01) {
      u.i[1] = uf.i & ~0xFF800000;
      u.i[0] = 0x7FF00000 | (uf.i & 0x80000000);
    } else {
      u.i[0] = uf.i & ~0xFF800000;
      u.i[1] = 0x7FF00000 | (uf.i & 0x80000000);
    }
    return u.d;
  }
  return (__BIGREAL_T)*f;
}

static char *
strip_blnk(char *to, char *from)
{
  char c;
  /* skip leading blanks */
  while (*from == ' ')
    from++;

  /* copy until blanks or null char */
  while ((c = *from++) != 0) {
    if (c == ' ')
      break;
    *to++ = c;
  }
  *to = '\0';
  return to; /* leave ptr at null char */
}
