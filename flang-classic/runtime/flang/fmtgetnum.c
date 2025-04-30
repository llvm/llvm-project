/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Utility module for extracting numerical values from a string.
 */

/*
 *  __fortio_getnum() - extracts integer or __BIGREAL_T scalar values from
 *	a string.  If successful, 0 is returned; otherwise, an i/o
 *	error code is returned.  Output arguments are used to pass
 *	back a flag indicating integer or __BIGREAL_T, the value of the
 *	constant, and the number of characters processed for the numeric.
 */

#include "global.h"
#include "format.h"

/* define a few things for run-time tracing */
static int dbgflag;
#undef DBGBIT
#define DBGBIT(v) (LOCAL_DEBUG && (dbgflag & v))

static char buf[128];
static char *buf_p = buf;
static int buf_size = sizeof(buf);

/*
 *  __fortio_getnum() - extracts integer or __BIGREAL_T scalar values from
 *	a string.  If successful, 0 is returned; otherwise, an i/o
 *	error code is returned.  Output arguments are used to pass
 *	back a flag indicating integer or __BIGREAL_T, the value of the
 *	constant, and the number of characters processed for the numeric.
 */
int
__fortio_getnum(
    char *currc, /* pointer to string to convert */
    int *type,   /* 0 ==> integer, 1 ==> __BIGREAL_T
                  * 2 ==> integer*8 (TM_I8-only).
                  * 3 ==> VMS degenerate REAL (e.g., 'd6', '-d6', 'e',...)
                  */
    void *_val,  /* value of token to return (overloaded).
                  * C90 hack - declare formal as void *; cast later
                  */
    int *len,    /* # of chars in number */
    bool dc_flag)
{
  char c;
  char *cp;
  char *fcptr;
  int ret_err;
  char *ep;
  char *bp;
  char decimal_char = '.';
  int itmp;
  union {
    __BIGINT_T i;
    __BIGREAL_T d;
    DBLINT64 i8v;
  } * val; /* value of token to return */

  if (dc_flag == TRUE)
    decimal_char = ',';
  else
    decimal_char = '.';
  val = _val;
  ret_err = 0;
  c = *(cp = currc);
  if (c == '-' || c == '+')
    c = *++cp;
  if (c == decimal_char) {
    *cp = '.';
    c = *++cp;
    if (!ISDIGIT(c))
      goto junk0;
    goto state2;
  }
  if (!ISDIGIT(c))
    goto junk0;

  /*state1:			digits */
  do {
    c = *++cp;
  } while (ISDIGIT(c));
  if (c == decimal_char) {
    *cp = '.';
    goto state2;
  }
  if (c == 'e' || c == 'E' || c == 'd' || c == 'D' || c == 'q' || c == 'Q')
    goto state3;
  if (c == '+' || c == '-')
    goto state6;
  goto return_integer;
state2: /* . digit [digits] or digits . [ digits ] */
  do {
    c = *++cp;
  } while (ISDIGIT(c));
  if (c == 'e' || c == 'E' || c == 'd' || c == 'D' || c == 'q' || c == 'Q')
    goto state3;
  if (c == '+' || c == '-')
    goto state6;
  goto return_real;

state3: /* digits [ . [ digits ] ] { e | d | q} */
  if (c == 'd' || c == 'D' || c == 'q' || c == 'Q')
    *cp = 'e'; /* to ensure that strtod works */
  c = *++cp;
  if (ISDIGIT(c))
    goto state5;
  if (c == '+' || c == '-')
    goto state4;
  /*
   * VMS extension: no digits, +, or - after e or d
   */
  *(cp - 1) = ' ';
  goto return_real;

state4: /* digits [ . [ digits ] ] { e | d } { + | - } */
  c = *++cp;
  if (!ISDIGIT(c)) {
    /*
     * VMS extension: no digits after { e | d } { + | - }
     */
    *(cp - 1) = ' ';
    *(cp - 2) = ' ';
    goto return_real;
  }

state5: /* digits [ . [ digits ] ] { e | d } [ + | - ] digits */
  c = *++cp;
  if (ISDIGIT(c))
    goto state5;
  goto return_real;

state6: /* digits [ . [ digits ] ] { + | - } */
        /*
         * VMS extension - form real value where digits after +/- comprise the
         * exponent.  cp points to +/-.
         */
  ep = cp;
  do { /* scan past exponent */
    c = *++cp;
  } while (ISDIGIT(c));
  if ((cp - currc) + 2 > buf_size) {
    buf_size = (cp - currc) + 64;
    if (buf_p != buf)
      free(buf_p);
    buf_p = (char *)malloc(buf_size);
  }
  itmp = ep - currc;
  memcpy(buf_p, currc, itmp);
  bp = buf_p + itmp;
  itmp = cp - ep;
  if (itmp > 1) {
    *bp++ = 'e';
    memcpy(bp, ep, itmp);
    bp += itmp;
  }
  *bp = '\0';
  fcptr = NULL;
  val->d = __io_strtold(buf_p, &fcptr);
  if (fcptr == buf_p) {
    /* illegal real constant */
    ret_err = FIO_EERR_DATA_CONVERSION;
    goto ret;
  }
  *type = 1;
  goto ret;

junk0:
  /*
   * VMS extension: don't have digits after { + | - | . | +. | -. }
   * Could have exponent of the form +/- digits after one of the above.
   */
  if (c == '+' || c == '-') {
    c = *++cp;
    while (ISDIGIT(c))
      c = *++cp;
  } else if (c == 'e' || c == 'E' || c == 'd' || c == 'D' || c == 'q' ||
             c == 'Q') {
    c = *++cp;
    if (c == '+' || c == '-')
      c = *++cp;
    while (ISDIGIT(c))
      c = *++cp;
  }
  *type = 3;
  val->i = 0;
  goto ret;

return_integer:
  *type = 0;
  fcptr = NULL;
  ret_err = __fort_atoxi32(currc, &val->i, cp - currc, 10);
  if (ret_err) {
    ret_err = __fort_atoxi64(currc, val->i8v, cp - currc, 10);
    *type = 2;
  }
  if (ret_err)
    /* illegal integer constant */
    ret_err = FIO_EERR_DATA_CONVERSION;
  goto ret;

return_real:
  *type = 1;
  fcptr = NULL;
  val->d = __io_strtold(currc, &fcptr);
  if (fcptr == currc)
    /* illegal real constant */
    ret_err = FIO_EERR_DATA_CONVERSION;

ret:
  *len = cp - currc;
  if (DBGBIT(0x1)) {
    __io_printf("fio_getnum: type=%d, len=%d, ret_err=%d\n", *type, *len,
                 ret_err);
    __io_printf("   str:#%.*s#, val:", *len, currc);
    if (*type) {
      __fortio_printbigreal(val->d);
      __io_printf("\n");
    } else
      __io_printf("%d\n", val->i);
  }
  c = *cp;
  if ((c == '\n') || (c == '\0') || (c == ',') || (c == ' ') || (c == '/') ||
      (c == ';') || (c == '\t') || (c == '\r') || (c == ')') || (c == '*')) {
    return ret_err;
  }
  ret_err = FIO_EERR_DATA_CONVERSION;
  return ret_err;
}
