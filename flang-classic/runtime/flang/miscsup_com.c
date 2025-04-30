/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief
 * Collection of misc Fortran intrinsics (present, min, max, ajustl, adjustl, ...).
 *
 */

#include <time.h>
#include <string.h>
#if !defined(_WIN64)
#include <sys/time.h>
#include <unistd.h>
#else
#include <winsock.h> // timeval
#endif
#include "stdioInterf.h"
#include "fioMacros.h"
#include "llcrit.h"
#include "global.h"
#include "memops.h"
#include "math.h"

MP_SEMAPHORE(static, sem);
#include "type.h"

extern double __fort_second();

#define time(x) __fort_time(x)

typedef __INT8_T MXINT_T;
static void store_mxint_t(void *, F90_Desc *, MXINT_T);
static MXINT_T mxint(F90_Desc *);

__INT_T
ENTFTN(ILEN, ilen)(void *ib, __INT_T *size)
{
  /*
   * if i is nonnegative,
   *     ilen(i) = ceiling(log2(i+1))
   * if i is negative,
   *     ilen(i) = ceiling(log2(-i))
   */
  unsigned ui;
  int i, k, ln;

  i = I8(__fort_varying_int)(ib, size);
  if (i < 0)
    i = -i;
  else
    ++i;

  /* find bit position (relative to 0) of the leftmost 1 bit */

  ui = i;
  ln = -1;
  k = (*size * 8) >> 1;
  while (k) {
    if (ui >> k) {
      ui >>= k;
      ln += k;
    }
    k >>= 1;
  }
  if (ui)
    ++ln;

  /* if i is larger than 2**(bit pos), increase by one */

  if (i ^ (1 << ln))
    ++ln;
  return ln;
}

__LOG_T
ENTF90(PRESENT, present)(void *p)
{
  if (p == NULL) {
    return 0;
  }

#if defined(DESC_I8)
  if (!((__INT4_T *)(p) >= ENTCOMN(0, 0) &&
        (__INT4_T *)(p) <= (ENTCOMN(0, 0) + 3)))
#else
  if (!((__INT_T *)(p) >= ENTCOMN(0, 0) &&
        (__INT_T *)(p) <= (ENTCOMN(0, 0) + 3)))
#endif
    return GET_DIST_TRUE_LOG;
  else
    return 0;
}

__LOG_T
ENTF90(PRESENT_PTR, present_ptr)(void *p)
{
  if (p == NULL) {
    return 0;
  }

#if defined(DESC_I8)
  if (!((__INT4_T *)(p) >= ENTCOMN(0, 0) &&
        (__INT4_T *)(p) <= (ENTCOMN(0, 0) + 3)) &&
      !(*(__INT4_T **)(p) >= ENTCOMN(0, 0) &&
        *(__INT4_T **)(p) <= (ENTCOMN(0, 0) + 3)))

#else
  if (!((__INT_T *)(p) >= ENTCOMN(0, 0) &&
        (__INT_T *)(p) <= (ENTCOMN(0, 0) + 3)) &&
      !(*(__INT_T **)(p) >= ENTCOMN(0, 0) &&
        *(__INT_T **)(p) <= (ENTCOMN(0, 0) + 3)))
#endif
    return GET_DIST_TRUE_LOG;
  else
    return 0;
}

__LOG_T
ENTF90(PRESENTCA, presentca)(DCHAR(p) DCLEN64(p))
{
  if (CADR(p) == NULL) {
    return 0;
  }

  if (CADR(p) != ABSENTC)
    return GET_DIST_TRUE_LOG;
  else
    return 0;
}

/* 32 bit CLEN version */
__LOG_T
ENTF90(PRESENTC, presentc)(DCHAR(p) DCLEN(p))
{
  return ENTF90(PRESENTCA, presentca)(CADR(p), (__CLEN_T)CLEN(p));
}

/** \brief
 * -i8 variant of present
 */
__LOG8_T
ENTF90(KPRESENT, kpresent)(void *p)
{


  return (__INT8_T)ISPRESENT(p) ? GET_DIST_TRUE_LOG : 0;
}

__LOG8_T
ENTF90(KPRESENT_PTR, kpresent_ptr)(void *p)
{

  /* 
   * -i8 variant of present
   */

  if (p == NULL) {
    return 0;
  }

#if defined(DESC_I8)
  if (!((__INT4_T *)(p) >= ENTCOMN(0, 0) &&
        (__INT4_T *)(p) <= (ENTCOMN(0, 0) + 3)) &&
      !(*(__INT4_T **)(p) >= ENTCOMN(0, 0) &&
        *(__INT4_T **)(p) <= (ENTCOMN(0, 0) + 3)))

#else
  if (!((__INT_T *)(p) >= ENTCOMN(0, 0) &&
        (__INT_T *)(p) <= (ENTCOMN(0, 0) + 3)) &&
      !(*(__INT_T **)(p) >= ENTCOMN(0, 0) &&
        *(__INT_T **)(p) <= (ENTCOMN(0, 0) + 3)))
#endif
    return GET_DIST_TRUE_LOG;
  else
    return 0;
}

__LOG8_T
ENTF90(KPRESENTCA, kpresentca)(DCHAR(p) DCLEN64(p))
{

  /* 
   * -i8 variant of PRESENTC
   */

  return (__INT8_T)ISPRESENTC(p) ? GET_DIST_TRUE_LOG : 0;
}

/* 32 bit CLEN version */
__LOG8_T
ENTF90(KPRESENTC, kpresentc)(DCHAR(p) DCLEN(p))
{
  return ENTF90(KPRESENTCA, kpresentca)(CADR(p), (__CLEN_T)CLEN(p));
}

__LOG_T
ENTF90(IS_IOSTAT_END, is_iostat_end)(__INT4_T i)
{

  return (i == -1) ? GET_DIST_TRUE_LOG : 0;
}

__LOG8_T
ENTF90(KIS_IOSTAT_END, kis_iostat_end)(__INT4_T i)
{

  return (i == -1) ? GET_DIST_TRUE_LOG : 0;
}

__LOG_T
ENTF90(IS_IOSTAT_EOR, is_iostat_eor)(__INT4_T i)
{

  return (i == -2) ? GET_DIST_TRUE_LOG : 0;
}

__LOG8_T
ENTF90(KIS_IOSTAT_EOR, kis_iostat_eor)(__INT4_T i)
{

  return (i == -2) ? GET_DIST_TRUE_LOG : 0;
}

void
*ENTF90(LOC, loc)(void *p)
{
  return p;
}

__INT_T
ENTF90(IMAX, imax)(__INT_T i, __INT_T j)
{
  return (i > j) ? i : j;
}

#if !defined(DESC_I8)
void
ENTF90(MIN, min)(int *nargs, ...)
{
  char *nextstr;
  char *minstr;
  char *result;
  int i, j;
  __CLEN_T clen;
  va_list argp;

  va_start(argp, nargs);
  j = *nargs;

  /* First loop through the argument list to get character len */
  result = va_arg(argp, char *);
  minstr = va_arg(argp, char *);
  if (result == NULL)
    return;

  /* argument list */
  for (i = 0; i < j; ++i) {
    nextstr = va_arg(argp, char *);
  }
  clen = va_arg(argp, __CLEN_T);
  va_end(argp);

  /* start real comparison */
  va_start(argp, nargs);
  result = va_arg(argp, char *);
  minstr = va_arg(argp, char *);
  if (minstr == NULL)
    return;
  for (i = 0; i < j - 1; ++i) {
    nextstr = va_arg(argp, char *);
    if (nextstr) {
      if (strncmp(nextstr, minstr, clen) < 0)
        minstr = nextstr;
    }
  }
  strncpy(result, minstr, clen);
  va_end(argp);
}

void
ENTF90(MAX, max)(int *nargs, ...)
{
  char *nextstr;
  char *maxstr;
  char *result;
  int i, j;
  __CLEN_T clen;
  va_list argp;

  va_start(argp, nargs);
  j = *nargs;

  /* First loop through the argument list to get character len */
  result = va_arg(argp, char *);
  maxstr = va_arg(argp, char *);
  if (result == NULL)
    return;

  /* argument list */
  for (i = 0; i < j; ++i) {
    nextstr = va_arg(argp, char *);
  }
  clen = va_arg(argp, __CLEN_T);
  va_end(argp);

  /* start real comparison */
  va_start(argp, nargs);
  result = va_arg(argp, char *);
  maxstr = va_arg(argp, char *);
  if (maxstr == NULL)
    return;
  for (i = 0; i < j - 1; ++i) {
    nextstr = va_arg(argp, char *);
    if (nextstr) {
      if (strncmp(nextstr, maxstr, clen) > 0)
        maxstr = nextstr;
    }
  }
  strncpy(result, maxstr, clen);
  va_end(argp);
}

#endif

__INT8_T
ENTF90(KICHARA, kichara)
(DCHAR(c) DCLEN64(c))
{
  return (__INT8_T)(CADR(c)[0] & 0xff);
}
/* 32 bit CLEN version */
__INT8_T
ENTF90(KICHAR, kichar)
(DCHAR(c) DCLEN(c))
{
  return ENTF90(KICHARA, kichara)(CADR(c), (__CLEN_T)CLEN(c));
}

__INT_T
ENTF90(LENA, lena)(DCHAR(s) DCLEN64(s))
{
  return (__INT_T)CLEN(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90(LEN, len)(DCHAR(s) DCLEN(s))
{
  return (__INT_T) ENTF90(LENA, lena)(CADR(s), (__CLEN_T)CLEN(s));
}

__INT8_T
ENTF90(KLENA, klena)(DCHAR(s) DCLEN64(s))
{
  return (__INT8_T)CLEN(s);
}
/* 32 bit CLEN version */
__INT8_T
ENTF90(KLEN, klen)(DCHAR(s) DCLEN(s))
{
  return ENTF90(KLENA, klena)(CADR(s), (__CLEN_T)CLEN(s));
}

__INT_T
ENTF90(NLENA, nlena)(DCHAR(s) DCLEN64(s))
{
  return (__INT_T)CLEN(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90(NLEN, nlen)(DCHAR(s) DCLEN(s))
{
  return ENTF90(NLENA, nlena)(CADR(s), (__CLEN_T)CLEN(s));
}

__CLEN_T
ENTF90(ADJUSTLA, adjustla)
(DCHAR(res), DCHAR(expr) DCLEN64(res) DCLEN64(expr))
{
  __CLEN_T i, j, elen, rlen;

  elen = CLEN(expr);
  rlen = CLEN(res);
  for (i = 0; i < elen && CADR(expr)[i] == ' '; ++i)
    ;
  for (j = 0; i < elen; ++i, ++j)
    CADR(res)[j] = CADR(expr)[i];
  for (; j < rlen; ++j)
    CADR(res)[j] = ' ';
  return elen;
}
/* 32 bit CLEN version */
__INT_T
ENTF90(ADJUSTL, adjustl)
(DCHAR(res), DCHAR(expr) DCLEN(res) DCLEN(expr))
{
  return (__INT_T)ENTF90(ADJUSTLA, adjustla)(CADR(res), CADR(expr),
                                     (__CLEN_T)CLEN(res), (__CLEN_T)CLEN(expr));
}

__CLEN_T
ENTF90(ADJUSTRA, adjustra)
(DCHAR(res), DCHAR(expr) DCLEN64(res) DCLEN64(expr))
{
  __CLEN_T i, j, len;

  len = CLEN(expr);
  for (i = len; i-- > 0 && CADR(expr)[i] == ' ';)
    ;
  for (++i, j = len-1; i-- > 0; --j)
    CADR(res)[j] = CADR(expr)[i];
  for (++j; j-- > 0; )
    CADR(res)[j] = ' ';
  return len;
}
/* 32 bit CLEN version */
__INT_T
ENTF90(ADJUSTR, adjustr)
(DCHAR(res), DCHAR(expr) DCLEN(res) DCLEN(expr))
{
  return (__INT_T)ENTF90(ADJUSTRA, adjustra)(CADR(res), CADR(expr),
                                     (__CLEN_T)CLEN(res), (__CLEN_T)CLEN(expr));
}

static void
fstrcpy(char *s1, char *s2, __CLEN_T len1, __CLEN_T len2)
{
  __CLEN_T i;

  if (len2 < len1) {
    for (i = 0; i < len2; ++i)
      s1[i] = s2[i];
    for (; i < len1; ++i)
      s1[i] = ' ';
  } else {
    for (i = 0; i < len1; ++i)
      s1[i] = s2[i];
  }
}

static const char *month[12] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};

static int
yr2(int yr)
{
  int y = yr;
  if (y > 99)
    y = y % 100;
  return y;
}

void
ENTFTN(DATEA, datea)(DCHAR(date), F90_Desc *dated DCLEN64(date))
{
  char loc_buf[16];
  time_t ltime;
  struct tm *lt;

  ltime = time();
  MP_P(sem);
  ;
  lt = localtime(&ltime);
  sprintf(loc_buf, "%2d-%3s-%02d", lt->tm_mday, month[lt->tm_mon],
          yr2(lt->tm_year));
  MP_V(sem);
  fstrcpy(CADR(date), loc_buf, CLEN(date), 9);
}
/* 32 bit CLEN version */
void
ENTFTN(DATE, date)(DCHAR(date), F90_Desc *dated DCLEN(date))
{
  ENTFTN(DATEA, datea)(CADR(date), dated, (__CLEN_T)CLEN(date));
}

void
ENTFTN(DATEW, datew)(void *date, F90_Desc *dated)
{
  char loc_buf[16];
  time_t ltime;
  struct tm *lt;

  ltime = time();
  MP_P(sem);
  ;
  lt = localtime(&ltime);
  sprintf(loc_buf, "%2d-%3s-%02d", lt->tm_mday, month[lt->tm_mon],
          yr2(lt->tm_year));
  MP_V(sem);
  fstrcpy(date, loc_buf, 9, 9);
}

void
ENTFTN(JDATE, jdate)(__INT4_T *i, __INT4_T *j, __INT4_T *k, F90_Desc *id,
                     F90_Desc *jd, F90_Desc *kd)
{
  time_t ltime;
  struct tm *ltimvar;

  ltime = time();
  MP_P(sem);
  ;
  ltimvar = localtime(&ltime);
  *i = ltimvar->tm_mon + 1;
  *j = ltimvar->tm_mday;
  *k = yr2(ltimvar->tm_year);
  MP_V(sem);
}

void
ENTFTN(IDATE, idate)(__INT2_T *i, __INT2_T *j, __INT2_T *k, F90_Desc *id,
                     F90_Desc *jd, F90_Desc *kd)
{
  time_t ltime;
  struct tm *ltimvar;

  ltime = time();
  MP_P(sem);
  ;
  ltimvar = localtime(&ltime);
  *i = ltimvar->tm_mon + 1;
  *j = ltimvar->tm_mday;
  *k = yr2(ltimvar->tm_year);
  MP_V(sem);
}

/* trying to deal with loss of significant digits in
   real*4 version.
 */
#define TIME_THRESHOLD2 1.033944E+09
#define TIME_THRESHOLD1 1.003944E+09

void
ENTFTN(CPU_TIME, cpu_time)(__REAL4_T *x)
{
  extern double __fort_second();
  double secs;
  __REAL4_T res;

  secs = __fort_second();
  if (secs > TIME_THRESHOLD2)
    res = secs - TIME_THRESHOLD2;
  else if (secs > TIME_THRESHOLD1)
    res = secs - TIME_THRESHOLD1;
  else
    res = secs;
  *x = res;
}

void
ENTFTN(CPU_TIMED, cpu_timed)(__REAL8_T *x)
{
  extern double __fort_second();
  double secs;
  __REAL8_T res;

  secs = __fort_second();
  /* probably not necessary for this version, except that
     user could mix real*4 and real*8 versions.
   */
  if (secs > TIME_THRESHOLD2)
    res = secs - TIME_THRESHOLD2;
  else if (secs > TIME_THRESHOLD1)
    res = secs - TIME_THRESHOLD1;
  else
    res = secs;
  *x = res;
}

#ifdef TARGET_SUPPORTS_QUADFP
void
ENTFTN(CPU_TIMEQ, cpu_timeq)(__REAL16_T *x)
{
  extern double __fort_second();
  float128_t secs;
  __REAL16_T res;

  secs = (float128_t)__fort_second();
  if (secs > TIME_THRESHOLD2)
    res = secs - TIME_THRESHOLD2;
  else if (secs > TIME_THRESHOLD1)
    res = secs - TIME_THRESHOLD1;
  else
    res = secs;
  *x = res;
}
#endif

__REAL4_T
ENTFTN(SECNDS, secnds)(__REAL4_T *x, F90_Desc *xd)
{
  static int called = 0;
  static int diffs;
  int i;
  time_t ltime;
  struct tm *lt;
  __REAL4_T f;

  ltime = time();
  if (called == 0) {
    called = 1; /* first time called */
                /*
                 * compute value to subtract from time(0) to give seconds since
                 * midnight
                 */
    MP_P(sem);
    ;
    lt = localtime(&ltime);
    i = lt->tm_sec + (60 * lt->tm_min) + (3600 * lt->tm_hour);
    MP_V(sem);
    diffs = ltime - i;
  }
  f = (__REAL4_T)(ltime - diffs);
  return (f - *x);
}

__REAL8_T
ENTFTN(SECNDSD, secndsd)(__REAL8_T *x, F90_Desc *xd)
{
  static int called = 0;
  static int diffs;
  int i;
  time_t ltime;
  struct tm *lt;
  __REAL8_T f;

  ltime = time();
  if (called == 0) {
    called = 1; /* first time called */
                /*
                 * compute value to subtract from time() to give seconds since
                 * midnight
                 */
    MP_P(sem);
    ;
    lt = localtime(&ltime);
    i = lt->tm_sec + (60 * lt->tm_min) + (3600 * lt->tm_hour);
    MP_V(sem);
    diffs = ltime - i;
  }
  f = (__REAL8_T)(ltime - diffs);
  return (f - *x);
}

void
ENTFTN(FTIMEA, ftimea)(DCHAR(tbuf), F90_Desc *tbufd DCLEN64(tbuf))
{
  char loc_buf[16];
  time_t ltime;
  struct tm *ltimvar;

  ltime = time();
  MP_P(sem);
  ;
  ltimvar = localtime(&ltime);
  sprintf(loc_buf, "%2.2d:%2.2d:%2.2d", ltimvar->tm_hour, ltimvar->tm_min,
          ltimvar->tm_sec);
  MP_V(sem);
  fstrcpy(CADR(tbuf), loc_buf, CLEN(tbuf), 8);
}
/* 32 bit CLEN version */
void
ENTFTN(FTIME, ftime)(DCHAR(tbuf), F90_Desc *tbufd DCLEN(tbuf))
{
  ENTFTN(FTIMEA, ftimea)(CADR(tbuf), tbufd, (__CLEN_T)CLEN(tbuf));
}

void
ENTFTN(FTIMEW, ftimew)(void *tbuf, F90_Desc *tbufd)
{
  char loc_buf[16];
  time_t ltime;
  struct tm *ltimvar;

  ltime = time();
  MP_P(sem);
  ;
  ltimvar = localtime(&ltime);
  sprintf(loc_buf, "%2.2d:%2.2d:%2.2d", ltimvar->tm_hour, ltimvar->tm_min,
          ltimvar->tm_sec);
  MP_V(sem);
  fstrcpy(tbuf, loc_buf, 8, 8);
}

static int
I8(next_index)(__INT_T *index, F90_Desc *s)
{
  __INT_T i;

  for (i = 0; i < F90_RANK_G(s); i++) {
    index[i]++;
    if (index[i] <= DIM_UBOUND_G(s, i)) {
      return 1; /* keep going */
    }
    index[i] = F90_DIM_LBOUND_G(s, i);
  }
  return 0; /* finished */
}

void
ENTFTN(DANDTA, dandta)(DCHAR(date), DCHAR(tbuf), DCHAR(zone),
                     __STAT_T *values, F90_Desc *dated, F90_Desc *tbufd,
                     F90_Desc *zoned,
                     F90_Desc *valuesd DCLEN64(date) DCLEN64(tbuf) DCLEN64(zone))
{
  int tvalues[8];
  int i;
  char c;
  time_t ltime;
  struct tm *tm, tmx;
  char loc_buf[16];
  int ms;
#if defined(TARGET_OSX)
  struct timeval t;
  struct timezone tz0;
#else
  struct timeval t;
#endif

#if defined(TARGET_OSX)
  gettimeofday(&t, &tz0);
  ltime = t.tv_sec;
  ms = t.tv_usec / 1000;
#else
  gettimeofday(&t, (void *)0);
  ltime = t.tv_sec;
  ms = t.tv_usec / 1000;
#endif
  MP_P(sem);
  ;
  tm = localtime(&ltime);
  if (tm == NULL) {
    fprintf(__io_stderr(), "BAD return value from localtime(0x%lx)\n",
            (long)ltime);
    perror("localtime: ");
    exit(1);
  }
  memcpy(&tmx, tm, sizeof(struct tm));
  tm = &tmx;
  MP_V(sem);
  if (ISPRESENTC(date) && CLEN(date) > 0) {
    sprintf(loc_buf, "%04d%02d%02d", tm->tm_year + 1900, tm->tm_mon + 1,
            tm->tm_mday);
    fstrcpy(CADR(date), loc_buf, CLEN(date), 8);
  }
  if (ISPRESENTC(tbuf) && CLEN(tbuf) > 0) {
    sprintf(loc_buf, "%02d%02d%02d.%03d", tm->tm_hour, tm->tm_min, tm->tm_sec,
            ms);
    fstrcpy(CADR(tbuf), loc_buf, CLEN(tbuf), 10);
  }
  if (ISPRESENTC(zone) && CLEN(zone) > 0) {
    i = __io_timezone(tm);
    c = '+';
    if (i < 0) {
      i = -i;
      c = '-';
    }
    i /= 60;
    sprintf(loc_buf, "%c%02d%02d", c, i / 60, i % 60);
    fstrcpy(CADR(zone), loc_buf, CLEN(zone), 5);
  }
  if (ISPRESENT(values)) {
    tvalues[0] = tm->tm_year + 1900;
    tvalues[1] = tm->tm_mon + 1;
    tvalues[2] = tm->tm_mday;
    i = __io_timezone(tm);
    c = '+';
    if (i < 0) {
      i = -i;
      c = '-';
    }
    i /= 60;
    if (c == '-')
      i = -i;
    tvalues[3] = i;
    tvalues[4] = tm->tm_hour;
    tvalues[5] = tm->tm_min;
    tvalues[6] = tm->tm_sec;
    tvalues[7] = ms;
    if (valuesd && F90_TAG_G(valuesd) == __DESC) {
      char *la;
      __INT_T index[7];

      for (i = 0; i < F90_RANK_G(valuesd); ++i) {
        if (DIM_UBOUND_G(valuesd, i) < F90_DIM_LBOUND_G(valuesd, i))
          return;
        index[i] = F90_DIM_LBOUND_G(valuesd, i);
      }
      for (i = 0; i < 8; ++i) {
        la = I8(__fort_local_address)(values, valuesd, index);
        if (la) {
          /*  *((int *)la) = tvalues[i];  */
          store_mxint_t(la, valuesd, tvalues[i]);
        }
        if (I8(next_index)(index, valuesd) == 0)
          break;
      }
    } else {
      for (i = 0; i < 8; ++i)
        values[i] = tvalues[i];
    }
  }
}
/* 32 bit CLEN version */
void
ENTFTN(DANDT, dandt)(DCHAR(date), DCHAR(tbuf), DCHAR(zone),
                     __STAT_T *values, F90_Desc *dated, F90_Desc *tbufd,
                     F90_Desc *zoned,
                     F90_Desc *valuesd DCLEN(date) DCLEN(tbuf) DCLEN(zone))
{
  ENTFTN(DANDTA, dandta)(CADR(date), CADR(tbuf), CADR(zone), values, dated,
                         tbufd, zoned, valuesd, (__CLEN_T)CLEN(date),
                         (__CLEN_T)CLEN(tbuf), (__CLEN_T)CLEN(zone));
}

void
ENTFTN(SYSCLK, sysclk)(__STAT_T *count, __STAT_T *count_rate,
                       __STAT_T *count_max, F90_Desc *countd,
                       F90_Desc *count_rated, F90_Desc *count_maxd)
{
  static MXINT_T resol; /* resolution, tics per second */
  int sz;

  if (resol == 0) {
    int def;
#if defined(TARGET_X8664) || defined(TARGET_LLVM_ARM64)
    def = 1000000;
#else
    def = sizeof(__STAT_T) < 8 ? 1000 : 1000000;
#endif
    resol = __fort_getoptn("-system_clock_rate", def);
    if (resol <= 0)
      __fort_abort("invalid value given for system_clock rate");
  }
  if (ISPRESENT(count_rate)) {
    if (ISPRESENT(count)) {
      sz = GET_DIST_SIZE_OF(TYPEKIND(countd));
    } else {
      sz = GET_DIST_SIZE_OF(TYPEKIND(count_rated));
    }
    switch (sz) {
    case 1:
      resol = 10;
      break;
    case 2:
      resol = 1000;
      break;
    case 4:
      resol = 1000000;
      break;
    default: /* big*/
      resol = 10000000;
      break;
    }
  }
  if (ISPRESENT(count)) {
    double t = __fort_second();
    MXINT_T mxt;
    mxt = mxint(countd);
    if (t * resol > mxt) {
      t = 0;
      __fort_set_second(t);
    }
    store_mxint_t(count, countd, (t)*resol);
  }
  if (ISPRESENT(count_rate)) {
    store_mxint_t(count_rate, count_rated, resol);
  }
  if (ISPRESENT(count_max)) {
    if (ISPRESENT(count)) {
      store_mxint_t(count_max, count_maxd, mxint(countd));
    } else {
      store_mxint_t(count_max, count_maxd, mxint(count_maxd));
    }
  }
}

void
ENTF90(MVBITS, mvbits)(void *from, void *frompos, void *len, void *to,
                       void *topos, __INT_T *szfrom, __INT_T *szfrompos,
                       __INT_T *szlen, __INT_T *sztopos)
{
  __INT1_T f1, t1, m1;
  __INT2_T f2, t2, m2;
  __INT4_T f4, t4, m4;
  __INT8_T f8, t8, m8;
  int fp = I8(__fort_varying_int)(frompos, szfrompos);
  int ln = I8(__fort_varying_int)(len, szlen);
  int tp = I8(__fort_varying_int)(topos, sztopos);

#undef MVBIT_OVFL
#define MVBIT_OVFL(w) ((fp + ln) > (w) || (tp + ln) > (w))

  if (fp < 0 || tp < 0 || ln <= 0)
    return;
  switch (*szfrom) {
  case 1:
    if (MVBIT_OVFL(8))
      break;
    if (ln == 8) {
      *(__INT1_T *)to = *(__INT1_T *)from;
      break;
    }
    f1 = *(__INT1_T *)from;
    t1 = *(__INT1_T *)to;
    m1 = (~(-1 << ln)) << tp;
    *(__INT1_T *)to = (t1 & ~m1) | (((f1 >> fp) << tp) & m1);
    break;
  case 2:
    if (MVBIT_OVFL(16))
      break;
    if (ln == 16) {
      *(__INT2_T *)to = *(__INT2_T *)from;
      break;
    }
    f2 = *(__INT2_T *)from;
    t2 = *(__INT2_T *)to;
    m2 = (~(-1 << ln)) << tp;
    *(__INT2_T *)to = (t2 & ~m2) | (((f2 >> fp) << tp) & m2);
    break;
  case 4:
    if (MVBIT_OVFL(32))
      break;
    if (ln == 32) {
      *(__INT4_T *)to = *(__INT4_T *)from;
      break;
    }
    f4 = *(__INT4_T *)from;
    t4 = *(__INT4_T *)to;
    m4 = (~(-1 << ln)) << tp;
    *(__INT4_T *)to = (t4 & ~m4) | (((f4 >> fp) << tp) & m4);
    break;
  case 8:
    if (MVBIT_OVFL(64))
      break;
    if (ln == 64) {
      *(__INT8_T *)to = *(__INT8_T *)from;
      break;
    }
    f8 = *(__INT8_T *)from;
    t8 = *(__INT8_T *)to;
    m8 = (~((__INT8_T)-1 << ln)) << tp;
    *(__INT8_T *)to = (t8 & ~m8) | (((f8 >> fp) << tp) & m8);
    break;
  default:
    __fort_abort("MVBITS: unsupported from/to integer size");
  }
}

/** \brief
 *Varargs:  __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<rank>, __INT_T *ub<rank>
 */
__INT_T
ENTF90(LB, lb)(__INT4_T *rank, __INT4_T *dim, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T d;

  d = *dim;
  if (d < 1 || d > *rank)
    __fort_abort("LBOUND: invalid dim");
  va_start(va, dim);
  while (d-- > 0) {
    lb = va_arg(va, __INT_T *);
    ub = va_arg(va, __INT_T *);
  }
  va_end(va);
  if (!ISPRESENT(lb))
    __fort_abort("LBOUND: lower bound not present for specified dim");
  if (!ISPRESENT(ub))
    /* presumably, it's the last dimension of an assumed array */
    return *lb;
  return (*lb <= *ub) ? *lb : 1;
}

/* Varargs:
 *  __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<rank>, __INT_T *ub<rank>
 */
__INT1_T
ENTF90(LB1, lb1)(__INT4_T *rank, __INT4_T *dim, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T d;

  d = *dim;
  if (d < 1 || d > *rank)
    __fort_abort("LBOUND: invalid dim");
  va_start(va, dim);
  while (d-- > 0) {
    lb = va_arg(va, __INT_T *);
    ub = va_arg(va, __INT_T *);
  }
  va_end(va);
  if (!ISPRESENT(lb))
    __fort_abort("LBOUND: lower bound not present for specified dim");
  if (!ISPRESENT(ub))
    /* presumably, it's the last dimension of an assumed array */
    return (__INT1_T)*lb;
  return (__INT1_T)(*lb <= *ub) ? *lb : 1;
}

/* Varargs:
 *  __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<rank>, __INT_T *ub<rank>
 */
__INT2_T
ENTF90(LB2, lb2)(__INT4_T *rank, __INT4_T *dim, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T d;

  d = *dim;
  if (d < 1 || d > *rank)
    __fort_abort("LBOUND: invalid dim");
  va_start(va, dim);
  while (d-- > 0) {
    lb = va_arg(va, __INT_T *);
    ub = va_arg(va, __INT_T *);
  }
  va_end(va);
  if (!ISPRESENT(lb))
    __fort_abort("LBOUND: lower bound not present for specified dim");
  if (!ISPRESENT(ub))
    /* presumably, it's the last dimension of an assumed array */
    return (__INT2_T)*lb;
  return (__INT2_T)(*lb <= *ub) ? *lb : 1;
}

/* Varargs:  __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<rank>, __INT_T *ub<rank> */
__INT4_T
ENTF90(LB4, lb4)(__INT4_T *rank, __INT4_T *dim, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T d;

  d = *dim;
  if (d < 1 || d > *rank)
    __fort_abort("LBOUND: invalid dim");
  va_start(va, dim);
  while (d-- > 0) {
    lb = va_arg(va, __INT_T *);
    ub = va_arg(va, __INT_T *);
  }
  va_end(va);
  if (!ISPRESENT(lb))
    __fort_abort("LBOUND: lower bound not present for specified dim");
  if (!ISPRESENT(ub))
    /* presumably, it's the last dimension of an assumed array */
    return (__INT4_T)*lb;
  return (__INT4_T)(*lb <= *ub) ? *lb : 1;
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<rank>, __INT_T *ub<rank> */
__INT8_T
ENTF90(LB8, lb8)(__INT4_T *rank, __INT4_T *dim, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T d;

  d = *dim;
  if (d < 1 || d > *rank)
    __fort_abort("LBOUND: invalid dim");
  va_start(va, dim);
  while (d-- > 0) {
    lb = va_arg(va, __INT_T *);
    ub = va_arg(va, __INT_T *);
  }
  va_end(va);
  if (!ISPRESENT(lb))
    __fort_abort("LBOUND: lower bound not present for specified dim");
  if (!ISPRESENT(ub))
    /* presumably, it's the last dimension of an assumed array */
    return (__INT8_T)*lb;
  return (__INT8_T)(*lb <= *ub) ? *lb : 1;
}

/* Varargs:  __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<rank>, __INT_T *ub<rank> */
__INT8_T
ENTF90(KLB, klb)(__INT4_T *rank, __INT4_T *dim, ...)
{

  /* 
   * -i8 variant of LB
   */

  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T d;

  d = *dim;
  if (d < 1 || d > *rank)
    __fort_abort("LBOUND: invalid dim");
  va_start(va, dim);
  while (d-- > 0) {
    lb = va_arg(va, __INT_T *);
    ub = va_arg(va, __INT_T *);
  }
  va_end(va);
  if (!ISPRESENT(lb))
    __fort_abort("LBOUND: lower bound not present for specified dim");
  if (!ISPRESENT(ub))
    /* presumably, it's the last dimension of an assumed array */
    return *lb;
  return (__INT8_T)(*lb <= *ub) ? *lb : 1;
}

/* Varargs:  __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<rank>, __INT_T *ub<rank> */
__INT_T
ENTF90(UB, ub)(__INT4_T *rank, __INT4_T *dim, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T d;

  d = *dim;
  if (d < 1 || d > *rank)
    __fort_abort("UBOUND: invalid dim");
  va_start(va, dim);
  while (d-- > 0) {
    lb = va_arg(va, __INT_T *);
    ub = va_arg(va, __INT_T *);
  }
  va_end(va);
  if (!ISPRESENT(ub))
    __fort_abort("UBOUND: upper bound not present for specified dim");
  return (*lb <= *ub) ? *ub : 0;
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<rank>, __INT_T *ub<rank> */
__INT1_T
ENTF90(UB1, ub1)(__INT4_T *rank, __INT4_T *dim, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T d;

  d = *dim;
  if (d < 1 || d > *rank)
    __fort_abort("UBOUND: invalid dim");
  va_start(va, dim);
  while (d-- > 0) {
    lb = va_arg(va, __INT_T *);
    ub = va_arg(va, __INT_T *);
  }
  va_end(va);
  if (!ISPRESENT(ub))
    __fort_abort("UBOUND: upper bound not present for specified dim");
  return (__INT1_T)(*lb <= *ub) ? *ub : 0;
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<rank>, __INT_T *ub<rank> */
__INT2_T
ENTF90(UB2, ub2)(__INT4_T *rank, __INT4_T *dim, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T d;

  d = *dim;
  if (d < 1 || d > *rank)
    __fort_abort("UBOUND: invalid dim");
  va_start(va, dim);
  while (d-- > 0) {
    lb = va_arg(va, __INT_T *);
    ub = va_arg(va, __INT_T *);
  }
  va_end(va);
  if (!ISPRESENT(ub))
    __fort_abort("UBOUND: upper bound not present for specified dim");
  return (__INT2_T)(*lb <= *ub) ? *ub : 0;
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<rank>, __INT_T *ub<rank> */
__INT4_T
ENTF90(UB4, ub4)(__INT4_T *rank, __INT4_T *dim, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T d;

  d = *dim;
  if (d < 1 || d > *rank)
    __fort_abort("UBOUND: invalid dim");
  va_start(va, dim);
  while (d-- > 0) {
    lb = va_arg(va, __INT_T *);
    ub = va_arg(va, __INT_T *);
  }
  va_end(va);
  if (!ISPRESENT(ub))
    __fort_abort("UBOUND: upper bound not present for specified dim");
  return (__INT4_T)(*lb <= *ub) ? *ub : 0;
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<rank>, __INT_T *ub<rank> */
__INT8_T
ENTF90(UB8, ub8)(__INT4_T *rank, __INT4_T *dim, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T d;

  d = *dim;
  if (d < 1 || d > *rank)
    __fort_abort("UBOUND: invalid dim");
  va_start(va, dim);
  while (d-- > 0) {
    lb = va_arg(va, __INT_T *);
    ub = va_arg(va, __INT_T *);
  }
  va_end(va);
  if (!ISPRESENT(ub))
    __fort_abort("UBOUND: upper bound not present for specified dim");
  return (__INT8_T)(*lb <= *ub) ? *ub : 0;
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<rank>, __INT_T *ub<rank> */
__INT8_T
ENTF90(KUB, kub)(__INT4_T *rank, __INT4_T *dim, ...)
{

  /* 
   * -i8 variant of UB
   */

  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T d;

  d = *dim;
  if (d < 1 || d > *rank)
    __fort_abort("UBOUND: invalid dim");
  va_start(va, dim);
  while (d-- > 0) {
    lb = va_arg(va, __INT_T *);
    ub = va_arg(va, __INT_T *);
  }
  va_end(va);
  if (!ISPRESENT(ub))
    __fort_abort("UBOUND: upper bound not present for specified dim");
  return (__INT8_T)(*lb <= *ub) ? *ub : 0;
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<size>, __INT_T *ub<size> */
void
ENTF90(LBA, lba)(__INT_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    lb = va_arg(va, __INT_T *);
    if (!ISPRESENT(lb))
      __fort_abort("LBOUND: lower bound not present");
    ub = va_arg(va, __INT_T *);
    if (!ISPRESENT(ub))
      /* presumably, it's the last dimension of an assumed array */
      *arr++ = *lb;
    else
      *arr++ = (*lb <= *ub) ? *lb : 1;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<size>, __INT_T *ub<size> */
void
ENTF90(LBA1, lba1)(__INT1_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    lb = va_arg(va, __INT_T *);
    if (!ISPRESENT(lb))
      __fort_abort("LBOUND: lower bound not present");
    ub = va_arg(va, __INT_T *);
    if (!ISPRESENT(ub))
      /* presumably, it's the last dimension of an assumed array */
      *arr++ = *lb;
    else
      *arr++ = (*lb <= *ub) ? *lb : 1;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<size>, __INT_T *ub<size> */
void
ENTF90(LBA2, lba2)(__INT2_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    lb = va_arg(va, __INT_T *);
    if (!ISPRESENT(lb))
      __fort_abort("LBOUND: lower bound not present");
    ub = va_arg(va, __INT_T *);
    if (!ISPRESENT(ub))
      /* presumably, it's the last dimension of an assumed array */
      *arr++ = *lb;
    else
      *arr++ = (*lb <= *ub) ? *lb : 1;
  }
  va_end(va);
}

/* Varargs:  __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<size>, __INT_T *ub<size> */
void
ENTF90(LBA4, lba4)(__INT4_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    lb = va_arg(va, __INT_T *);
    if (!ISPRESENT(lb))
      __fort_abort("LBOUND: lower bound not present");
    ub = va_arg(va, __INT_T *);
    if (!ISPRESENT(ub))
      /* presumably, it's the last dimension of an assumed array */
      *arr++ = *lb;
    else
      *arr++ = (*lb <= *ub) ? *lb : 1;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<size>, __INT_T *ub<size> */
void
ENTF90(LBA8, lba8)(__INT8_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    lb = va_arg(va, __INT_T *);
    if (!ISPRESENT(lb))
      __fort_abort("LBOUND: lower bound not present");
    ub = va_arg(va, __INT_T *);
    if (!ISPRESENT(ub))
      /* presumably, it's the last dimension of an assumed array */
      *arr++ = *lb;
    else
      *arr++ = (*lb <= *ub) ? *lb : 1;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<size>, __INT_T *ub<size> */ 
void
ENTF90(KLBA, klba)(__INT8_T *arr, __INT4_T *size, ...)
{

  /* 
   * -i8 variant of LBA
   */

  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    lb = va_arg(va, __INT_T *);
    if (!ISPRESENT(lb))
      __fort_abort("LBOUND: lower bound not present");
    ub = va_arg(va, __INT_T *);
    if (!ISPRESENT(ub))
      /* presumably, it's the last dimension of an assumed array */
      *arr++ = *lb;
    else
      *arr++ = (*lb <= *ub) ? *lb : 1;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<size>, __INT_T *ub<size> */
void
ENTF90(UBA, uba)(__INT_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    lb = va_arg(va, __INT_T *);
    ub = va_arg(va, __INT_T *);
    if (!ISPRESENT(ub))
      __fort_abort("UBOUND: upper bound not present");
    *arr++ = (*lb <= *ub) ? *ub : 0;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<size>, __INT_T *ub<size> */
void
ENTF90(UBA1, uba1)(__INT1_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    lb = va_arg(va, __INT_T *);
    ub = va_arg(va, __INT_T *);
    if (!ISPRESENT(ub))
      __fort_abort("UBOUND: upper bound not present");
    *arr++ = (*lb <= *ub) ? *ub : 0;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<size>, __INT_T *ub<size> */
void
ENTF90(UBA2, uba2)(__INT2_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    lb = va_arg(va, __INT_T *);
    ub = va_arg(va, __INT_T *);
    if (!ISPRESENT(ub))
      __fort_abort("UBOUND: upper bound not present");
    *arr++ = (*lb <= *ub) ? *ub : 0;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<size>, __INT_T *ub<size> */
void
ENTF90(UBA4, uba4)(__INT4_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    lb = va_arg(va, __INT_T *);
    ub = va_arg(va, __INT_T *);
    if (!ISPRESENT(ub))
      __fort_abort("UBOUND: upper bound not present");
    *arr++ = (*lb <= *ub) ? *ub : 0;
  }
  va_end(va);
}

/* Varargs; __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<size>, __INT_T *ub<size> */
void
ENTF90(UBA8, uba8)(__INT8_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    lb = va_arg(va, __INT_T *);
    ub = va_arg(va, __INT_T *);
    if (!ISPRESENT(ub))
      __fort_abort("UBOUND: upper bound not present");
    *arr++ = (*lb <= *ub) ? *ub : 0;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<size>, __INT_T *ub<size> */
void
ENTF90(KUBA, kuba)(__INT8_T *arr, __INT4_T *size, ...)
{

  /* 
   * -i8 variant of UBA
   */

  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    lb = va_arg(va, __INT_T *);
    ub = va_arg(va, __INT_T *);
    if (!ISPRESENT(ub))
      __fort_abort("UBOUND: upper bound not present");
    *arr++ = (*lb <= *ub) ? *ub : 0;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<size>, __INT_T *ub<size> */
void
ENTF90(LBAZ, lbaz)(__INT4_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    lb = va_arg(va, __INT_T *);
    if (!ISPRESENT(lb))
      __fort_abort("LBOUND: lower bound not present");
    ub = va_arg(va, __INT_T *);
    if (!ISPRESENT(ub))
      /* presumably, it's the last dimension of an assumed array */
      *arr++ = *lb;
    else
      *arr++ = (*lb <= *ub) ? *lb : 1;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<size>, __INT_T *ub<size> */
void
ENTF90(LBAZ1, lbaz1)(__INT1_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    lb = va_arg(va, __INT_T *);
    if (!ISPRESENT(lb))
      __fort_abort("LBOUND: lower bound not present");
    ub = va_arg(va, __INT_T *);
    if (!ISPRESENT(ub))
      /* presumably, it's the last dimension of an assumed array */
      *arr++ = *lb;
    else
      *arr++ = (*lb <= *ub) ? *lb : 1;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<size>, __INT_T *ub<size> */
void
ENTF90(LBAZ2, lbaz2)(__INT2_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    lb = va_arg(va, __INT_T *);
    if (!ISPRESENT(lb))
      __fort_abort("LBOUND: lower bound not present");
    ub = va_arg(va, __INT_T *);
    if (!ISPRESENT(ub))
      /* presumably, it's the last dimension of an assumed array */
      *arr++ = *lb;
    else
      *arr++ = (*lb <= *ub) ? *lb : 1;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<size>, __INT_T *ub<size> */
void
ENTF90(LBAZ4, lbaz4)(__INT4_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    lb = va_arg(va, __INT_T *);
    if (!ISPRESENT(lb))
      __fort_abort("LBOUND: lower bound not present");
    ub = va_arg(va, __INT_T *);
    if (!ISPRESENT(ub))
      /* presumably, it's the last dimension of an assumed array */
      *arr++ = *lb;
    else
      *arr++ = (*lb <= *ub) ? *lb : 1;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<size>, __INT_T *ub<size> */
void
ENTF90(LBAZ8, lbaz8)(__INT8_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    lb = va_arg(va, __INT_T *);
    if (!ISPRESENT(lb))
      __fort_abort("LBOUND: lower bound not present");
    ub = va_arg(va, __INT_T *);
    if (!ISPRESENT(ub))
      /* presumably, it's the last dimension of an assumed array */
      *arr++ = *lb;
    else
      *arr++ = (*lb <= *ub) ? *lb : 1;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<size>, __INT_T *ub<size> */
void
ENTF90(KLBAZ, klbaz)(__INT8_T *arr, __INT4_T *size, ...)
{
  /* 
   * -i8 variant of LBAZ
   */

  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    lb = va_arg(va, __INT_T *);
    if (!ISPRESENT(lb))
      __fort_abort("LBOUND: lower bound not present");
    ub = va_arg(va, __INT_T *);
    if (!ISPRESENT(ub))
      /* presumably, it's the last dimension of an assumed array */
      *arr++ = *lb;
    else
      *arr++ = (*lb <= *ub) ? *lb : 1;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<size>, __INT_T *ub<size> */
void
ENTF90(UBAZ, ubaz)(__INT4_T *arr, __INT_T *size, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    lb = va_arg(va, __INT_T *);
    ub = va_arg(va, __INT_T *);
    if (!ISPRESENT(ub))
      __fort_abort("UBOUND: upper bound not present");
    *arr++ = (*lb <= *ub) ? *ub : 0;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<size>, __INT_T *ub<size> */
void
ENTF90(UBAZ1, ubaz1)(__INT1_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    lb = va_arg(va, __INT_T *);
    ub = va_arg(va, __INT_T *);
    if (!ISPRESENT(ub))
      __fort_abort("UBOUND: upper bound not present");
    *arr++ = (*lb <= *ub) ? *ub : 0;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<size>, __INT_T *ub<size> */
void
ENTF90(UBAZ2, ubaz2)(__INT2_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    lb = va_arg(va, __INT_T *);
    ub = va_arg(va, __INT_T *);
    if (!ISPRESENT(ub))
      __fort_abort("UBOUND: upper bound not present");
    *arr++ = (*lb <= *ub) ? *ub : 0;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<size>, __INT_T *ub<size> */
void
ENTF90(UBAZ4, ubaz4)(__INT4_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    lb = va_arg(va, __INT_T *);
    ub = va_arg(va, __INT_T *);
    if (!ISPRESENT(ub))
      __fort_abort("UBOUND: upper bound not present");
    *arr++ = (*lb <= *ub) ? *ub : 0;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<size>, __INT_T *ub<size> */
void
ENTF90(UBAZ8, ubaz8)(__INT8_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    lb = va_arg(va, __INT_T *);
    ub = va_arg(va, __INT_T *);
    if (!ISPRESENT(ub))
      __fort_abort("UBOUND: upper bound not present");
    *arr++ = (*lb <= *ub) ? *ub : 0;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, __INT_T *ub1, ... __INT_T *lb<size>, __INT_T *ub<size> */
void 
ENTF90(KUBAZ, kubaz)(__INT8_T *arr, __INT4_T *size, ...)
{

  /* 
   * -i8 variant of UBAZ
   */

  va_list va;
  __INT_T *lb;
  __INT_T *ub;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    lb = va_arg(va, __INT_T *);
    ub = va_arg(va, __INT_T *);
    if (!ISPRESENT(ub))
      __fort_abort("UBOUND: upper bound not present");
    *arr++ = (*lb <= *ub) ? *ub : 0;
  }
  va_end(va);
}

/* Varargs: __INT_T *ub1, ... __INT_T *ub<rank> */
__INT_T
ENTF90(LBOUND, lbound)(__INT4_T *rank, __INT4_T *dim, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T d;

  d = *dim;
  if (d < 1 || d > *rank)
    __fort_abort("LBOUND: invalid dim");
  va_start(va, dim);
  while (d-- > 0)
    bnd = va_arg(va, __INT_T *);
  va_end(va);
  if (!ISPRESENT(bnd))
    __fort_abort("LBOUND: lower bound not present for specified dim");
  return *bnd;
}

/* Varargs: __INT_T *ub1, ... __INT_T *ub<rank> */
__INT1_T
ENTF90(LBOUND1, lbound1)(__INT4_T *rank, __INT4_T *dim, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T d;

  d = *dim;
  if (d < 1 || d > *rank)
    __fort_abort("LBOUND: invalid dim");
  va_start(va, dim);
  while (d-- > 0)
    bnd = va_arg(va, __INT_T *);
  va_end(va);
  if (!ISPRESENT(bnd))
    __fort_abort("LBOUND: lower bound not present for specified dim");
  return (__INT1_T)*bnd;
}

/* Varargs: __INT_T *ub1, ... __INT_T *ub<rank> */
__INT2_T
ENTF90(LBOUND2, lbound2)(__INT4_T *rank, __INT4_T *dim, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T d;

  d = *dim;
  if (d < 1 || d > *rank)
    __fort_abort("LBOUND: invalid dim");
  va_start(va, dim);
  while (d-- > 0)
    bnd = va_arg(va, __INT_T *);
  va_end(va);
  if (!ISPRESENT(bnd))
    __fort_abort("LBOUND: lower bound not present for specified dim");
  return (__INT2_T)*bnd;
}

/* Varargs: __INT_T *ub1, ... __INT_T *ub<rank> */
__INT4_T
ENTF90(LBOUND4, lbound4)(__INT4_T *rank, __INT4_T *dim, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T d;

  d = *dim;
  if (d < 1 || d > *rank)
    __fort_abort("LBOUND: invalid dim");
  va_start(va, dim);
  while (d-- > 0)
    bnd = va_arg(va, __INT_T *);
  va_end(va);
  if (!ISPRESENT(bnd))
    __fort_abort("LBOUND: lower bound not present for specified dim");
  return (__INT4_T)*bnd;
}

/* Varargs: __INT_T *ub1, ... __INT_T *ub<rank> */
__INT8_T
ENTF90(LBOUND8, lbound8)(__INT4_T *rank, __INT4_T *dim, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T d;

  d = *dim;
  if (d < 1 || d > *rank)
    __fort_abort("LBOUND: invalid dim");
  va_start(va, dim);
  while (d-- > 0)
    bnd = va_arg(va, __INT_T *);
  va_end(va);
  if (!ISPRESENT(bnd))
    __fort_abort("LBOUND: lower bound not present for specified dim");
  return (__INT8_T)*bnd;
}

/* Varargs: __INT_T *ub1, ... __INT_T *ub<rank> */
__INT8_T
ENTF90(KLBOUND, klbound)(__INT4_T *rank, __INT4_T *dim, ...)
{

  /* 
   * -i8 variant of lbound
   */

  va_list va;
  __INT_T *bnd;
  __INT_T d;

  d = *dim;
  if (d < 1 || d > *rank)
    __fort_abort("LBOUND: invalid dim");
  va_start(va, dim);

  while (d-- > 0)
    bnd = va_arg(va, __INT_T *);
  va_end(va);
  if (!ISPRESENT(bnd))
    __fort_abort("LBOUND: lower bound not present for specified dim");
  return (__INT8_T)*bnd;
}

/* Varargs: __INT_T *ub1, ... __INT_T *ub<rank> */
__INT_T
ENTF90(UBOUND, ubound)(__INT4_T *rank, __INT4_T *dim, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T d;

  d = *dim;
  if (d < 1 || d > *rank)
    __fort_abort("UBOUND: invalid dim");
  va_start(va, dim);
  while (d-- > 0)
    bnd = va_arg(va, __INT_T *);
  va_end(va);
  if (!ISPRESENT(bnd))
    __fort_abort("UBOUND: upper bound not present for specified dim");
  return *bnd;
}

/* Varargs: __INT_T *ub1, ... __INT_T *ub<rank> */
__INT1_T
ENTF90(UBOUND1, ubound1)(__INT4_T *rank, __INT4_T *dim, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T d;

  d = *dim;
  if (d < 1 || d > *rank)
    __fort_abort("UBOUND: invalid dim");
  va_start(va, dim);
  while (d-- > 0)
    bnd = va_arg(va, __INT_T *);
  va_end(va);
  if (!ISPRESENT(bnd))
    __fort_abort("UBOUND: upper bound not present for specified dim");
  return (__INT1_T)*bnd;
}

/* Varargs: __INT_T *ub1, ... __INT_T *ub<rank> */
__INT2_T
ENTF90(UBOUND2, ubound2)(__INT4_T *rank, __INT4_T *dim, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T d;

  d = *dim;
  if (d < 1 || d > *rank)
    __fort_abort("UBOUND: invalid dim");
  va_start(va, dim);
  while (d-- > 0)
    bnd = va_arg(va, __INT_T *);
  va_end(va);
  if (!ISPRESENT(bnd))
    __fort_abort("UBOUND: upper bound not present for specified dim");
  return (__INT2_T)*bnd;
}

/* Varargs: __INT_T *ub1, ... __INT_T *ub<rank> */
__INT4_T
ENTF90(UBOUND4, ubound4)(__INT4_T *rank, __INT4_T *dim, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T d;

  d = *dim;
  if (d < 1 || d > *rank)
    __fort_abort("UBOUND: invalid dim");
  va_start(va, dim);
  while (d-- > 0)
    bnd = va_arg(va, __INT_T *);
  va_end(va);
  if (!ISPRESENT(bnd))
    __fort_abort("UBOUND: upper bound not present for specified dim");
  return (__INT4_T)*bnd;
}

/* Varargs: __INT_T *ub1, ... __INT_T *ub<rank> */
__INT8_T
ENTF90(UBOUND8, ubound8)(__INT4_T *rank, __INT4_T *dim, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T d;

  d = *dim;
  if (d < 1 || d > *rank)
    __fort_abort("UBOUND: invalid dim");
  va_start(va, dim);
  while (d-- > 0)
    bnd = va_arg(va, __INT_T *);
  va_end(va);
  if (!ISPRESENT(bnd))
    __fort_abort("UBOUND: upper bound not present for specified dim");
  return (__INT8_T)*bnd;
}

/* Varargs: __INT_T *ub1, ... __INT_T *ub<rank> */
__INT8_T
ENTF90(KUBOUND, kubound)(__INT4_T *rank, __INT4_T *dim, ...)
{

  /* 
   * -i8 variant of ubound
   */

  va_list va;
  __INT_T *bnd;
  __INT_T d;

  d = *dim;
  if (d < 1 || d > *rank)
    __fort_abort("UBOUND: invalid dim");
  va_start(va, dim);
  while (d-- > 0)
    bnd = va_arg(va, __INT_T *);
  va_end(va);
  if (!ISPRESENT(bnd))
    __fort_abort("UBOUND: upper bound not present for specified dim");
  return (__INT8_T)*bnd;
}

/* Varargs: __INT_T *lb1, ... __INT_T *lb<size> */
void
ENTF90(LBOUNDA, lbounda)(__INT_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    bnd = va_arg(va, __INT_T *);
    if (!ISPRESENT(bnd))
      __fort_abort("LBOUND: lower bound not present");
    *arr++ = *bnd;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, ... __INT_T *lb<size> */
void
ENTF90(LBOUNDA1, lbounda1)(__INT1_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    bnd = va_arg(va, __INT_T *);
    if (!ISPRESENT(bnd))
      __fort_abort("LBOUND: lower bound not present");
    *arr++ = *bnd;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, ... __INT_T *lb<size> */
void
ENTF90(LBOUNDA2, lbounda2)(__INT2_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    bnd = va_arg(va, __INT_T *);
    if (!ISPRESENT(bnd))
      __fort_abort("LBOUND: lower bound not present");
    *arr++ = *bnd;
  }
  va_end(va);
}

/*Varargs: __INT_T *lb1, ... __INT_T *lb<size> */
void
ENTF90(LBOUNDA4, lbounda4)(__INT4_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    bnd = va_arg(va, __INT_T *);
    if (!ISPRESENT(bnd))
      __fort_abort("LBOUND: lower bound not present");
    *arr++ = *bnd;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, ... __INT_T *lb<size> */
void
ENTF90(LBOUNDA8, lbounda8)(__INT8_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    bnd = va_arg(va, __INT_T *);
    if (!ISPRESENT(bnd))
      __fort_abort("LBOUND: lower bound not present");
    *arr++ = *bnd;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, ... __INT_T *lb<size> */
void
ENTF90(KLBOUNDA, klbounda)(__INT8_T *arr, __INT4_T *size, ...)
{

  /* 
   * -i8 variant of LBOUNDA
   */

  va_list va;
  __INT_T *bnd;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    bnd = va_arg(va, __INT_T *);
    if (!ISPRESENT(bnd))
      __fort_abort("LBOUND: lower bound not present");
    *arr++ = *bnd;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, ... __INT_T *lb<size> */
void
ENTF90(UBOUNDA, ubounda)(__INT_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    bnd = va_arg(va, __INT_T *);
    if (!ISPRESENT(bnd))
      __fort_abort("UBOUND: upper bound not present");
    *arr++ = *bnd;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, ... __INT_T *lb<size> */
void
ENTF90(UBOUNDA1, ubounda1)(__INT1_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    bnd = va_arg(va, __INT_T *);
    if (!ISPRESENT(bnd))
      __fort_abort("UBOUND: upper bound not present");
    *arr++ = *bnd;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, ... __INT_T *lb<size> */
void
ENTF90(UBOUNDA2, ubounda2)(__INT2_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    bnd = va_arg(va, __INT_T *);
    if (!ISPRESENT(bnd))
      __fort_abort("UBOUND: upper bound not present");
    *arr++ = *bnd;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, ... __INT_T *lb<size> */
void
ENTF90(UBOUNDA4, ubounda4)(__INT4_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    bnd = va_arg(va, __INT_T *);
    if (!ISPRESENT(bnd))
      __fort_abort("UBOUND: upper bound not present");
    *arr++ = *bnd;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, ... __INT_T *lb<size> */
void
ENTF90(UBOUNDA8, ubounda8)(__INT8_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    bnd = va_arg(va, __INT_T *);
    if (!ISPRESENT(bnd))
      __fort_abort("UBOUND: upper bound not present");
    *arr++ = *bnd;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, ... __INT_T *lb<size> */
void
ENTF90(KUBOUNDA, kubounda)(__INT8_T *arr, __INT4_T *size, ...)
{

  /* 
   * -i8 variant of ubounda
   */

  va_list va;
  __INT_T *bnd;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    bnd = va_arg(va, __INT_T *);
    if (!ISPRESENT(bnd))
      __fort_abort("UBOUND: upper bound not present");
    *arr++ = *bnd;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, ... __INT_T *lb<size> */
void
ENTF90(LBOUNDAZ, lboundaz)(__INT4_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    bnd = va_arg(va, __INT_T *);
    if (!ISPRESENT(bnd))
      __fort_abort("LBOUND: lower bound not present");
    *arr++ = *bnd;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, ... __INT_T *lb<size> */
void
ENTF90(LBOUNDAZ1, lboundaz1)(__INT1_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    bnd = va_arg(va, __INT_T *);
    if (!ISPRESENT(bnd))
      __fort_abort("LBOUND: lower bound not present");
    *arr++ = *bnd;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, ... __INT_T *lb<size> */
void
ENTF90(LBOUNDAZ2, lboundaz2)(__INT2_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    bnd = va_arg(va, __INT_T *);
    if (!ISPRESENT(bnd))
      __fort_abort("LBOUND: lower bound not present");
    *arr++ = *bnd;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, ... __INT_T *lb<size> */
void
ENTF90(LBOUNDAZ4, lboundaz4)(__INT4_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    bnd = va_arg(va, __INT_T *);
    if (!ISPRESENT(bnd))
      __fort_abort("LBOUND: lower bound not present");
    *arr++ = *bnd;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, ... __INT_T *lb<size> */
void
ENTF90(LBOUNDAZ8, lboundaz8)(__INT8_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    bnd = va_arg(va, __INT_T *);
    if (!ISPRESENT(bnd))
      __fort_abort("LBOUND: lower bound not present");
    *arr++ = *bnd;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, ... __INT_T *lb<size> */
void
ENTF90(KLBOUNDAZ, klboundaz)(__INT8_T *arr, __INT4_T *size, ...)
{

  /* 
   * -i8 variant of LBOUNDAZ
   */

  va_list va;
  __INT_T *bnd;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    bnd = va_arg(va, __INT_T *);
    if (!ISPRESENT(bnd))
      __fort_abort("LBOUND: lower bound not present");
    *arr++ = *bnd;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, ... __INT_T *lb<size> */
void
ENTF90(UBOUNDAZ, uboundaz)(__INT4_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    bnd = va_arg(va, __INT_T *);
    if (!ISPRESENT(bnd))
      __fort_abort("UBOUND: upper bound not present");
    *arr++ = *bnd;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, ... __INT_T *lb<size> */
void
ENTF90(UBOUNDAZ1, uboundaz1)(__INT1_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    bnd = va_arg(va, __INT_T *);
    if (!ISPRESENT(bnd))
      __fort_abort("UBOUND: upper bound not present");
    *arr++ = *bnd;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, ... __INT_T *lb<size> */
void
ENTF90(UBOUNDAZ2, uboundaz2)(__INT2_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    bnd = va_arg(va, __INT_T *);
    if (!ISPRESENT(bnd))
      __fort_abort("UBOUND: upper bound not present");
    *arr++ = *bnd;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, ... __INT_T *lb<size> */
void
ENTF90(UBOUNDAZ4, uboundaz4)(__INT4_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    bnd = va_arg(va, __INT_T *);
    if (!ISPRESENT(bnd))
      __fort_abort("UBOUND: upper bound not present");
    *arr++ = *bnd;
  }
  va_end(va);
}

/* Varargs: __INT_T *lb1, ... __INT_T *lb<size> */
void
ENTF90(UBOUNDAZ8, uboundaz8)(__INT8_T *arr, __INT4_T *size, ...)
{
  va_list va;
  __INT_T *bnd;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    bnd = va_arg(va, __INT_T *);
    if (!ISPRESENT(bnd))
      __fort_abort("UBOUND: upper bound not present");
    *arr++ = *bnd;
  }
  va_end(va);
}

/* Vargs: __INT_T *lb1, ... __INT_T *lb<size> */
void
ENTF90(KUBOUNDAZ, kuboundaz)(__INT8_T *arr, __INT4_T *size, ...)
{

  /* 
   * -i8 variant of uboundaz
   */

  va_list va;
  __INT_T *bnd;
  __INT_T s;

  s = *size;
  va_start(va, size);
  while (s-- > 0) {
    bnd = va_arg(va, __INT_T *);
    if (!ISPRESENT(bnd))
      __fort_abort("UBOUND: upper bound not present");
    *arr++ = *bnd;
  }
  va_end(va);
}

/* Vargs: { *lwb, *upb, *stride }* */
__INT_T
ENTF90(SIZE, size)(__INT4_T *rank, __INT4_T *dim, ...)
{
  va_list va;
  __INT_T *lwb, *upb, *stride;
  int d;
  __INT_T extent;

  va_start(va, dim);

  if (!ISPRESENT(dim)) {

    /* size = product of all extents */

    extent = 1;
    d = *rank;
    while (d-- > 0) {
      lwb = va_arg(va, __INT_T *);
      upb = va_arg(va, __INT_T *);
      stride = va_arg(va, __INT_T *);
      if (!ISPRESENT(lwb) || !ISPRESENT(upb) || !ISPRESENT(stride))
        __fort_abort("SIZE: bounds not present");
      extent *= (*upb - *lwb + *stride) / *stride;
      if (extent < 0)
        extent = 0;
    }
  } else {

    /* size = extent in dimension 'dim' */

    d = *dim;
    if (d < 1 || d > *rank)
      __fort_abort("SIZE: invalid dim");

    while (d-- > 0) {
      lwb = va_arg(va, __INT_T *);
      upb = va_arg(va, __INT_T *);
      stride = va_arg(va, __INT_T *);
    }
    if (!ISPRESENT(lwb) || !ISPRESENT(upb) || !ISPRESENT(stride))
      __fort_abort("SIZE: bounds not present for specified dim");
    extent = (*upb - *lwb + *stride) / *stride;
    if (extent < 0)
      extent = 0;
  }
  va_end(va);
  return extent;
}

/* Vargs: { *lwb, *upb, *stride }* */
__INT8_T
ENTF90(KSIZE, ksize)(__INT4_T *rank, __INT4_T *dim, ...)
{

  /* 
   * -i8 variant of SIZE
   */

  va_list va;
  __INT_T *lwb, *upb, *stride;
  int d;
  __INT_T extent;

  va_start(va, dim);

  if (!ISPRESENT(dim)) {

    /* size = product of all extents */

    extent = 1;
    d = *rank;
    while (d-- > 0) {
      lwb = va_arg(va, __INT_T *);
      upb = va_arg(va, __INT_T *);
      stride = va_arg(va, __INT_T *);
      if (!ISPRESENT(lwb) || !ISPRESENT(upb) || !ISPRESENT(stride))
        __fort_abort("SIZE: bounds not present");
      extent *= (*upb - *lwb + *stride) / *stride;
      if (extent < 0)
        extent = 0;
    }
  } else {

    /* size = extent in dimension 'dim' */

    d = *dim;
    if (d < 1 || d > *rank)
      __fort_abort("SIZE: invalid dim");

    while (d-- > 0) {
      lwb = va_arg(va, __INT_T *);
      upb = va_arg(va, __INT_T *);
      stride = va_arg(va, __INT_T *);
    }
    if (!ISPRESENT(lwb) || !ISPRESENT(upb) || !ISPRESENT(stride))
      __fort_abort("SIZE: bounds not present for specified dim");
    extent = (*upb - *lwb + *stride) / *stride;
    if (extent < 0)
      extent = 0;
  }
  va_end(va);
  return (__INT8_T)extent;
}

#define BITS_PER_BYTE 8
__INT8_T
ENTF90(CLASS_OBJ_SIZE, class_obj_size)(F90_Desc *d)
{
  __INT8_T storage_sz;

  if (!d->tag)
    return 0;

  storage_sz = ENTF90(KGET_OBJECT_SIZE, kget_object_size)(d);
  return storage_sz;
}

/* Vargs: { *lwb, *upb, *stride }* */
void
ENTF90(SHAPE, shape)(__INT4_T *arr, __INT_T *rank, ...)
{
  va_list va;
  __INT_T *lwb, *upb, *stride;
  int d;
  __INT_T extent;

  d = *rank;
  va_start(va, rank);
  while (d-- > 0) {
    lwb = va_arg(va, __INT_T *);
    upb = va_arg(va, __INT_T *);
    stride = va_arg(va, __INT_T *);
    if (!ISPRESENT(lwb) || !ISPRESENT(upb) || !ISPRESENT(stride))
      __fort_abort("SHAPE: bounds not present");
    extent = (*upb - *lwb + *stride) / *stride;
    if (extent < 0)
      extent = 0;
    *arr++ = extent;
  }
  va_end(va);
}

/* Vargs: { *lwb, *upb, *stride }* */
void
ENTF90(SHAPE1, shape1)(__INT1_T *arr, __INT_T *rank, ...)
{
  va_list va;
  __INT_T *lwb, *upb, *stride;
  int d;
  __INT_T extent;

  d = *rank;
  va_start(va, rank);
  while (d-- > 0) {
    lwb = va_arg(va, __INT_T *);
    upb = va_arg(va, __INT_T *);
    stride = va_arg(va, __INT_T *);
    if (!ISPRESENT(lwb) || !ISPRESENT(upb) || !ISPRESENT(stride))
      __fort_abort("SHAPE: bounds not present");
    extent = (*upb - *lwb + *stride) / *stride;
    if (extent < 0)
      extent = 0;
    *arr++ = extent;
  }
  va_end(va);
}

/* Vargs: { *lwb, *upb, *stride }* */
void
ENTF90(SHAPE2, shape2)(__INT2_T *arr, __INT_T *rank, ...)
{
  va_list va;
  __INT_T *lwb, *upb, *stride;
  int d;
  __INT_T extent;

  d = *rank;
  va_start(va, rank);
  while (d-- > 0) {
    lwb = va_arg(va, __INT_T *);
    upb = va_arg(va, __INT_T *);
    stride = va_arg(va, __INT_T *);
    if (!ISPRESENT(lwb) || !ISPRESENT(upb) || !ISPRESENT(stride))
      __fort_abort("SHAPE: bounds not present");
    extent = (*upb - *lwb + *stride) / *stride;
    if (extent < 0)
      extent = 0;
    *arr++ = extent;
  }
  va_end(va);
}

/* Vargs: { *lwb, *upb, *stride }* */
void
ENTF90(SHAPE4, shape4)(__INT4_T *arr, __INT_T *rank, ...)
{
  va_list va;
  __INT_T *lwb, *upb, *stride;
  int d;
  __INT_T extent;

  d = *rank;
  va_start(va, rank);
  while (d-- > 0) {
    lwb = va_arg(va, __INT_T *);
    upb = va_arg(va, __INT_T *);
    stride = va_arg(va, __INT_T *);
    if (!ISPRESENT(lwb) || !ISPRESENT(upb) || !ISPRESENT(stride))
      __fort_abort("SHAPE: bounds not present");
    extent = (*upb - *lwb + *stride) / *stride;
    if (extent < 0)
      extent = 0;
    *arr++ = extent;
  }
  va_end(va);
}

/* Vargs: { *lwb, *upb, *stride }* */
void
ENTF90(SHAPE8, shape8)(__INT8_T *arr, __INT_T *rank, ...)
{
  va_list va;
  __INT_T *lwb, *upb, *stride;
  int d;
  __INT_T extent;

  d = *rank;
  va_start(va, rank);
  while (d-- > 0) {
    lwb = va_arg(va, __INT_T *);
    upb = va_arg(va, __INT_T *);
    stride = va_arg(va, __INT_T *);
    if (!ISPRESENT(lwb) || !ISPRESENT(upb) || !ISPRESENT(stride))
      __fort_abort("SHAPE: bounds not present");
    extent = (*upb - *lwb + *stride) / *stride;
    if (extent < 0)
      extent = 0;
    *arr++ = extent;
  }
  va_end(va);
}

/** \brief Vargs: { *lwb, *upb, *stride }* */
void 
ENTF90(KSHAPE, kshape)(__INT8_T *arr, __INT_T *rank, ...)
{
  va_list va;
  __INT_T *lwb, *upb, *stride;
  int d;
  __INT_T extent;

  d = *rank;
  va_start(va, rank);
  while (d-- > 0) {
    lwb = va_arg(va, __INT_T *);
    upb = va_arg(va, __INT_T *);
    stride = va_arg(va, __INT_T *);
    if (!ISPRESENT(lwb) || !ISPRESENT(upb) || !ISPRESENT(stride))
      __fort_abort("SHAPE: bounds not present");
    extent = (*upb - *lwb + *stride) / *stride;
    if (extent < 0)
      extent = 0;
    *arr++ = extent;
  }
  va_end(va);
}

/** \brief ACHAR function returns LEN;
 *
 * avoid confusion on returning character result
 */
__INT_T
ENTF90(ACHARA, achara)
(DCHAR(res), void *i, __INT_T *size DCLEN64(res))
{
  *CADR(res) = I8(__fort_varying_int)(i, size);
  return 1;
}
/* 32 bit CLEN version */
__INT_T
ENTF90(ACHAR, achar)
(DCHAR(res), void *i, __INT_T *size DCLEN(res))
{
  return ENTF90(ACHARA, achara) (CADR(res), i, size, (__CLEN_T)CLEN(res));
}

__CLEN_T
ENTF90(REPEATA, repeata)
(DCHAR(res), DCHAR(expr), void *ncopies, __INT_T *size DCLEN64(res) DCLEN64(expr))
{
  __CLEN_T len;
  int i;
  int _ncopies;

  len = CLEN(expr);
  _ncopies = I8(__fort_varying_int)(ncopies, size);
  for (i = 0; i < _ncopies; ++i) {
    strncpy(CADR(res) + i * len, CADR(expr), len);
  }
  return _ncopies * len;
}
/* 32 bit CLEN version */
__INT_T
ENTF90(REPEAT, repeat)
(DCHAR(res), DCHAR(expr), void *ncopies, __INT_T *size DCLEN(res) DCLEN(expr))
{
  return (__INT_T)ENTF90(REPEATA, repeata) (CADR(res), CADR(expr), ncopies,
                               size, (__CLEN_T)CLEN(res), (__CLEN_T)CLEN(expr));
}

__INT_T
ENTF90(TRIMA, trima)
(DCHAR(res), DCHAR(expr) DCLEN64(res) DCLEN64(expr))
{
  int i, j;
  char *rcptr;
  char *ecptr;

  /*
   *  The for loop below results in a call to _c_mcopy1.
   *  Not the most efficient thing to do when len is around 4 or so.
   *  If the target generates illegal alignment errors, need to not do
   *  int copies.  So, enable fast code for x8664 only for now.
   *
      i = CLEN(expr)-1;
      while (i >= 0 && CADR(expr)[i] == ' ')
          --i;
      if (i < 0)
          return 0;
      for (j = 0; j <= i; ++j)
          CADR(res)[j] = CADR(expr)[j];
      return i+1;
  */
  i = CLEN(expr);
  while (i > 0) {
    if (CADR(expr)[i - 1] != ' ') {
#if defined(TARGET_X8664) || defined(TARGET_LLVM_ARM64)
      if (i <= 11) {
        int *rptr = ((int *)CADR(res));
        int *eptr = ((int *)CADR(expr));
        if (i & 0xc) {
          *rptr = *eptr;
          if (i == 4)
            return i;
          rptr++;
          eptr++;
          if (i & 8) {
            *rptr = *eptr;
            if (i == 8)
              return i;
            rptr++;
            eptr++;
          }
        }
        rcptr = (char *)rptr;
        ecptr = (char *)eptr;
#else
      if (i <= 3) {
        rcptr = ((char *)CADR(res));
        ecptr = ((char *)CADR(expr));
#endif
        j = i & 3;
        if (j > 2)
          *rcptr++ = *ecptr++;
        if (j > 1)
          *rcptr++ = *ecptr++;
        if (j > 0)
          *rcptr = *ecptr;
      } else {
        for (j = 0; j < i; ++j)
          CADR(res)[j] = CADR(expr)[j];
      }
      return i;
    } else {
      --i;
    }
  }
  return 0;
}
/* 32 bit CLEN version */
__INT_T
ENTF90(TRIM, trim)
(DCHAR(res), DCHAR(expr) DCLEN(res) DCLEN(expr))
{
  return ENTF90(TRIMA, trima) (CADR(res), CADR(expr), (__CLEN_T)CLEN(res), (__CLEN_T)CLEN(expr));
}

__INT_T
ENTF90(IACHARA, iachara)(DCHAR(c) DCLEN64(c)) { return *CADR(c); }
/* 32 bit CLEN version */
__INT_T
ENTF90(IACHAR, iachar)(DCHAR(c) DCLEN(c))
{
  return ENTF90(IACHARA, iachara)(CADR(c), (__CLEN_T)CLEN(c));
}

/** \brief
 * -i8 variant of iachar
 */
__INT8_T
ENTF90(KIACHARA, kiachara)(DCHAR(c) DCLEN64(c))
{
  return (__INT8_T)*CADR(c);
}
/* 32 bit CLEN version */
__INT8_T
ENTF90(KIACHAR, kiachar)(DCHAR(c) DCLEN(c))
{
  return ENTF90(KIACHARA, kiachara)(CADR(c), (__CLEN_T)CLEN(c));
}

void 
ENTF90(MERGECHA, mergecha)(DCHAR(result), DCHAR(tsource), DCHAR(fsource),
                              void *mask, __INT_T *szmask DCLEN64(result)
                                              DCLEN64(tsource) DCLEN64(fsource))
{
  if (I8(__fort_varying_log)(mask, szmask))
    fstrcpy(CADR(result), CADR(tsource), CLEN(result), CLEN(tsource));
  else
    fstrcpy(CADR(result), CADR(fsource), CLEN(result), CLEN(fsource));
}
/* 32 bit CLEN version */
void 
ENTF90(MERGECH, mergech)(DCHAR(result), DCHAR(tsource), DCHAR(fsource),
                              void *mask, __INT_T *szmask DCLEN(result)
                                              DCLEN(tsource) DCLEN(fsource))
{
  ENTF90(MERGECHA, mergecha)(CADR(result), CADR(tsource), CADR(fsource), mask,
                             szmask, (__CLEN_T)CLEN(result),
                             (__CLEN_T)CLEN(tsource), (__CLEN_T)CLEN(fsource));
}

void 
ENTF90(MERGEDT, mergedt)(void *result, void *tsource, void *fsource,
                              __INT_T *size, void *mask, __INT_T *szmask)
{
  __fort_bcopy(result, I8(__fort_varying_log)(mask, szmask) ? tsource : fsource,
              *size);
}

__INT1_T
ENTF90(MERGEI1, mergei1)
(__INT1_T *tsource, __INT1_T *fsource, void *mask, __INT_T *size)
{
  return I8(__fort_varying_log)(mask, size) ? *tsource : *fsource;
}

__INT2_T
ENTF90(MERGEI2, mergei2)
(__INT2_T *tsource, __INT2_T *fsource, void *mask, __INT_T *size)
{
  return I8(__fort_varying_log)(mask, size) ? *tsource : *fsource;
}

__INT4_T
ENTF90(MERGEI, mergei)
(__INT4_T *tsource, __INT4_T *fsource, void *mask, __INT_T *size)
{
  return I8(__fort_varying_log)(mask, size) ? *tsource : *fsource;
}

__INT8_T
ENTF90(MERGEI8, mergei8)
(__INT8_T *tsource, __INT8_T *fsource, void *mask, __INT_T *size)
{
  return I8(__fort_varying_log)(mask, size) ? *tsource : *fsource;
}

__LOG1_T
ENTF90(MERGEL1, mergel1)
(__LOG1_T *tsource, __LOG1_T *fsource, void *mask, __INT_T *size)
{
  return I8(__fort_varying_log)(mask, size) ? *tsource : *fsource;
}

__LOG2_T
ENTF90(MERGEL2, mergel2)
(__LOG2_T *tsource, __LOG2_T *fsource, void *mask, __INT_T *size)
{
  return I8(__fort_varying_log)(mask, size) ? *tsource : *fsource;
}

__LOG4_T
ENTF90(MERGEL, mergel)
(__LOG4_T *tsource, __LOG4_T *fsource, void *mask, __INT_T *size)
{
  return I8(__fort_varying_log)(mask, size) ? *tsource : *fsource;
}

__LOG8_T
ENTF90(MERGEL8, mergel8)
(__LOG8_T *tsource, __LOG8_T *fsource, void *mask, __INT_T *size)
{
  return I8(__fort_varying_log)(mask, size) ? *tsource : *fsource;
}

__REAL4_T
ENTF90(MERGER, merger)
(__REAL4_T *tsource, __REAL4_T *fsource, void *mask, __INT_T *size)
{
  return I8(__fort_varying_log)(mask, size) ? *tsource : *fsource;
}

__REAL8_T
ENTF90(MERGED, merged)
(__REAL8_T *tsource, __REAL8_T *fsource, void *mask, __INT_T *size)
{
  return I8(__fort_varying_log)(mask, size) ? *tsource : *fsource;
}

__REAL16_T
ENTF90(MERGEQ, mergeq)
(__REAL16_T *tsource, __REAL16_T *fsource, void *mask, __INT_T *size)
{
  return I8(__fort_varying_log)(mask, size) ? *tsource : *fsource;
}

__INT_T
ENTF90(LENTRIMA, lentrima)(DCHAR(str) DCLEN64(str))
{
  __INT_T i;

  for (i = (__INT_T)CLEN(str); i-- > 0;)
    if (CADR(str)[i] != ' ')
      break;
  return i + 1;
}
/* 32 bit CLEN version */
__INT_T
ENTF90(LENTRIM, lentrim)(DCHAR(str) DCLEN(str))
{
  return ENTF90(LENTRIMA, lentrima)(CADR(str), (__CLEN_T)CLEN(str));
}

__INT8_T
ENTF90(KLENTRIMA, klentrima)(DCHAR(str) DCLEN64(str))
{
  /* 
   * -i8 variant of lentrim
   */

  __INT8_T i;

  for (i = (__INT8_T)CLEN(str); i-- > 0;)
    if (CADR(str)[i] != ' ')
      break;
  return i + 1;
}
/* 32 bit CLEN version */
__INT8_T
ENTF90(KLENTRIM, klentrim)(DCHAR(str) DCLEN(str))
{
  return (__INT8_T) ENTF90(KLENTRIMA, klentrima)(CADR(str), (__CLEN_T)CLEN(str));
}

__INT_T
ENTF90(SCANA, scana)
(DCHAR(str), DCHAR(set), void *back, __INT_T *size DCLEN64(str) DCLEN64(set))
{
  __INT_T i, j;

  if (ISPRESENT(back) && I8(__fort_varying_log)(back, size)) {
    for (i = (__INT_T)CLEN(str); i-- > 0;)
      for (j = 0; j < (__INT_T)CLEN(set); ++j)
        if (CADR(set)[j] == CADR(str)[i])
          return i + 1;
    return 0;
  } else {
    for (i = 0; i < (__INT_T)CLEN(str); ++i)
      for (j = 0; j < (__INT_T)CLEN(set); ++j)
        if (CADR(set)[j] == CADR(str)[i])
          return i + 1;
    return 0;
  }
}
/* 32 bit CLEN version */
__INT_T
ENTF90(SCAN, scan)
(DCHAR(str), DCHAR(set), void *back, __INT_T *size DCLEN(str) DCLEN(set))
{
  return ENTF90(SCANA, scana) (CADR(str), CADR(set), back, size,
                               (__CLEN_T)CLEN(str), (__CLEN_T)CLEN(set));
}

/** \brief
 * -i8 variant of SCAN
 */
__INT8_T
ENTF90(KSCANA, kscana)
(DCHAR(str), DCHAR(set), void *back, __INT_T *size DCLEN64(str) DCLEN64(set))
{
  __INT8_T i, j;

  if (ISPRESENT(back) && I8(__fort_varying_log)(back, size)) {
    for (i = (__INT8_T)CLEN(str); i-- > 0;)
      for (j = 0; j < (__INT8_T)CLEN(set); ++j)
        if (CADR(set)[j] == CADR(str)[i])
          return i + 1;
    return 0;
  } else {
    for (i = 0; i < (__INT8_T)CLEN(str); ++i)
      for (j = 0; j < (__INT8_T)CLEN(set); ++j)
        if (CADR(set)[j] == CADR(str)[i])
          return i + 1;
    return 0;
  }
}
/* 32 bit CLEN version */
__INT8_T
ENTF90(KSCAN, kscan)
(DCHAR(str), DCHAR(set), void *back, __INT_T *size DCLEN(str) DCLEN(set))
{
  return ENTF90(KSCANA, kscana) (CADR(str), CADR(set), back, size,
                                        (__CLEN_T)CLEN(str), (__CLEN_T)CLEN(set));
}

__INT_T
ENTF90(VERIFYA, verifya)
(DCHAR(str), DCHAR(set), void *back, __INT_T *size DCLEN64(str) DCLEN64(set))
{
  __INT_T i, j;

  if (ISPRESENT(back) && I8(__fort_varying_log)(back, size)) {
    for (i = (__INT_T)CLEN(str); i-- > 0;) {
      for (j = 0; j < (__INT_T)CLEN(set); ++j)
        if (CADR(set)[j] == CADR(str)[i])
          goto contb;
      return i + 1;
    contb:;
    }
    return 0;
  } else {
    for (i = 0; i < (__INT_T)CLEN(str); ++i) {
      for (j = 0; j < (__INT_T)CLEN(set); ++j)
        if (CADR(set)[j] == CADR(str)[i])
          goto contf;
      return i + 1;
    contf:;
    }
    return 0;
  }
}
/* 32 bit CLEN version */
__INT_T
ENTF90(VERIFY, verify)
(DCHAR(str), DCHAR(set), void *back, __INT_T *size DCLEN(str) DCLEN(set))
{
  return ENTF90(VERIFYA, verifya) (CADR(str), CADR(set), back, size,
                                     (__CLEN_T)CLEN(str), (__CLEN_T)CLEN(set));
}

/** \brief
 * -i8 variant of VERIFY
 */
__INT8_T
ENTF90(KVERIFYA, kverifya)
(DCHAR(str), DCHAR(set), void *back, __INT_T *size DCLEN64(str) DCLEN64(set))
{
  __INT8_T i, j;

  if (ISPRESENT(back) && I8(__fort_varying_log)(back, size)) {
    for (i = (__INT8_T)CLEN(str); i-- > 0;) {
      for (j = 0; j < (__INT8_T)CLEN(set); ++j)
        if (CADR(set)[j] == CADR(str)[i])
          goto contb;
      return i + 1;
    contb:;
    }
    return 0;
  } else {
    for (i = 0; i < (__INT8_T)CLEN(str); ++i) {
      for (j = 0; j < (__INT8_T)CLEN(set); ++j)
        if (CADR(set)[j] == CADR(str)[i])
          goto contf;
      return i + 1;
    contf:;
    }
    return 0;
  }
}
/* 32 bit CLEN version */
__INT8_T
ENTF90(KVERIFY, kverify)
(DCHAR(str), DCHAR(set), void *back, __INT_T *size DCLEN(str) DCLEN(set))
{
  return ENTF90(KVERIFYA, kverifya) (CADR(str), CADR(set), back,
                              size, (__CLEN_T)CLEN(str), (__CLEN_T)CLEN(set));
}

/** \brief
 * -i8 variant of INDEX
 */
__INT8_T
ENTF90(KINDEXA, kindexa)
(DCHAR(string), DCHAR(substring), void *back,
 __INT_T *size DCLEN64(string) DCLEN64(substring))
{
  __INT8_T i, n;

  n = (__INT8_T)CLEN(string) - (__INT8_T)CLEN(substring);
  if (n < 0)
    return 0;
  if (ISPRESENT(back) && I8(__fort_varying_log)(back, size)) {
    if (CLEN(substring) == 0)
      return (__INT8_T)CLEN(string) + 1;
    for (i = n; i >= 0; --i) {
      if (CADR(string)[i] == CADR(substring)[0] &&
          strncmp(CADR(string) + i, CADR(substring), CLEN(substring)) == 0)
        return i + 1;
    }
  } else {
    if (CLEN(substring) == 0)
      return 1;
    for (i = 0; i <= n; ++i) {
      if (CADR(string)[i] == CADR(substring)[0] &&
          strncmp(CADR(string) + i, CADR(substring), CLEN(substring)) == 0)
        return i + 1;
    }
  }
  return 0;
}
/* 32 bit CLEN version */
__INT8_T
ENTF90(KINDEX, kindex)
(DCHAR(string), DCHAR(substring), void *back,
 __INT_T *size DCLEN(string) DCLEN(substring))
{
  return ENTF90(KINDEXA, kindexa) (CADR(string), CADR(substring),
                back, size, (__CLEN_T)CLEN(string), (__CLEN_T)CLEN(substring));
}

__INT_T
ENTF90(INDEXA, indexa)
(DCHAR(string), DCHAR(substring), void *back,
 __INT_T *size DCLEN64(string) DCLEN64(substring))
{
  __INT_T i, n;

  n = (__INT_T)CLEN(string) - (__INT_T)CLEN(substring);
  if (n < 0)
    return 0;
  if (ISPRESENT(back) && I8(__fort_varying_log)(back, size)) {
    if (CLEN(substring) == 0)
      return (__INT_T)CLEN(string) + 1;
    for (i = n; i >= 0; --i) {
      if (CADR(string)[i] == CADR(substring)[0] &&
          strncmp(CADR(string) + i, CADR(substring), CLEN(substring)) == 0)
        return i + 1;
    }
  } else {
    if (CLEN(substring) == 0)
      return 1;
    for (i = 0; i <= n; ++i) {
      if (CADR(string)[i] == CADR(substring)[0] &&
          strncmp(CADR(string) + i, CADR(substring), CLEN(substring)) == 0)
        return i + 1;
    }
  }
  return 0;
}
/* 32 bit CLEN version */
__INT_T
ENTF90(INDEX, index)
(DCHAR(string), DCHAR(substring), void *back,
 __INT_T *size DCLEN(string) DCLEN(substring))
{
  return ENTF90(INDEXA, indexa) (CADR(string), CADR(substring), back,
                      size, (__CLEN_T)CLEN(string), (__CLEN_T)CLEN(substring));
}

__INT_T
ENTFTN(LEADZ, leadz)(void *i, __INT_T *size)
{
  unsigned ui; /* unsigned representation of 'i' */
  int nz;      /* number of leading zero bits in 'i' */
  int k;

  ui = (unsigned)I8(__fort_varying_int)(i, size);
  nz = *size * 8;
  k = nz >> 1;
  while (k) {
    if (ui >> k) {
      ui >>= k;
      nz -= k;
    }
    k >>= 1;
  }
  if (ui)
    --nz;
  return nz;
}

__INT_T
ENTFTN(TRAILZ, trailz)(void *i, __INT_T *size)
{
  unsigned ui= (unsigned)I8(__fort_varying_int)(i, size);

  return (ui) ? __builtin_ctz(ui): (*size * 8);
}

__INT_T
ENTFTN(POPCNT, popcnt)(void *i, __INT_T *size)
{
  unsigned ui = 0, uj; /* unsigned representation of 'i' */
  __INT8_T ll;

  switch (*size) {
  case 8:
    ll = *(__INT8_T *)i;
    ui = (unsigned)(ll & 0xFFFFFFFF);
    uj = (unsigned)((ll >> 32) & 0xFFFFFFFF);
    ui = (ui & 0x55555555) + (ui >> 1 & 0x55555555);
    uj = (uj & 0x55555555) + (uj >> 1 & 0x55555555);
    ui = (ui & 0x33333333) + (ui >> 2 & 0x33333333);
    uj = (uj & 0x33333333) + (uj >> 2 & 0x33333333);
    ui = (ui & 0x07070707) + (ui >> 4 & 0x07070707);
    ui += (uj & 0x07070707) + (uj >> 4 & 0x07070707);
    ui += ui >> 8;
    ui += ui >> 16;
    ui &= 0x7f;
    break;
  case 4:
    ui = (unsigned)(*(__INT4_T *)i);
    ui = (ui & 0x55555555) + (ui >> 1 & 0x55555555);
    ui = (ui & 0x33333333) + (ui >> 2 & 0x33333333);
    ui = (ui & 0x07070707) + (ui >> 4 & 0x07070707);
    ui += ui >> 8;
    ui += ui >> 16;
    ui &= 0x3f;
    break;
  case 2:
    ui = (unsigned)(*(__INT2_T *)i);
    ui = (ui & 0x5555) + (ui >> 1 & 0x5555);
    ui = (ui & 0x3333) + (ui >> 2 & 0x3333);
    ui = (ui & 0x0707) + (ui >> 4 & 0x0707);
    ui += ui >> 8;
    ui &= 0x1f;
    break;
  case 1:
    ui = (unsigned)(*(__INT1_T *)i);
    ui = (ui & 0x55) + (ui >> 1 & 0x55);
    ui = (ui & 0x33) + (ui >> 2 & 0x33);
    ui += ui >> 4;
    ui &= 0xf;
    break;
  default:
    __fort_abort("POPCNT: invalid size");
  }
  return ui;
}

__INT_T
ENTFTN(POPPAR, poppar)(void *i, __INT_T *size)
{
  int ii = 0;
  __INT8_T ll;

  switch (*size) {
  case 8:
    ll = *(__INT8_T *)i;
    ii = ll ^ ll >> 32;
    ii ^= ii >> 16;
    ii ^= ii >> 8;
    break;
  case 4:
    ii = *(__INT4_T *)i;
    ii ^= ii >> 16;
    ii ^= ii >> 8;
    break;
  case 2:
    ii = *(__INT2_T *)i;
    ii ^= ii >> 8;
    break;
  case 1:
    ii = *(__INT1_T *)i;
    break;
  default:
    __fort_abort("POPPAR: invalid size");
  }
  ii ^= ii >> 4;
  ii ^= ii >> 2;
  ii ^= ii >> 1;
  return ii & 1;
}

/*
 * The following functions are called when the compiler needs to invoke an
 * intrinsic which has lost its intrinsic property (i.e, the intrinsic is
 * a user-defined object).
 */

__INT4_T
ENTF90(JMAX0, jmax0)(__INT4_T *a, __INT4_T *b) { return (*a > *b) ? *a : *b; }

__INT4_T
ENTF90(MAX0, max0)(__INT4_T *a, __INT4_T *b) { return (*a > *b) ? *a : *b; }

__INT8_T
ENTF90(KMAX, kmax)(__INT8_T *a, __INT8_T *b) { return (*a > *b) ? *a : *b; }

__INT4_T
ENTF90(MIN0, min0)(__INT4_T *a, __INT4_T *b) { return (*a < *b) ? *a : *b; }

__INT4_T
ENTF90(MOD, mod)(__INT4_T *a, __INT4_T *b) { return *a % *b; }

__INT_T
ENTF90(INT, int)(void *a, __INT_T *ty)
{
  switch (*ty) {
  case __INT1:
    return *(__INT1_T *)a;
  case __LOG1:
    return *(__LOG1_T *)a;
  case __INT2:
    return *(__INT2_T *)a;
  case __LOG2:
    return *(__LOG2_T *)a;
  case __INT4:
    return *(__INT4_T *)a;
  case __LOG4:
    return *(__LOG4_T *)a;
  case __INT8:
    return *(__INT8_T *)a;
  case __LOG8:
    return *(__LOG8_T *)a;
  case __REAL4:
    return *(__REAL4_T *)a;
  case __REAL8:
    return *(__REAL8_T *)a;
  case __REAL16:
    return *(__REAL16_T *)a;
  case __CPLX8:
    return ((__CPLX8_T *)a)->r;
  case __CPLX16:
    return ((__CPLX16_T *)a)->r;
  case __CPLX32:
    return ((__CPLX32_T *)a)->r;
  default:
    __fort_abort("INT: invalid argument type");
  }
  return 0; /* sgi warning: gotta have a return! */
}

__INT1_T
ENTF90(INT1, int1)(void *a, __INT_T *ty)
{
  switch (*ty) {
  case __INT1:
    return *(__INT1_T *)a;
  case __LOG1:
    return *(__LOG1_T *)a;
  case __INT2:
    return *(__INT2_T *)a;
  case __LOG2:
    return *(__LOG2_T *)a;
  case __INT4:
    return *(__INT4_T *)a;
  case __LOG4:
    return *(__LOG4_T *)a;
  case __INT8:
    return *(__INT8_T *)a;
  case __LOG8:
    return *(__LOG8_T *)a;
  case __REAL4:
    return *(__REAL4_T *)a;
  case __REAL8:
    return *(__REAL8_T *)a;
  case __REAL16:
    return *(__REAL16_T *)a;
  case __CPLX8:
    return ((__CPLX8_T *)a)->r;
  case __CPLX16:
    return ((__CPLX16_T *)a)->r;
  case __CPLX32:
    return ((__CPLX32_T *)a)->r;
  default:
    __fort_abort("INT1: invalid argument type");
  }
  return 0; /* sgi warning: gotta have a return! */
}

__INT2_T
ENTF90(INT2, int2)(void *a, __INT_T *ty)
{
  switch (*ty) {
  case __INT1:
    return *(__INT1_T *)a;
  case __LOG1:
    return *(__LOG1_T *)a;
  case __INT2:
    return *(__INT2_T *)a;
  case __LOG2:
    return *(__LOG2_T *)a;
  case __INT4:
    return *(__INT4_T *)a;
  case __LOG4:
    return *(__LOG4_T *)a;
  case __INT8:
    return *(__INT8_T *)a;
  case __LOG8:
    return *(__LOG8_T *)a;
  case __REAL4:
    return *(__REAL4_T *)a;
  case __REAL8:
    return *(__REAL8_T *)a;
  case __REAL16:
    return *(__REAL16_T *)a;
  case __CPLX8:
    return ((__CPLX8_T *)a)->r;
  case __CPLX16:
    return ((__CPLX16_T *)a)->r;
  case __CPLX32:
    return ((__CPLX32_T *)a)->r;
  default:
    __fort_abort("INT2: invalid argument type");
  }
  return 0; /* sgi warning: gotta have a return! */
}

__INT4_T
ENTF90(INT4, int4)(void *a, __INT_T *ty)
{
  switch (*ty) {
  case __INT1:
    return *(__INT1_T *)a;
  case __LOG1:
    return *(__LOG1_T *)a;
  case __INT2:
    return *(__INT2_T *)a;
  case __LOG2:
    return *(__LOG2_T *)a;
  case __INT4:
    return *(__INT4_T *)a;
  case __LOG4:
    return *(__LOG4_T *)a;
  case __INT8:
    return *(__INT8_T *)a;
  case __LOG8:
    return *(__LOG8_T *)a;
  case __REAL4:
    return *(__REAL4_T *)a;
  case __REAL8:
    return *(__REAL8_T *)a;
  case __REAL16:
    return *(__REAL16_T *)a;
  case __CPLX8:
    return ((__CPLX8_T *)a)->r;
  case __CPLX16:
    return ((__CPLX16_T *)a)->r;
  case __CPLX32:
    return ((__CPLX32_T *)a)->r;
  default:
    __fort_abort("INT4: invalid argument type");
  }
  return 0; /* sgi warning: gotta have a return! */
}

__INT8_T
ENTF90(INT8, int8)(void *a, __INT_T *ty)
{
  switch (*ty) {
  case __INT1:
    return *(__INT1_T *)a;
  case __LOG1:
    return *(__LOG1_T *)a;
  case __INT2:
    return *(__INT2_T *)a;
  case __LOG2:
    return *(__LOG2_T *)a;
  case __INT4:
    return *(__INT4_T *)a;
  case __LOG4:
    return *(__LOG4_T *)a;
  case __INT8:
    return *(__INT8_T *)a;
  case __LOG8:
    return *(__LOG8_T *)a;
  case __REAL4:
    return *(__REAL4_T *)a;
  case __REAL8:
    return *(__REAL8_T *)a;
  case __REAL16:
    return *(__REAL16_T *)a;
  case __CPLX8:
    return ((__CPLX8_T *)a)->r;
  case __CPLX16:
    return ((__CPLX16_T *)a)->r;
  case __CPLX32:
    return ((__CPLX32_T *)a)->r;
  default:
    __fort_abort("INT8: invalid argument type");
  }
  return 0; /* sgi warning: gotta have a return! */
}

__LOG1_T
ENTF90(LOG1, log1)(void *a, __INT_T *ty)
{
  switch (*ty) {
  case __INT1:
    return *(__INT1_T *)a & GET_DIST_MASK_LOG1 ? GET_DIST_TRUE_LOG1 : 0;
  case __LOG1:
    return *(__LOG1_T *)a & GET_DIST_MASK_LOG1 ? GET_DIST_TRUE_LOG1 : 0;
  case __INT2:
    return *(__INT2_T *)a & GET_DIST_MASK_LOG2 ? GET_DIST_TRUE_LOG1 : 0;
  case __LOG2:
    return *(__LOG2_T *)a & GET_DIST_MASK_LOG2 ? GET_DIST_TRUE_LOG1 : 0;
  case __INT4:
    return *(__INT4_T *)a & GET_DIST_MASK_LOG4 ? GET_DIST_TRUE_LOG1 : 0;
  case __LOG4:
    return *(__LOG4_T *)a & GET_DIST_MASK_LOG4 ? GET_DIST_TRUE_LOG1 : 0;
  case __INT8:
    return *(__INT8_T *)a & GET_DIST_MASK_LOG8 ? GET_DIST_TRUE_LOG1 : 0;
  case __LOG8:
    return *(__LOG8_T *)a & GET_DIST_MASK_LOG8 ? GET_DIST_TRUE_LOG1 : 0;
  default:
    __fort_abort("LOG1: invalid argument type");
  }
  return 0; /* sgi warning: gotta have a return! */
}

__LOG2_T
ENTF90(LOG2, log2)(void *a, __INT_T *ty)
{
  switch (*ty) {
  case __INT1:
    return *(__INT1_T *)a & GET_DIST_MASK_LOG1 ? GET_DIST_TRUE_LOG2 : 0;
  case __LOG1:
    return *(__LOG1_T *)a & GET_DIST_MASK_LOG1 ? GET_DIST_TRUE_LOG2 : 0;
  case __INT2:
    return *(__INT2_T *)a & GET_DIST_MASK_LOG2 ? GET_DIST_TRUE_LOG2 : 0;
  case __LOG2:
    return *(__LOG2_T *)a & GET_DIST_MASK_LOG2 ? GET_DIST_TRUE_LOG2 : 0;
  case __INT4:
    return *(__INT4_T *)a & GET_DIST_MASK_LOG4 ? GET_DIST_TRUE_LOG2 : 0;
  case __LOG4:
    return *(__LOG4_T *)a & GET_DIST_MASK_LOG4 ? GET_DIST_TRUE_LOG2 : 0;
  case __INT8:
    return *(__INT8_T *)a & GET_DIST_MASK_LOG8 ? GET_DIST_TRUE_LOG2 : 0;
  case __LOG8:
    return *(__LOG8_T *)a & GET_DIST_MASK_LOG8 ? GET_DIST_TRUE_LOG2 : 0;
  default:
    __fort_abort("LOG2: invalid argument type");
  }
  return 0; /* sgi warning: gotta have a return! */
}

__LOG4_T
ENTF90(LOG4, log4)(void *a, __INT_T *ty)
{
  switch (*ty) {
  case __INT1:
    return *(__INT1_T *)a & GET_DIST_MASK_LOG1 ? GET_DIST_TRUE_LOG4 : 0;
  case __LOG1:
    return *(__LOG1_T *)a & GET_DIST_MASK_LOG1 ? GET_DIST_TRUE_LOG4 : 0;
  case __INT2:
    return *(__INT2_T *)a & GET_DIST_MASK_LOG2 ? GET_DIST_TRUE_LOG4 : 0;
  case __LOG2:
    return *(__LOG2_T *)a & GET_DIST_MASK_LOG2 ? GET_DIST_TRUE_LOG4 : 0;
  case __INT4:
    return *(__INT4_T *)a & GET_DIST_MASK_LOG4 ? GET_DIST_TRUE_LOG4 : 0;
  case __LOG4:
    return *(__LOG4_T *)a & GET_DIST_MASK_LOG4 ? GET_DIST_TRUE_LOG4 : 0;
  case __INT8:
    return *(__INT8_T *)a & GET_DIST_MASK_LOG8 ? GET_DIST_TRUE_LOG4 : 0;
  case __LOG8:
    return *(__LOG8_T *)a & GET_DIST_MASK_LOG8 ? GET_DIST_TRUE_LOG4 : 0;
  default:
    __fort_abort("LOG4: invalid argument type");
  }
  return 0; /* sgi warning: gotta have a return! */
}

__LOG8_T
ENTF90(LOG8, log8)(void *a, __INT_T *ty)
{
  switch (*ty) {
  case __INT1:
    return *(__INT1_T *)a & GET_DIST_MASK_LOG1 ? GET_DIST_TRUE_LOG8 : 0;
  case __LOG1:
    return *(__LOG1_T *)a & GET_DIST_MASK_LOG1 ? GET_DIST_TRUE_LOG8 : 0;
  case __INT2:
    return *(__INT2_T *)a & GET_DIST_MASK_LOG2 ? GET_DIST_TRUE_LOG8 : 0;
  case __LOG2:
    return *(__LOG2_T *)a & GET_DIST_MASK_LOG2 ? GET_DIST_TRUE_LOG8 : 0;
  case __INT4:
    return *(__INT4_T *)a & GET_DIST_MASK_LOG4 ? GET_DIST_TRUE_LOG8 : 0;
  case __LOG4:
    return *(__LOG4_T *)a & GET_DIST_MASK_LOG4 ? GET_DIST_TRUE_LOG8 : 0;
  case __INT8:
    return *(__INT8_T *)a & GET_DIST_MASK_LOG8 ? GET_DIST_TRUE_LOG8 : 0;
  case __LOG8:
    return *(__LOG8_T *)a & GET_DIST_MASK_LOG8 ? GET_DIST_TRUE_LOG8 : 0;
  default:
    __fort_abort("LOG8: invalid argument type");
  }
  return 0; /* sgi warning: gotta have a return! */
}

__REAL_T
ENTF90(REAL, real)(void *a, __INT_T *ty)
{
  switch (*ty) {
  case __INT1:
    return *(__INT1_T *)a;
  case __LOG1:
    return *(__LOG1_T *)a;
  case __INT2:
    return *(__INT2_T *)a;
  case __LOG2:
    return *(__LOG2_T *)a;
  case __INT4:
    return *(__INT4_T *)a;
  case __LOG4:
    return *(__LOG4_T *)a;
  case __INT8:
    return *(__INT8_T *)a;
  case __LOG8:
    return *(__LOG8_T *)a;
  case __REAL4:
    return *(__REAL4_T *)a;
  case __CPLX8:
    return ((__CPLX8_T *)a)->r;
  case __REAL8:
    return *(__REAL8_T *)a;
  case __CPLX16:
    return ((__CPLX16_T *)a)->r;
  default:
    __fort_abort("REAL: invalid argument type");
  }
  return 0.0; /* sgi warning: gotta have a return! */
}

__DBLE_T
ENTF90(DBLE, dble)(void *a, __INT_T *ty)
{
  switch (*ty) {
  case __INT1:
    return *(__INT1_T *)a;
  case __LOG1:
    return *(__LOG1_T *)a;
  case __INT2:
    return *(__INT2_T *)a;
  case __LOG2:
    return *(__LOG2_T *)a;
  case __INT4:
    return *(__INT4_T *)a;
  case __LOG4:
    return *(__LOG4_T *)a;
  case __INT8:
    return *(__INT8_T *)a;
  case __LOG8:
    return *(__LOG8_T *)a;
  case __REAL4:
    return *(__REAL4_T *)a;
  case __CPLX8:
    return ((__CPLX8_T *)a)->r;
  case __REAL8:
    return *(__REAL8_T *)a;
  case __CPLX16:
    return ((__CPLX16_T *)a)->r;
  default:
    __fort_abort("DBLE: invalid argument type");
  }
  return 0.0; /* sgi warning: gotta have a return! */
}

__REAL4_T
ENTF90(REAL4, real4)(void *a, __INT_T *ty)
{
  switch (*ty) {
  case __INT1:
    return *(__INT1_T *)a;
  case __LOG1:
    return *(__LOG1_T *)a;
  case __INT2:
    return *(__INT2_T *)a;
  case __LOG2:
    return *(__LOG2_T *)a;
  case __INT4:
    return *(__INT4_T *)a;
  case __LOG4:
    return *(__LOG4_T *)a;
  case __INT8:
    return *(__INT8_T *)a;
  case __LOG8:
    return *(__LOG8_T *)a;
  case __REAL4:
    return *(__REAL4_T *)a;
  case __CPLX8:
    return ((__CPLX8_T *)a)->r;
  case __REAL8:
    return *(__REAL8_T *)a;
  case __CPLX16:
    return ((__CPLX16_T *)a)->r;
  default:
    __fort_abort("REAL4: invalid argument type");
  }
  return 0.0; /* sgi warning: gotta have a return! */
}

__REAL8_T
ENTF90(REAL8, real8)(void *a, __INT_T *ty)
{
  switch (*ty) {
  case __INT1:
    return *(__INT1_T *)a;
  case __LOG1:
    return *(__LOG1_T *)a;
  case __INT2:
    return *(__INT2_T *)a;
  case __LOG2:
    return *(__LOG2_T *)a;
  case __INT4:
    return *(__INT4_T *)a;
  case __LOG4:
    return *(__LOG4_T *)a;
  case __INT8:
    return *(__INT8_T *)a;
  case __LOG8:
    return *(__LOG8_T *)a;
  case __REAL4:
    return *(__REAL4_T *)a;
  case __CPLX8:
    return ((__CPLX8_T *)a)->r;
  case __REAL8:
    return *(__REAL8_T *)a;
  case __CPLX16:
    return ((__CPLX16_T *)a)->r;
  default:
    __fort_abort("REAL8: invalid argument type");
  }
  return 0.0; /* sgi warning: gotta have a return! */
}

__REAL_T
ENTF90(AMAX1, amax1)(__REAL_T *a, __REAL_T *b) { return (*a > *b) ? *a : *b; }

__DBLE_T
ENTF90(DMAX1, dmax1)(__DBLE_T *a, __DBLE_T *b) { return (*a > *b) ? *a : *b; }

__REAL_T
ENTF90(AMIN1, amin1)(__REAL_T *a, __REAL_T *b) { return (*a < *b) ? *a : *b; }

__DBLE_T
ENTF90(DMIN1, dmin1)(__DBLE_T *a, __DBLE_T *b) { return (*a < *b) ? *a : *b; }

__REAL_T
ENTF90(AMOD, amod)(__REAL_T *a, __REAL_T *b) { return fmod(*a, *b); }

__DBLE_T
ENTF90(DMOD, dmod)(__DBLE_T *a, __DBLE_T *b)
{
  return fmod(*a, *b);
}

__INT4_T
ENTF90(MODULO, modulo)(__INT4_T *a, __INT4_T *p)
{
  __INT4_T q, r;

  q = (*a) / (*p);
  r = (*a) - q * (*p);
  if (r != 0 && ((*a) ^ (*p)) < 0) { /* signs differ */
    r += (*p);
  }
  return r;
}

__INT8_T
ENTF90(I8MODULO, i8modulo)(__INT8_T *a, __INT8_T *p)
{
  __INT8_T q, r;

  q = (*a) / (*p);
  r = (*a) - q * (*p);
  if (r != 0 && ((*a) ^ (*p)) < 0) { /* signs differ */
    r += (*p);
  }
  return r;
}

__INT2_T
ENTF90(IMODULO, imodulo)(__INT2_T *a, __INT2_T *p)
{
  __INT_T q, r;

  q = (*a) / (*p);
  r = (*a) - q * (*p);
  if (r != 0 && ((*a) ^ (*p)) < 0) { /* signs differ */
    r += (*p);
  }
  return r;
}

__REAL_T
ENTF90(AMODULO, amodulo)(__REAL_T *x, __REAL_T *y)
{
  double d;
  d = fmod(*x, *y);
  if (d != 0 && ((*x < 0 && *y > 0) || (*x > 0 && *y < 0)))
    d += *y;
  return d;
}

__DBLE_T
ENTF90(DMODULO, dmodulo)(__DBLE_T *x, __DBLE_T *y)
{
  double d;
  d = fmod(*x, *y);
  if (d != 0 && ((*x < 0 && *y > 0) || (*x > 0 && *y < 0)))
    d += *y;
  return d;
}

#ifdef TARGET_SUPPORTS_QUADFP
__QUAD_T
ENTF90(QMODULO, qmodulo)(__QUAD_T *x, __QUAD_T *y)
{
  __QUAD_T q;
  q = (__QUAD_T)fmodl(*x, *y);
  if (q != 0 && ((*x < 0 && *y > 0) || (*x > 0 && *y < 0)))
    q += *y;
  return q;
}
#endif

__INT4_T
ENTF90(MODULOv, modulov)(__INT4_T a, __INT4_T p)
{
  __INT4_T q, r;

  q = (a) / (p);
  r = (a)-q * (p);
  if (r != 0 && ((a) ^ (p)) < 0) { /* signs differ */
    r += (p);
  }
  return r;
}

__INT8_T
ENTF90(I8MODULOv, i8modulov)(__INT8_T a, __INT8_T p)
{
  __INT8_T q, r;

  q = (a) / (p);
  r = (a)-q * (p);
  if (r != 0 && ((a) ^ (p)) < 0) { /* signs differ */
    r += (p);
  }
  return r;
}

__INT2_T
ENTF90(IMODULOv, imodulov)(__INT2_T a, __INT2_T p)
{
  __INT_T q, r;

  q = (a) / (p);
  r = (a)-q * (p);
  if (r != 0 && ((a) ^ (p)) < 0) { /* signs differ */
    r += (p);
  }
  return r;
}

__REAL_T
ENTF90(AMODULOv, amodulov)(__REAL_T x, __REAL_T y)
{
  double d;
  d = fmod(x, y);
  if (d != 0 && ((x < 0 && y > 0) || (x > 0 && y < 0)))
    d += y;
  return d;
}

__DBLE_T
ENTF90(DMODULOv, dmodulov)(__DBLE_T x, __DBLE_T y)
{
  double d;
  d = fmod(x, y);
  if (d != 0 && ((x < 0 && y > 0) || (x > 0 && y < 0)))
    d += y;
  return d;
}

#ifdef TARGET_SUPPORTS_QUADFP
__QUAD_T
ENTF90(QMODULOv, qmodulov)(__QUAD_T x, __QUAD_T y)
{
  __QUAD_T q = (__QUAD_T)fmodl(x, y);
  if (q != 0 && ((x < 0 && y > 0) || (x > 0 && y < 0)))
    q += y;
  return q;
}
#endif

__INT_T
ENTF90(CEILING, ceiling)(__REAL_T *r)
{
  __DBLE_T x;
  int a;

  x = *r;
  a = x; /* integer part */
  if (a == x)
    return x;
  if (x > 0)
    return a + 1;
  return a;
}

__INT8_T
ENTF90(KCEILING, kceiling)(__REAL_T *r)
{
  __DBLE_T x;
  __INT8_T a;

  x = *r;
  a = x; /* integer part */
  if (a == x)
    return x;
  if (x > 0)
    return a + 1;
  return a;
}

__INT_T
ENTF90(DCEILING, dceiling)(__DBLE_T *r)
{
  __DBLE_T x;
  int a;

  x = *r;
  a = x; /* integer part */
  if (a == x)
    return x;
  if (x > 0)
    return a + 1;
  return a;
}

__INT8_T
ENTF90(KDCEILING, kdceiling)(__DBLE_T *r)
{
  __DBLE_T x;
  __INT8_T a;

  x = *r;
  a = x; /* integer part */
  if (a == x)
    return x;
  if (x > 0)
    return a + 1;
  return a;
}

__INT_T
ENTF90(CEILINGv, ceilingv)(__REAL_T r)
{
  __DBLE_T x;
  int a;

  x = r;
  a = x; /* integer part */
  if (a == x)
    return x;
  if (x > 0)
    return a + 1;
  return a;
}

__INT8_T
ENTF90(KCEILINGv, kceilingv)(__REAL_T r)
{
  __DBLE_T x;
  __INT8_T a;

  x = r;
  a = x; /* integer part */
  if (a == x)
    return x;
  if (x > 0)
    return a + 1;
  return a;
}

__INT_T
ENTF90(DCEILINGv, dceilingv)(__DBLE_T r)
{
  __DBLE_T x;
  int a;

  x = r;
  a = x; /* integer part */
  if (a == x)
    return x;
  if (x > 0)
    return a + 1;
  return a;
}

__INT8_T
ENTF90(KDCEILINGv, kdceilingv)(__DBLE_T r)
{
  __DBLE_T x;
  __INT8_T a;

  x = r;
  a = x; /* integer part */
  if (a == x)
    return x;
  if (x > 0)
    return a + 1;
  return a;
}

__INT_T
ENTF90(FLOOR, floor)(__REAL_T *r)
{
  __DBLE_T x;
  int a;

  x = *r;
  a = x; /* integer part */
  if (a == x)
    return x;
  if (x < 0)
    return a - 1;
  return a;
}

__INT_T
ENTF90(KFLOOR, kfloor)(__REAL_T *r)
{
  __DBLE_T x;
  __INT8_T a;

  x = *r;
  a = x; /* integer part */
  if (a == x)
    return x;
  if (x < 0)
    return a - 1;
  return a;
}

__INT_T
ENTF90(DFLOOR, dfloor)(__DBLE_T *r)
{
  __DBLE_T x;
  int a;

  x = *r;
  a = x; /* integer part */
  if (a == x)
    return x;
  if (x < 0)
    return a - 1;
  return a;
}

__INT8_T
ENTF90(KDFLOOR, kdfloor)(__DBLE_T *r)
{
  __DBLE_T x;
  __INT8_T a;

  x = *r;
  a = x; /* integer part */
  if (a == x)
    return x;
  if (x < 0)
    return a - 1;
  return a;
}

__INT_T
ENTF90(FLOORv, floorv)(__REAL_T r)
{
  __DBLE_T x;
  int a;

  x = r;
  a = x; /* integer part */
  if (a == x)
    return x;
  if (x < 0)
    return a - 1;
  return a;
}

__INT_T
ENTF90(KFLOORv, kfloorv)(__REAL_T r)
{
  __DBLE_T x;
  __INT8_T a;

  x = r;
  a = x; /* integer part */
  if (a == x)
    return x;
  if (x < 0)
    return a - 1;
  return a;
}

__INT_T
ENTF90(DFLOORv, dfloorv)(__DBLE_T r)
{
  __DBLE_T x;
  int a;

  x = r;
  a = x; /* integer part */
  if (a == x)
    return x;
  if (x < 0)
    return a - 1;
  return a;
}

__INT8_T
ENTF90(KDFLOORv, kdfloorv)(__DBLE_T r)
{
  __DBLE_T x;
  __INT8_T a;

  x = r;
  a = x; /* integer part */
  if (a == x)
    return x;
  if (x < 0)
    return a - 1;
  return a;
}

/** \brief selected_int_kind(r) */
__INT_T
ENTF90(SEL_INT_KIND, sel_int_kind)
(char *rb, F90_Desc *rd)
{
  int r;

  r = I8(__fort_fetch_int)(rb, rd);
  if (r <= 2)
    return 1;
  if (r <= 4)
    return 2;
  if (r <= 9)
    return 4;
  if (r <= 18)
    return 8;
  return -1;
}

/** \brief
 * -i8 variant of SEL_INT_KIND
 */
__INT8_T
ENTF90(KSEL_INT_KIND, ksel_int_kind)
(char *rb, F90_Desc *rd)
{
  int r;

  r = I8(__fort_fetch_int)(rb, rd);
  if (r <= 2)
    return 1;
  if (r <= 4)
    return 2;
  if (r <= 9)
    return 4;
  if (r <= 18)
    return 8;
  return -1;
}

/* selected_char_kind(r) */

/** \brief Check charset
 *
 * Make sure this routine is consistent with
 * - fe90: semfunc.c:_selected_char_kind()
 * - f90:  dinit.c:_selected_char_kind()
 */
static int
_selected_char_kind(char *p, __CLEN_T len)
{
  if (__fortio_eq_str(p, len, "ASCII"))
    return 1;
  else if (__fortio_eq_str(p, len, "DEFAULT"))
    return 1;
  return -1;
}

__INT_T
ENTF90(SEL_CHAR_KINDA, sel_char_kinda)
(DCHAR(p), F90_Desc *rd DCLEN64(p))
{
  int r;
  r = _selected_char_kind(CADR(p), CLEN(p));
  return r;
}
/* 32 bit CLEN version */
__INT_T
ENTF90(SEL_CHAR_KIND, sel_char_kind)
(DCHAR(p), F90_Desc *rd DCLEN(p))
{
  return ENTF90(SEL_CHAR_KINDA, sel_char_kinda) (CADR(p), rd, (__CLEN_T)CLEN(p));
}

__INT8_T
ENTF90(KSEL_CHAR_KINDA, ksel_char_kinda)
(DCHAR(p), F90_Desc *rd DCLEN(p))
{
  int r;
  r = _selected_char_kind(CADR(p), CLEN(p));
  return r;
}
/* 32 bit CLEN version */
__INT8_T
ENTF90(KSEL_CHAR_KIND, ksel_char_kind)
(DCHAR(p), F90_Desc *rd DCLEN(p))
{
  return ENTF90(KSEL_CHAR_KINDA, ksel_char_kinda) (CADR(p), rd, (__CLEN_T)CLEN(p));
}

/* Real model support functions */

/* IEEE floating point - 4 byte reals, 8 byte double precision */

/** \brief selected_real_kind(p,r,radix),add radix argument for f2008 */
__INT8_T
ENTF90(KSEL_REAL_KIND, ksel_real_kind)
(char *pb, char *rb, char *radixb, F90_Desc *pd, F90_Desc *rd, F90_Desc *radixd)
{
  /* 
   * -i8 variant of SEL_REAL_KIND
   */

  int p, r, e, k;

  e = 0;
  k = 0;
  if (ISPRESENT(pb)) {
    p = I8(__fort_fetch_int)(pb, pd);
    if (p <= 6)
      k = 4;
    else if (p <= 15)
      k = 8;
#ifdef TARGET_SUPPORTS_QUADFP
    else if (p <= 33)
      k = 16;
#endif
    else
      e -= 1;
  }
  if (ISPRESENT(rb)) {
    r = I8(__fort_fetch_int)(rb, rd);
    if (r <= 37) {
      if (k < 4)
        k = 4;
    }
    else if (r <= 307) {
      if (k < 8)
        k = 8;
    }
#ifdef TARGET_SUPPORTS_QUADFP
    else if (r <= 4931) {
      if (k < 16)
        k = 16;
    }
#endif
    else
      e -= 2;
  }
  if (ISPRESENT(radixb)) {
    int radix = I8(__fort_fetch_int)(radixb, radixd);
    if (radix != 2)
      e = -5;
    else
      k = k ? k : 4;
  }
  return (__INT8_T)e ? e : k;
}

__INT_T
ENTF90(SEL_REAL_KIND, sel_real_kind)
(char *pb, char *rb, char *radixb, F90_Desc *pd, F90_Desc *rd, F90_Desc *radixd)
{
  int p, r, e, k;

  e = 0;
  k = 0;
  if (ISPRESENT(pb)) {
    p = I8(__fort_fetch_int)(pb, pd);
    if (p <= 6)
      k = 4;
    else if (p <= 15)
      k = 8;
#ifdef TARGET_SUPPORTS_QUADFP
    else if (p <= 33)
      k = 16;
#endif
    else
      e -= 1;
  }
  if (ISPRESENT(rb)) {
    r = I8(__fort_fetch_int)(rb, rd);
    if (r <= 37) {
      if (k < 4)
        k = 4;
    }
    else if (r <= 307) {
      if (k < 8)
        k = 8;
    }
#ifdef TARGET_SUPPORTS_QUADFP
    else if (r <= 4931) {
      if (k < 16)
        k = 16;
    }
#endif
    else
      e -= 2;
  }
  if (ISPRESENT(radixb)) {
    int radix = I8(__fort_fetch_int)(radixb, radixd);
    if (radix != 2)
      e = -5;
    else
      k = k ? k : 4;
  }
  return e ? e : k;
}

__INT_T
ENTF90(EXPONX, exponx)(__REAL4_T f)
{
  union {
    __INT4_T i;
    __REAL4_T r;
  } g;
  g.r = f;
  if ((g.i & ~0x80000000) == 0)
    return 0;
  else
    return ((g.i >> 23) & 0xFF) - 126;
}

__INT_T
ENTF90(EXPON, expon)(__REAL4_T *f)
{
  return ENTF90(EXPONX, exponx)(*f);
}

__INT_T
ENTF90(EXPONDX, expondx)(__REAL8_T d)
{
  union {
    __INT8_T i;
    __REAL8_T r;
  } g;
  g.r = d;
  if ((((g.i >> 32) & ~0x80000000) | (g.i & 0xffffffff)) == 0)
    return 0;
  else
    return ((g.i >> 52) & 0x7FF) - 1022;
}

__INT_T
ENTF90(EXPOND, expond)(__REAL8_T *d)
{
  return ENTF90(EXPONDX, expondx)(*d);
}

#ifdef TARGET_SUPPORTS_QUADFP
__INT_T
ENTF90(EXPONQX, exponqx)(__REAL16_T q)
{
  __REAL16_SPLIT g;
  g.q = q;
  if (((g.i.h & 0x7fffffff) | (g.i.j & 0xffffffff) |
       (g.i.k & 0xffffffff) | (g.i.l & 0xffffffff)) == 0)
    return 0;
  else
    return ((g.ll.h >> 48) & 0x7FFF) - 16382;
}

__INT_T
ENTF90(EXPONQ, exponq)(const __REAL16_T *q)
{
   return ENTF90(EXPONQX, exponqx)(*q);
}
#endif

__INT8_T
ENTF90(KEXPONX, kexponx)(__REAL4_T f)
{
  return ENTF90(EXPONX, exponx)(f);
}

__INT8_T
ENTF90(KEXPON, kexpon)(__REAL4_T *f)
{
  return ENTF90(KEXPONX, kexponx)(*f);
}

__INT8_T
ENTF90(KEXPONDX, kexpondx)(__REAL8_T d)
{
  return ENTF90(EXPONDX, expondx)(d);
}

__INT8_T
ENTF90(KEXPOND, kexpond)(__REAL8_T *d) {
  return ENTF90(KEXPONDX, kexpondx)(*d);
}

__REAL4_T
ENTF90(FRACX, fracx)(__REAL4_T f)
{
  union {
    __REAL4_T f;
    __INT4_T i;
  } x;

  x.f = f;
  if (x.f != 0.0) {
    x.i &= ~0x7F800000;
    x.i |= 0x3F000000;
  }
  return x.f;
}

__REAL4_T
ENTF90(FRAC, frac)(__REAL4_T *f) { return ENTF90(FRACX, fracx)(*f); }

__REAL8_T
ENTF90(FRACDX, fracdx)(__REAL8_T d)
{
  __REAL8_SPLIT x;

  x.d = d;
  if (x.d != 0.0) {
    x.i.h &= ~0x7FF00000;
    x.i.h |= 0x3FE00000;
  }
  return x.d;
}

__REAL8_T
ENTF90(FRACD, fracd)(__REAL8_T *d) { return ENTF90(FRACDX, fracdx)(*d); }

#ifdef TARGET_SUPPORTS_QUADFP
__REAL16_T
ENTF90(FRACQX, fracqx)(__REAL16_T q)
{
  __REAL16_SPLIT x;

  x.q = q;
  if (x.q != 0.0) {
    x.i.h &= ~0x7FFF0000;
    x.i.h |= 0x3FFE0000;
  }
  return x.q;
}

__REAL16_T
ENTF90(FRACQ, fracq)(__REAL16_T *q)
{
  return ENTF90(FRACQX, fracqx)(*q);
}
#endif

/** \brief NEAREST(X,S) has a value equal to the machine representable number
 * distinct from X and nearest to it in the direction of the infinity
 * with the same sign as S.
 *
 * The 'sign' argument is equal to the
 * fortran logical expression (S .ge. 0.0). */
__REAL4_T
ENTF90(NEARESTX, nearestx)(__REAL4_T f, __LOG_T sign)
{
  union {
    __REAL4_T f;
    __INT4_T i;
  } x;

  x.f = f;
  if (x.f == 0.0) {
    x.i = (sign & GET_DIST_MASK_LOG) ? 0x00800000 : 0x80800000;
  } else if ((x.i & 0x7F800000) != 0x7F800000) { /* not nan or inf */
    if ((x.f < 0.0) ^ (sign & GET_DIST_MASK_LOG))
      ++x.i;
    else
      --x.i;
  } else if (!(sign & GET_DIST_MASK_LOG) && (x.i == 0x7F800000)) {
    x.i = 0x7F7FFFFF;
  } else if ((sign & GET_DIST_MASK_LOG) && (x.i == 0xFF800000)) {
    x.i = 0xFF7FFFFF;
  }
  return x.f;
}

__REAL4_T
ENTF90(NEAREST, nearest)(__REAL4_T *f, __LOG_T *sign)
{
  return ENTF90(NEARESTX, nearestx)(*f, *sign);
}

__REAL8_T
ENTF90(NEARESTDX, nearestdx)(__REAL8_T d, __LOG_T sign)
{
  __REAL8_SPLIT x;

  x.d = d;
  if (x.d == 0.0) {
    x.i.h = (sign & 1) ? 0x00100000 : 0x80100000;
    x.i.l = 0;
  } else if ((x.ll >> 52 & 0x7FF) != 0x7FF) { /* not nan or inf */
      if ((x.d < 0) ^ (sign & GET_DIST_MASK_LOG))
        ++x.ll;
      else
        --x.ll;
  } else if (!(sign & GET_DIST_MASK_LOG) && (x.ll == 0x7FF0000000000000)) {
    x.ll = 0x7FEFFFFFFFFFFFFF;
  } else if ((sign & GET_DIST_MASK_LOG) && (x.ll == 0xFFF0000000000000)) {
    x.ll = 0xFFEFFFFFFFFFFFFF;
  }
  return x.d;
}

__REAL8_T
ENTF90(NEARESTD, nearestd)(__REAL8_T *d, __LOG_T *sign)
{
  return ENTF90(NEARESTDX, nearestdx)(*d, *sign);
}

#ifdef TARGET_SUPPORTS_QUADFP
__REAL16_T
ENTF90(NEARESTQX, nearestqx)(__REAL16_T q, __LOG_T sign)
{
  __REAL16_SPLIT x;

  x.q = q;
  if (x.q == 0.0) {
    x.i.h = (sign & 1) ? 0x00000000 : 0x80000000;
    x.i.j = x.i.k = 0;
    x.i.l = 0x00000001;
  } else {
    if ((x.ll.h >> 48 & 0x7FFF) != 0x7FFF) { /* not nan or inf */
      if ((int)(x.q < 0.0) ^ (int)(sign & GET_DIST_MASK_LOG)) {
        if ((x.ll.l + 1) < x.ll.l) {
          ++x.ll.h;
          x.ll.l = 0;
        } else {
          ++x.ll.l;
        }
      } else {
        if (x.ll.l == 0) {
          --x.ll.h;
          --x.ll.l;
        } else {
          --x.ll.l;
        }
      }
    }
  }
  return x.q;
}

__REAL16_T
ENTF90(NEARESTQ, nearestq)(const __REAL16_T *q, const __LOG_T *sign)
{
  return ENTF90(NEARESTQX, nearestqx)(*q, *sign);
}
#endif

__REAL4_T
ENTF90(RRSPACINGX, rrspacingx)(__REAL4_T f)
{
  union {
    __REAL4_T f;
    __INT4_T i;
  } x, y;

  x.f = f;
  if (x.f == 0)
    return 0;
  y.i = (x.i & 0xFF << 23) ^ 0xFF << 23;
  x.f *= y.f;
  if (x.f < 0)
    x.f = -x.f;
  y.i = (22 + 127) << 23;
  x.f *= y.f;
  return x.f;
}

__REAL4_T
ENTF90(RRSPACING, rrspacing)(__REAL4_T *f)
{
  return ENTF90(RRSPACINGX, rrspacingx)(*f);
}

__REAL8_T
ENTF90(RRSPACINGDX, rrspacingdx)(__REAL8_T d)
{
  __REAL8_SPLIT x, y;

  x.d = d;
  if (x.d == 0)
    return 0;
  y.i.h = (x.i.h & 0x7FF << 20) ^ 0x7FF << 20;
  y.i.l = 0;
  x.d *= y.d;
  if (x.d < 0)
    x.d = -x.d;
  y.i.h = (51 + 1023) << 20;
  y.i.l = 0;
  x.d *= y.d;
  return x.d;
}

__REAL8_T
ENTF90(RRSPACINGD, rrspacingd)(__REAL8_T *d)
{
  return ENTF90(RRSPACINGDX, rrspacingdx)(*d);
}

#ifdef TARGET_SUPPORTS_QUADFP
__REAL16_T
ENTF90(RRSPACINGQX, rrspacingqx)(__REAL16_T q)
{
  __REAL16_SPLIT x, y;

  x.q = q;
  if (x.q == 0)
    return 0;
  y.i.h = (x.i.h & 0x7FFF << 16) ^ 0x7FFF << 16;
  y.i.j = 0;
  y.i.k = 0;
  y.i.l = 0;
  x.q *= y.q;
  if (x.q < 0)
    x.q = -x.q;
  y.i.h = (111 + 16383) << 16;
  y.i.k = 0;
  y.i.l = 0;
  y.i.l = 0;
  x.q *= y.q;
  return x.q;
}

__REAL16_T
ENTF90(RRSPACINGQ, rrspacingq)(__REAL16_T *q)
{
  return ENTF90(RRSPACINGQX, rrspacingqx)(*q);
}
#endif

__REAL4_T
ENTF90(SCALEX, scalex)(__REAL4_T f, __INT_T i)
{
  int e;
  union {
    __REAL4_T f;
    __INT4_T i;
  } x;

  e = 127 + i;
  if (e < 0)
    e = 0;
  else if (e > 255)
    e = 255;
  x.i = e << 23;
  return f * x.f;
}

__REAL4_T
ENTF90(SCALE, scale)(__REAL4_T *f, void *i, __INT_T *size)
{
  int e;
  union {
    __REAL4_T f;
    __INT4_T i;
  } x;

  e = 127 + I8(__fort_varying_int)(i, size);
  if (e < 0)
    e = 0;
  else if (e > 255)
    e = 255;
  x.i = e << 23;
  return *f * x.f;
}

__REAL8_T
ENTF90(SCALEDX, scaledx)(__REAL8_T d, __INT_T i)
{
  int e;
  __REAL8_SPLIT x;

  e = 1023 + i;
  if (e < 0)
    e = 0;
  else if (e > 2047)
    e = 2047;
  x.i.h = e << 20;
  x.i.l = 0;
  return d * x.d;
}

__REAL8_T
ENTF90(SCALED, scaled)(__REAL8_T *d, void *i, __INT_T *size)
{
  int e;
  __REAL8_SPLIT x;

  e = 1023 + I8(__fort_varying_int)(i, size);
  if (e < 0)
    e = 0;
  else if (e > 2047)
    e = 2047;
  x.i.h = e << 20;
  x.i.l = 0;
  return *d * x.d;
}

#ifdef TARGET_SUPPORTS_QUADFP
__REAL16_T
ENTF90(SCALEQX, scaleqx)(__REAL16_T q, __INT_T i)
{
  int e;
  __REAL16_SPLIT x;

  e = 16383 + i;
  if (e < 0)
    e = 0;
  else if (e > 32767)
    e = 32767;
  x.i.h = e << 16;
  x.i.j = 0;
  x.i.k = 0;
  x.i.l = 0;
  return q * x.q;
}

__REAL16_T
ENTF90(SCALEQ, scaleq)(__REAL16_T *q, void *i, __INT_T *size)
{
  int e;
  __REAL16_SPLIT x;

  e = 16383 + I8(__fort_varying_int)(i, size);
  if (e < 0)
    e = 0;
  else if (e > 32767)
    e = 32767;
  x.i.h = e << 16;
  x.i.j = 0;
  x.i.k = 0;
  x.i.l = 0;
  return *q * x.q;
}
#endif

__REAL4_T
ENTF90(SETEXPX, setexpx)(__REAL4_T f, __INT_T i)
{
  int e;
  union {
    __REAL4_T f;
    __INT4_T i;
  } x, y;

  y.f = f;
  if (y.f == 0.0)
    return y.f;
  if ((y.i & 0x7FFFFFFF) == 0x7F800000) {
    y.i = 0x7FC00000;
    return y.f;
  }
  y.i &= ~0x7F800000;
  y.i |= 0x3F800000;
  e = 126 + i;
  if (e < 0)
    e = 0;
  else if (e > 255)
    e = 255;
  x.i = e << 23;
  return x.f * y.f;
}

__REAL4_T
ENTF90(SETEXP, setexp)(__REAL4_T *f, void *i, __INT_T *size)
{
  int e;
  union {
    __REAL4_T f;
    __INT4_T i;
  } x, y;

  y.f = *f;
  if (y.f == 0.0)
    return y.f;
  if ((y.i & 0x7FFFFFFF) == 0x7F800000) {
    y.i = 0x7FC00000;
    return y.f;
  }
  y.i &= ~0x7F800000;
  y.i |= 0x3F800000;
  e = 126 + I8(__fort_varying_int)(i, size);
  if (e < 0)
    e = 0;
  else if (e > 255)
    e = 255;
  x.i = e << 23;
  return x.f * y.f;
}

__REAL8_T
ENTF90(SETEXPDX, setexpdx)(__REAL8_T d, __INT_T i)
{
  int e;
  __REAL8_SPLIT x, y;

  y.d = d;
  if (y.d == 0.0)
    return y.d;
  if ((y.ll & 0x7FFFFFFFFFFFFFFF) == 0x7FF0000000000000){
    y.ll = 0x7FF8000000000000;
    return y.d;
  }
  y.i.h &= ~0x7FF00000;
  y.i.h |= 0x3FF00000;
  e = 1022 + i;
  if (e < 0)
    e = 0;
  else if (e > 2047)
    e = 2047;
  x.i.h = e << 20;
  x.i.l = 0;
  return x.d * y.d;
}

__REAL8_T
ENTF90(SETEXPD, setexpd)(__REAL8_T *d, void *i, __INT_T *size)
{
  int e;
  __REAL8_SPLIT x, y;

  y.d = *d;
  if (y.d == 0.0)
    return y.d;
  if ((y.ll & 0x7FFFFFFFFFFFFFFF) == 0x7FF0000000000000) {
    y.ll = 0x7FF8000000000000;
    return y.d;
  }
  y.i.h &= ~0x7FF00000;
  y.i.h |= 0x3FF00000;
  e = 1022 + I8(__fort_varying_int)(i, size);
  if (e < 0)
    e = 0;
  else if (e > 2047)
    e = 2047;
  x.i.h = e << 20;
  x.i.l = 0;
  return x.d * y.d;
}

#ifdef TARGET_SUPPORTS_QUADFP
__REAL16_T
ENTF90(SETEXPQX, setexpqx)(__REAL16_T q, __INT_T i)
{
  int e;
  __REAL16_SPLIT x, y;

  y.q = q;
  if (y.q == 0.0)
    return y.q;
  if ((y.ll.h & 0x7fffffffffffffff) == 0x7fff000000000000 &&
          (y.ll.l & ~0x0) == 0x0) {
    y.ll.h = 0x7FFF800000000000;
    return y.q;
  }
  y.i.h &= ~0x7FFF0000;
  y.i.h |= 0x3FFF0000;
  e = 16382 + i;
  if (e < 0)
    e = 0;
  else if (e > 32767)
    e = 32767;
  x.i.h = e << 16;
  x.i.j = 0;
  x.i.k = 0;
  x.i.l = 0;
  return x.q * y.q;
}

__REAL16_T
ENTF90(SETEXPQ, setexpq)(__REAL16_T q, void *i, __INT_T *size)
{
  int e;
  __REAL16_SPLIT x, y;

  y.q = q;
  if (y.q == 0.0)
    return y.q;
  if ((y.ll.h & 0x7fffffffffffffff) == 0x7fff000000000000 &&
          (y.ll.l & ~0x0) == 0x0) {
    y.ll.h = 0x7fff800000000000;
    return y.q;
  }
  y.i.h &= ~0x7FFF0000;
  y.i.h |= 0x3FFF0000;
  e = 16382 + I8(__fort_varying_int)(i, size);
  if (e < 0)
    e = 0;
  else if (e > 32767)
    e = 32767;
  x.i.h = e << 16;
  x.i.j = 0;
  x.i.k = 0;
  x.i.l = 0;
  return x.q * y.q;
}
#endif

__REAL4_T
ENTF90(SPACINGX, spacingx)(__REAL4_T f)
{
  int e;
  union {
    __REAL4_T f;
    __INT4_T i;
  } x;

  x.f = f;
  e = ((x.i >> 23) & 0xFF) - 23;
  if (e < 1)
    e = 1;
  x.i = e << 23;
  return x.f;
}

__REAL4_T
ENTF90(SPACING, spacing)(__REAL4_T *f)
{
  return ENTF90(SPACINGX, spacingx)(*f);
}

__REAL8_T
ENTF90(SPACINGDX, spacingdx)(__REAL8_T d)
{
  int e;
  __REAL8_SPLIT x;

  x.d = d;
  e = ((x.i.h >> 20) & 0x7FF) - 52;
  if (e < 1)
    e = 1;
  x.i.h = e << 20;
  x.i.l = 0;
  return x.d;
}

__REAL8_T
ENTF90(SPACINGD, spacingd)(__REAL8_T *d)
{
  return ENTF90(SPACINGDX, spacingdx)(*d);
}


#ifdef TARGET_SUPPORTS_QUADFP
__REAL16_T
ENTF90(SPACINGQX, spacingqx)(__REAL16_T q)
{
  int e;
  __REAL16_SPLIT x;

  x.q = q;
  e = ((x.i.h >> 16) & 0x7FFF) - 112;
  if (e < 1)
    e = 1;
  x.i.h = e << 16;
  x.i.l = x.i.j = x.i.k = 0;
  return x.q;
}

__REAL16_T
ENTF90(SPACINGQ, spacingq)(const __REAL16_T *q)
{
  return ENTF90(SPACINGQX, spacingqx)(*q);
}
#endif

#ifndef DESC_I8

typedef __INT8_T SZ_T;

#undef _MZERO
#define _MZERO(n, t)                                                    \
void									\
ENTF90(MZEROERO##n, mzero##n)(void *d, SZ_T size)                       \
{                                                                       \
  if (d && size > 0)                                                    \
    __c_mzero##n(d, size);                                              \
}

_MZERO(1, char)

_MZERO(2, short)

_MZERO(4, int)

_MZERO(8, long long)

#ifdef TARGET_SUPPORTS_QUADFP
_MZERO(16, long long)
#endif

void
ENTF90(MZEROZ8, mzeroz8)(void *d, SZ_T size)
{
  if (d && size > 0) {
    __c_mzero4(d, size * 2);
  }
}

void
ENTF90(MZEROZ16, mzeroz16)(void *d, SZ_T size)
{
  if (d && size > 0) {
    __c_mzero8(d, size * 2);
  }
}

#ifdef TARGET_SUPPORTS_QUADFP
void
ENTF90(MZEROZ32, mzeroz32)(void *d, SZ_T size)
{
  if (d && size > 0) {
    __c_mzero16(d, size * 2);
  }
}
#endif

#undef _MSET
#define _MSET(n, t)                                                            \
  void ENTF90(MSET##n, mset##n)(void *d, void *v, SZ_T size)                   \
  {                                                                            \
    if (d && size > 0)                                                         \
      __c_mset##n(d, *((t *)v), size);                                         \
  }

_MSET(1, char)

_MSET(2, short)

_MSET(4, int)

_MSET(8, long long)

#ifdef TARGET_SUPPORTS_QUADFP
_MSET(16, float128_t)
#endif

void
ENTF90(MSETZ8, msetz8)(void *d, void *v, SZ_T size)
{
  if (d) {
    SZ_T i;
    int *pd;
    int v0, v1;
    pd = (int *)d;
    v0 = ((int *)v)[0];
    v1 = ((int *)v)[1];
    for (i = 0; i < size; i++) {
      pd[0] = v0;
      pd[1] = v1;
      pd += 2;
    }
  }
}

void
ENTF90(MSETZ16, msetz16)(void *d, void *v, SZ_T size)
{
  if (d) {
    SZ_T i;
    long long *pd;
    long long v0, v1;
    pd = (long long *)d;
    v0 = ((long long *)v)[0];
    v1 = ((long long *)v)[1];
    for (i = 0; i < size; i++) {
      pd[0] = v0;
      pd[1] = v1;
      pd += 2;
    }
  }
}

void
ENTF90(MSETZ32, msetz32)(void *d, void *v, SZ_T size)
{
  if (d) {
    SZ_T i;
    float128_t *pd;
    float128_t v0, v1;
    pd = (float128_t *)d;
    v0 = ((float128_t *)v)[0];
    v1 = ((float128_t *)v)[1];
    for (i = 0; i < size; i++) {
      pd[0] = v0;
      pd[1] = v1;
      pd += 2;
    }
  }
}

#undef _MCOPY
#define _MCOPY(n, t)                                                           \
  void ENTF90(MCOPY##n, mcopy##n)(void *d, void *v, SZ_T size)                 \
  {                                                                            \
    if (d && v && size > 0)                                                    \
      __c_mcopy##n(d, v, size);                                                \
  }

_MCOPY(1, char)

_MCOPY(2, short)

_MCOPY(4, int)

_MCOPY(8, long long)

#ifdef TARGET_SUPPORTS_QUADFP
_MCOPY(16, long double)
#endif

void
ENTF90(MCOPYZ8, mcopyz8)(void *d, void *v, SZ_T size)
{
  if (d && v && size) {
    __c_mcopy4(d, v, size * 2);
  }
}

void
ENTF90(MCOPYZ16, mcopyz16)(void *d, void *v, SZ_T size)
{
  if (d && v && size) {
    __c_mcopy8(d, v, size * 2);
  }
}

#ifdef TARGET_SUPPORTS_QUADFP
void
ENTF90(MCOPYZ32, mcopyz32)(void *d, void *v, SZ_T size)
{
  if (d && v && size) {
    __c_mcopy16(d, v, size * 2);
  }
}
#endif

#endif /* #if !defined(DESC_I8) */

/** \brief
 * helper function to store the MXINT_T value into a simple numerical type
 */
static void
store_mxint_t(void *b, F90_Desc *bd, MXINT_T v)
{
  switch (TYPEKIND(bd)) {
  case __INT1:
  case __LOG1:
    *(__INT1_T *)b = (__INT1_T)v;
    break;
  case __INT2:
  case __LOG2:
    *(__INT2_T *)b = (__INT2_T)v;
    break;
  case __INT4:
  case __LOG4:
    *(__INT4_T *)b = (__INT4_T)v;
    break;
  case __INT8:
  case __LOG8:
    *(__INT8_T *)b = (__INT8_T)v;
    break;
  case __REAL4:
    *(__REAL4_T *)b = (__REAL4_T)v;
    break;
  case __REAL8:
    *(__REAL8_T *)b = (__REAL8_T)v;
    break;
  case __REAL16:
    *(__REAL16_T *)b = (__REAL16_T)v;
    break;
  default:
    *(__STAT_T *)b = (__STAT_T)v;
  }
}

/** \brief
 * helper function to store the STAT_T value into a varying int
 */
static MXINT_T
mxint(F90_Desc *bd)
{
  MXINT_T v;

  switch (TYPEKIND(bd)) {
  case __INT1:
  case __LOG1:
    v = ~((__INT1_T)1 << (8 * sizeof(__INT1_T) - 1));
    break;
  case __INT2:
  case __LOG2:
    v = ~((__INT2_T)1 << (8 * sizeof(__INT2_T) - 1));
    break;
  case __INT8:
  case __LOG8:
    v = ~((__INT8_T)1 << (8 * sizeof(__INT8_T) - 1));
    break;
  case __INT4:
  case __LOG4:
  default:
    v = ~((__INT4_T)1 << (8 * sizeof(__INT4_T) - 1));
    break;
  }
  return v;
}
