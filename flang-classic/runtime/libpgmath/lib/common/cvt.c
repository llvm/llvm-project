/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define IEEE 1

/*
 *	ecvt converts to decimal
 *	the number of digits is specified by ndigit
 *	decpt is set to the position of the decimal point
 *	sign is set to 0 for positive, 1 for negative
 */

#ifdef IEEE

typedef union {
  double x;
  /* LITTLE ENDIAN: */
  struct {
    unsigned int ml;
    unsigned int mh : 20;
    unsigned int e : 11;
    unsigned int s;
  } f;
  /*  BIG ENDIAN: */
  /*
  struct {
      unsigned int s;
      unsigned int e:11;
      unsigned int mh:20;
      unsigned int ml;
  }f;
  */
} IEEE_DBL;

static int
__isspecial(double f, char *bp)
{
  IEEE_DBL *ip;
  ip = (IEEE_DBL *)&f;
  if ((ip->f.e & 0x7ff) != 0x7ff)
    return (0);
  if (ip->f.mh || ip->f.ml)
    strcpy(bp, "NaN");
  else if (ip->f.s)
    strcpy(bp, "-Infinity");
  else
    strcpy(bp, "Infinity");
  return (1);
}

#define NDIG 512
#else
#define NDIG 80
#endif

static char *
cvt(double arg, int ndigits, int *decpt, int *sign, int eflag)
{
  register int r2;
  double fi, fj;
  register char *p, *p1;
  static char buf[NDIG];

#ifdef IEEE
  /* XXX */
  if (__isspecial(arg, buf))
    return (buf);
#endif
  if (ndigits >= NDIG - 1)
    ndigits = NDIG - 2;
  r2 = 0;
  *sign = 0;
  p = &buf[0];
  if (arg < 0) {
    *sign = 1;
    arg = -arg;
  }
  arg = modf(arg, &fi);
  p1 = &buf[NDIG];
  /*
   * Do integer part
   */
  if (fi != 0) {
    p1 = &buf[NDIG];
    while (fi != 0) {
      fj = modf(fi / 10, &fi);
      *--p1 = (int)((fj + .03) * 10) + '0';
      r2++;
    }
    while (p1 < &buf[NDIG])
      *p++ = *p1++;
  } else if (arg > 0) {
    while ((fj = arg * 10) < 1) {
      arg = fj;
      r2--;
    }
  }
  p1 = &buf[ndigits];
  if (eflag == 0)
    p1 += r2;
  *decpt = r2;
  if (p1 < &buf[0]) {
    buf[0] = '\0';
    return (buf);
  }
  while (p <= p1 && p < &buf[NDIG]) {
    arg *= 10;
    arg = modf(arg, &fj);
    *p++ = (int)fj + '0';
  }
  if (p1 >= &buf[NDIG]) {
    buf[NDIG - 1] = '\0';
    return (buf);
  }
  p = p1;
  *p1 += 5;
  while (*p1 > '9') {
    *p1 = '0';
    if (p1 > buf)
      ++*--p1;
    else {
      *p1 = '1';
      (*decpt)++;
      if (eflag == 0) {
        if (p > buf)
          *p = '0';
        p++;
      }
    }
  }
  *p = '\0';
  return (buf);
}

char *
ecvt(double arg, int ndigits, int *decpt, int *sign)
{
  return (cvt(arg, ndigits, decpt, sign, 1));
}

char *
fcvt(double arg, int ndigits, int *decpt, int *sign)
{
  return (cvt(arg, ndigits, decpt, sign, 0));
}
