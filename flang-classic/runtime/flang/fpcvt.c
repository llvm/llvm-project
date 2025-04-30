/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <string.h>
#include <ctype.h>
#if !defined(_WIN64)
#include <fenv.h>
#endif
#include "float128.h"
#include "fioMacros.h"
#include "stdioInterf.h"
#include "fio_fcb_flags.h"

/* this continues down to __io_fcvt() definition.
    we use our __io_fcvt for C90 but we can't use any of
    these 64-bit IEEE routines for it */

#include <errno.h>

union ieee {
  double d;
  struct {
    unsigned int lm : 32;
    unsigned int hm : 20;
    unsigned int e : 11;
    unsigned int s : 1;
  } v;
  int i[sizeof(double)/sizeof(int)];
};

/* The ieeeq union is invalid If float128_t is not really 128 bits wide,
   so its declaration and uses are guarded with TARGET_SUPPORTS_QUADFP. */
#ifdef TARGET_SUPPORTS_QUADFP
union ieeeq {
  float128_t q;
  struct {
    unsigned int lm[3];
    unsigned int hm : 16;
    unsigned int e : 15;
    unsigned int s : 1;
  } v;
  int i[sizeof(float128_t)/sizeof(int)];
};
#endif

typedef long INT;
typedef unsigned long UINT;
typedef double IEEE64;
typedef enum { ZERO, NIL, NORMAL, BIG, INFIN, NAN, DIVZ, SUBNORMAL } VAL;
typedef struct {
  VAL fval;
  int fsgn;
  int fexp;
  INT fman[4];
} UFP;

#define IEEE64_SUBNORMAL(a) (a.v.e == 0 && (a.v.hm != 0L || a.v.lm != 0L))

static int ufpdnorm(UFP *, int);

static void mtherr(const char *, int);
static void fperror(int x);

/* fperror error msg selectors */
#define FP_NOERR                 0 
#define FP_ILLEGAL_INPUT_OR_NAN -1
#define FP_OVERFLOW_ERROR       -2
#define FP_UNDERFLOW_ERROR      -3
#define FP_UNDEFINED_ERROR      -4

#ifdef FLANG_FPCVT_UNUSED
static void
ui64toa(INT m[2], char *s, int n, int decpl)
{
  int i, j;
  INT lo, hi;
  char buff[30];

  lo = m[1] & 0xFFFFL;
  hi = (m[1] >> 16) & 0xFFFFL;
  for (i = 0; m[0] != 0L || hi != 0L || lo != 0L; i++) {
    hi |= (m[0] % 10) << 16;
    lo |= (hi % 10) << 16;
    buff[i] = "0123456789"[lo % 10];
    m[0] /= 10;
    hi /= 10;
    lo /= 10;
  }
  if (n == 0)
    n = 1;
  if (i > n)
    n = i;
  if (decpl + 2 > n)
    n = decpl + 2;
  n--;
  i--;
  j = 0;
  for (j = 0; n > i; j++, n--) {
    s[j] = '0';
  }
  for (; i >= 0; i--, j++)
    s[j] = buff[i];
  s[j] = '\0';
}
#endif

static void
manshftr(INT *m, /* m[4] */
         int n)
{
  register int i;
  register int j;
  long mask;
  for (i = n; i >= 32; i -= 32) {
    m[3] = m[2];
    m[2] = m[1];
    m[1] = m[0];
    m[0] = 0L;
  }
  if (i > 0) {
    j = 32 - i;
    mask = ((UINT)(1L << (j)) - 1);
    m[3] = ((m[3] >> i) & mask) | (m[2] << j);
    m[2] = ((m[2] >> i) & mask) | (m[1] << j);
    m[1] = ((m[1] >> i) & mask) | (m[0] << j);
    m[0] = (m[0] >> i) & mask;
  }
}

static void
manshftl(INT *m, /* m[4] */
         int n)
{
  register int i;
  register int j;
  long mask;
  for (i = n; i >= 32; i -= 32) {
    m[0] = m[1];
    m[1] = m[2];
    m[2] = m[3];
    m[3] = 0L;
  }
  if (i > 0) {
    j = 32 - i;
    mask = ((UINT)(1L << (i)) - 1);
    m[0] = (m[0] << i) | ((m[1] >> j) & mask);
    m[1] = (m[1] << i) | ((m[2] >> j) & mask);
    m[2] = (m[2] << i) | ((m[3] >> j) & mask);
    m[3] = (m[3] << i);
  }
}

static void manadd(INT *m1, /* m1[4] */
                   INT *m2) /* m2[4] */
{
  INT t1, t2, carry;
  INT lo, hi;
  int i;
  carry = 0;
  for (i = 3; i >= 0; i--) {
    t1 = m1[i] & 0x0000FFFFL;
    t2 = m2[i] & 0x0000FFFFL;
    lo = t1 + t2 + carry;
    carry = (lo >> 16) & 0x0000FFFFL;
    lo &= 0x0000FFFFL;
    t1 = (m1[i] >> 16) & 0x0000FFFFL;
    t2 = (m2[i] >> 16) & 0x0000FFFFL;
    hi = t1 + t2 + carry;
    carry = (hi >> 16) & 0x0000FFFFL;
    hi <<= 16;
    m1[i] = hi | lo;
  }
}

static void
manrnd(INT *m, /* m[4] */
       int bits)
{
  int rndwrd, rndbit;
  int oddwrd, oddbit;
  INT round[4];
  static INT one[4] = {0L, 0L, 0L, 1L};
  rndwrd = bits / 32;
  rndbit = 31 - (bits % 32);
  oddwrd = (bits - 1) / 32;
  oddbit = 31 - ((bits - 1) % 32);
  if ((((INT)(m[rndwrd]) >> (rndbit)) &
       ((UINT)(1L << ((rndbit) - (rndbit) + 1)) - 1)) == 1) {
    round[0] = 0xFFFFFFFFL;
    round[1] = 0xFFFFFFFFL;
    round[2] = 0xFFFFFFFFL;
    round[3] = 0xFFFFFFFFL;
    manshftr(round, bits + 1);
    manadd(m, round);
    if ((((INT)(m[rndwrd]) >> (rndbit)) &
         ((UINT)(1L << ((rndbit) - (rndbit) + 1)) - 1)) == 1 &&
        (((INT)(m[oddwrd]) >> (oddbit)) &
         ((UINT)(1L << ((oddbit) - (oddbit) + 1)) - 1)) == 1) {
      manadd(m, one);
    }
  }
  manshftr(m, 128 - bits);
  manshftl(m, 128 - bits);
}

static void manmul(INT *m1, /* m1[4] */
                   INT *m2) /* m2[4] */
{
  register INT carry;
  register int i, j, k;
  INT p[8];
  INT n1[4];
  INT n2[4];
  static int jval[8] = {0, 0, 0, 0, 0, 1, 2, 3};
  static int kval[8] = {0, 0, 1, 2, 3, 3, 3, 3};
  for (i = 0, j = 0; i < 2; i++, j += 2) {
    n1[j] = (m1[i] >> 16) & 0x0000FFFFL;
    n1[j + 1] = m1[i] & 0x0000FFFFL;
    n2[j] = (m2[i] >> 16) & 0x0000FFFFL;
    n2[j + 1] = m2[i] & 0x0000FFFFL;
  }
  carry = 0;
  for (i = 7; i > 0; i--) {
    p[i] = carry & 0x0000FFFFL;
    carry = (carry >> 16) & 0x0000FFFFL;
    for (j = jval[i], k = kval[i]; j <= kval[i]; j++, k--) {
      p[i] += n1[j] * n2[k];
      carry += (p[i] >> 16) & 0x0000FFFFL;
      p[i] &= 0x0000FFFFL;
    }
  }
  p[0] = carry;
  for (i = 0, j = 0; i < 4; i++, j += 2)
    m1[i] = (p[j] << 16) | p[j + 1];
}

static void
ufpnorm(UFP *u)
{
  if (u->fman[0] == 0 && u->fman[1] == 0 && u->fman[2] == 0 && u->fman[3] == 0)
    return;
  while ((((INT)(u->fman[0]) >> (21)) &
          ((UINT)(1L << ((31) - (21) + 1)) - 1)) != 0) {
    manshftr(u->fman, 1);
    u->fexp++;
  }
  while ((((INT)(u->fman[0]) >> (20)) &
          ((UINT)(1L << ((20) - (20) + 1)) - 1)) == 0) {
    manshftl(u->fman, 1);
    u->fexp--;
  }
}

static void
ufprnd(UFP *u, int bits)
{
  ufpnorm(u);
  manrnd(u->fman, bits + 12);
  ufpnorm(u);
}
static INT ftab1[29][3] = {
    {0xA05C0DD7, 0x0F6E1619, -1162}, {0xA5CED43B, 0x7E3E9188, -1079},
    {0xAB70FE17, 0xC79AC6CA, -996},  {0xB1442798, 0xF49FFB4A, -913},
    {0xB749FAED, 0x14125D36, -830},  {0xBD8430BD, 0x08277231, -747},
    {0xC3F490AA, 0x77BD60FC, -664},  {0xCA9CF1D2, 0x06FDC03B, -581},
    {0xD17F3B51, 0xFCA3A7A0, -498},  {0xD89D64D5, 0x7A607744, -415},
    {0xDFF97724, 0x70297EBD, -332},  {0xE7958CB8, 0x7392C2C2, -249},
    {0xEF73D256, 0xA5C0F77C, -166},  {0xF79687AE, 0xD3EEC551, -83},
    {0x80000000, 0x00000000, 1},     {0x84595161, 0x401484A0, 84},
    {0x88D8762B, 0xF324CD0F, 167},   {0x8D7EB760, 0x70A08AEC, 250},
    {0x924D692C, 0xA61BE758, 333},   {0x9745EB4D, 0x50CE6332, 416},
    {0x9C69A972, 0x84B578D7, 499},   {0xA1BA1BA7, 0x9E1632DC, 582},
    {0xA738C6BE, 0xBB12D16C, 665},   {0xACE73CBF, 0xDC0BFB7B, 748},
    {0xB2C71D5B, 0xCA9023F8, 831},   {0xB8DA1662, 0xE7B00A17, 914},
    {0xBF21E440, 0x03ACDD2C, 997},   {0xC5A05277, 0x621BE293, 1080},
    {0xCC573C2A, 0x0ECCDAA6, 1163},
};
static INT ftab2[25][3] = {
    {0x80000000, 0x00000000, 1},  {0xA0000000, 0x00000000, 4},
    {0xC8000000, 0x00000000, 7},  {0xFA000000, 0x00000000, 10},
    {0x9C400000, 0x00000000, 14}, {0xC3500000, 0x00000000, 17},
    {0xF4240000, 0x00000000, 20}, {0x98968000, 0x00000000, 24},
    {0xBEBC2000, 0x00000000, 27}, {0xEE6B2800, 0x00000000, 30},
    {0x9502F900, 0x00000000, 34}, {0xBA43B740, 0x00000000, 37},
    {0xE8D4A510, 0x00000000, 40}, {0x9184E72A, 0x00000000, 44},
    {0xB5E620F4, 0x80000000, 47}, {0xE35FA931, 0xA0000000, 50},
    {0x8E1BC9BF, 0x04000000, 54}, {0xB1A2BC2E, 0xC5000000, 57},
    {0xDE0B6B3A, 0x76400000, 60}, {0x8AC72304, 0x89E80000, 64},
    {0xAD78EBC5, 0xAC620000, 67}, {0xD8D726B7, 0x177A8000, 70},
    {0x87867832, 0x6EAC9000, 74}, {0xA968163F, 0x0A57B400, 77},
    {0xD3C21BCE, 0xCCEDA100, 80},
};
static void
ufpxten(UFP *u, int exp)
{
  int i, j;
  if (exp < -350) {
    u->fval = NIL;
    return;
  }
  if (exp > 374) {
    u->fval = BIG;
    return;
  }
  i = (exp + 350) / 25;
  j = (exp + 350) % 25;
  ufpnorm(u);
  manshftl(u->fman, 11);
  manmul(u->fman, ftab1[i]);
  manmul(u->fman, ftab2[j]);
  manshftr(u->fman, 11);
  u->fexp += ftab1[i][2] + ftab2[j][2];
}

#ifdef FLANG_FPCVT_UNUSED
static void
ufptosci(UFP *u, char *s, int dp, int *decpt, int *sign)
{
  INT man[2];
  int exp10, exp2;

  *sign = u->fsgn;
  *decpt = 0;
  if (u->fval == NAN) {
    strcpy(s, "NaN");
    *sign = 0;
    return;
  }
  if (u->fval == INFIN) {
    strcpy(s, "Inf");
    return;
  }
  if (u->fval == SUBNORMAL)
    ufpnorm(u);
  man[0] = u->fman[0];
  man[1] = u->fman[1];
  exp2 = u->fexp;
  exp10 = (30103 * exp2 + 1000 * 100000L) / 100000L - 1000;
again:
  ufpxten(u, dp - exp10);
  u->fexp -= 52;
  if (u->fexp > 0)
    manshftl(u->fman, u->fexp);
  else
    manshftr(u->fman, -u->fexp);
  manrnd(u->fman, 64);
  ui64toa(u->fman, s, 0, dp);
  if (strlen(s) > dp + 2) {
    u->fman[0] = man[0];
    u->fman[1] = man[1];
    u->fexp = exp2;
    exp10++;
    goto again;
  }
  *decpt = exp10;
}
#endif

#ifdef FLANG_FPCVT_UNUSED
static void
ufptodec(UFP *u, char *s, int dp, int *decpt, int *sign)
{
  *sign = u->fsgn;
  *decpt = 0;
  if (u->fval == NAN) {
    strcpy(s, "NAN");
    return;
  }
  if (u->fval == INFIN) {
    strcpy(s, "INF");
    return;
  }
  ufpxten(u, dp);
  u->fexp -= 52;
  if (u->fexp > 0)
    manshftl(u->fman, u->fexp);
  else
    manshftr(u->fman, -u->fexp);
  manrnd(u->fman, 64);
  ui64toa(u->fman, s, 0, dp);
}
#endif

#ifdef FLANG_FPCVT_UNUSED
static void
dtoufp(IEEE64 d, UFP *u)
{
  union ieee v;

  v.d = d;
  u->fval = NORMAL;
  u->fexp = v.v.e - 1023;
  u->fsgn = v.v.s;
  u->fman[0] = v.v.hm;
  u->fman[1] = v.v.lm;
  u->fman[2] = 0L;
  u->fman[3] = 0L;
  if (IEEE64_SUBNORMAL(v)) {
    u->fval = SUBNORMAL;
    u->fexp = -1022;
    u->fman[0] = u->fman[0] & 0xffefffff;
  } else if (u->fexp == 1024) {
    if (u->fman[0] == 0 && u->fman[1] == 0)
      u->fval = INFIN;
    else
      u->fval = NAN;
    u->fman[0] |= 0x00100000L;
  } else if (u->fexp == -1023) {
    /* denorm to 0 for now */
    u->fval = ZERO;
    u->fexp = 0;
    u->fman[0] = 0L;
    u->fman[1] = 0L;
  } else
    u->fman[0] |= 0x00100000L;
}
#endif

static void
ufptod(UFP *u, IEEE64 *r)
{
  union ieee v;
  int bias = 1023;

  ufprnd(u, 52);
  if (u->fval == ZERO) {
    u->fexp = -1023;
    u->fman[0] = 0L;
    u->fman[1] = 0L;
  }
  if (u->fval == NAN) {
    u->fexp = 1024;
    u->fman[0] = ~0L;
    u->fman[1] = ~0L;
    __io_set_errno(ERANGE);
  }
  if (u->fval == INFIN || u->fval == BIG || u->fval == DIVZ) {
    u->fexp = 1024;
    u->fman[0] = 0L;
    u->fman[1] = 0L;
    __io_set_errno(ERANGE);
  }
  if (u->fval == NORMAL && u->fexp <= -1023) {
    if (ufpdnorm(u, 1022) < 0) {
      u->fval = NIL;
      __io_set_errno(ERANGE);
    } else
      u->fval = SUBNORMAL;
  } else if (u->fval == SUBNORMAL)
    (void)ufpdnorm(u, 1022);

  if (u->fval == NORMAL && u->fexp >= 1024) {
    u->fval = BIG;
    u->fexp = 1024;
    u->fman[0] = 0L;
    u->fman[1] = 0L;
  }

  /* IEEE format for subnormal floating point number has an exp of
     (-bias + 1).  if number is denormalized, need to modify the bias
     that is used below to calculate the ieee exponent */
  if (u->fval == SUBNORMAL || u->fval == NIL)
    bias = 1022;

  /* clear the pipes */
  v.i[0] = v.i[1] = 0;
  v.d = v.d * v.d + v.d;
  v.v.s = u->fsgn;
  v.v.e = u->fexp + bias;
  v.v.hm = u->fman[0];
  v.v.lm = u->fman[1];
  *r = v.d;
}

static int
ufpdnorm(UFP *u, int bias)
{
  /*  adjust the denormalized number, unset the implicit bit, and
      report true underflow condition */
  int diff;
  diff = u->fexp + bias;
  if (diff > 0)
    manshftl(u->fman, diff);
  else
    manshftr(u->fman, -diff);
  manrnd(u->fman, 64);
  u->fexp = -bias;
  if (bias >= 1022)
    u->fman[0] = u->fman[0] & 0xffefffff;
  else
    u->fman[0] = u->fman[0] & 0xff7fffff;
  if (u->fman[0] == 0 && u->fman[1] == 0)
    return -1;
  return 1;
}

static int
atoxi(char *s, INT *i, int n, int base)
{
  register char *end;
  register INT value;
  int sign;

  end = s + n;
  *i = 0;
  for (; s < end && isspace(*s); s++)
    ;
  if (s >= end)
    return (-1);
  sign = 1;
  if (*s == '-') {
    sign = -1;
    s++;
  } else if (*s == '+')
    s++;
  if (s >= end)
    return (-1);
  switch (base) {
  case 2:
    for (value = 0; s < end; s++) {
      if ((value & 0x80000000L) != 0)
        goto ovflo;
      value <<= 1;
      if (*s < '0' || *s > '1')
        return (-1);
      if (*s == '1')
        value |= 1L;
    }
    break;
  case 8:
    for (value = 0; s < end; s++) {
      if ((value & 0xE0000000L) != 0)
        goto ovflo;
      value <<= 3;
      if (*s < '0' || *s > '7')
        return (-1);
      value |= (*s - '0');
    }
    break;
  case 16:
    for (value = 0; s < end; s++) {
      if ((value & 0xF0000000L) != 0)
        goto ovflo;
      value <<= 4;
      if (*s < '0')
        return (-1);
      else if (*s <= '9')
        value |= (*s - '0');
      else if (*s < 'A')
        return (-1);
      else if (*s <= 'F')
        value |= (*s - 'A' + 10);
      else if (*s < 'a')
        return (-1);
      else if (*s <= 'f')
        value |= (*s - 'a' + 10);
      else
        return (-1);
    }
    break;
  case 10:
    for (value = 0; s < end; s++) {
      if (((value >> 1) & 0x7FFFFFFFL) > (0x7FFFFFFFL / 5))
        goto ovflo;
      value *= 5;
      value <<= 1;
      if (*s < '0' || *s > '9')
        return (-1);
      value += (*s - '0');
    }
    break;
  default:
    return (-1);
  }
  if (sign == -1) {
    if ((value & 0x80000000L) != 0 && value != 0x80000000L)
      goto ovflo;
    *i = (~value) + 1;
  } else
    *i = value;
  return (0);
ovflo:
  *i = value;
  return -2;
}

static void
atoui64(char *s, INT *m, /* m[2] */
        int n, INT *exp)
{
  char *end;
  int dp;
  INT lo;
  INT hi;

  m[0] = 0;
  m[1] = 0;
  *exp = 0;
  dp = -1;
  for (end = s + n; s < end; s++) {
    if (*s == '.') {
      if (dp != -1)
        break;
      dp = 0;
      continue;
    }
    if (*s < '0' || *s > '9')
      break;
    if ((m[0] & 0xF8000000L) != 0) {
      if (dp == -1)
        (*exp)++;
      continue;
    }
    lo = m[1] & 0xFFFFL;
    lo *= 10;
    hi = (m[1] >> 16) & 0xFFFFL;
    hi *= 10;
    hi += (lo >> 16) & 0xFFFFL;
    m[0] *= 10;
    m[0] += (hi >> 16) & 0xFFFFL;
    m[1] = ((hi & 0xFFFFL) << 16) | (lo & 0xFFFF);
    /* Must propagate carry here! */
    lo = m[1] & 0xFFFFL;
    lo += *s - '0';
    hi = (m[1] >> 16) & 0xFFFFL;
    hi += (lo >> 16) & 0xFFFFL;
    m[0] += (hi >> 16) & 0xFFFFL;
    m[1] = ((hi & 0xFFFFL) << 16) | (lo & 0xFFFF);

    if (dp != -1)
      dp++;
  }
  if (dp == -1)
    dp = 0;
  *exp -= dp;
}

static void
atoxufp(char *s, UFP *u, char **p)
{
  INT exp;
  int sign;
  int err;
  char *start;

  *p = s;
  u->fval = NORMAL;
  u->fsgn = 0;
  for (; isspace(*s); s++)
    ;
  if (*s == 0) {
    u->fval = ZERO;
    u->fman[0] = u->fman[1] = u->fman[2] = u->fman[3] = 0;
    return;
  }
  if (*s == '-') {
    u->fsgn = 1;
    s++;
  } else if (*s == '+')
    s++;
  if ((*s < '0' || *s > '9') && *s != '.') {
    u->fval = ZERO;
    u->fman[0] = u->fman[1] = u->fman[2] = u->fman[3] = 0;
    return;
  }
  start = s;
  if (*s == '.') {
    ++s;
    while (isdigit(*s))
      ++s;
  } else {
    while (isdigit(*s))
      ++s;
    if (*s == '.') {
      ++s;
      while (isdigit(*s))
        ++s;
    }
  }
  atoui64(start, u->fman, s - start, &exp);
  if (u->fman[0] == 0 && u->fman[1] == 0) {
    u->fval = ZERO;
    u->fsgn = 0; /* -0 -> +0 */
  }
  u->fman[2] = 0;
  u->fman[3] = 0;
  u->fexp = exp;
  if (*s != 'd' && *s != 'D' && *s != 'e' && *s != 'E' && *s != 'q' &&
      *s != 'Q') {
    goto ret;
  }
  s++;
  sign = 1;
  if (*s == '-') {
    sign = -1;
    s++;
  } else if (*s == '+')
    s++;
  start = s;
  while (isdigit(*s))
    s++;
  err = atoxi(start, &exp, (int)(s - start), 10);
  if (err == -1) {
    u->fval = ZERO;
    u->fman[0] = u->fman[1] = u->fman[2] = u->fman[3] = 0;
    return;
  }
  if (err == -2) {
    u->fval = sign > 0 ? BIG : ZERO;
    *p = s;
    goto ret;
  }
  u->fexp += sign * exp;
ret:
  if (*s == 'f' || *s == 'F' || *s == 'l' || *s == 'L')
    ++s;
  *p = s;
}

double
__fortio_strtod(char *s, char **p)
{
  IEEE64 d;
  UFP u;
  int exp;
  char *q;

  atoxufp(s, &u, &q);
  if (p != 0)
    *p = q;
  exp = u.fexp;
  u.fexp = 52;
  ufpxten(&u, exp);
  ufptod(&u, &d);
  return d;
}

/*
 * Convert to ndigit digits.  *decpt is position of decimal point
 * (0 is before first digit).  *sign is sign.
 */

#if defined(_WIN64)
#define FE_TONEAREST 0
#define FE_DOWNWARD 1024
#define FE_UPWARD 2048
#define FE_TOWARDZERO 3072
#endif

int __fenv_fegetround();

static void
writefmt(char *fmt, int prec, char c, int is_quad)
{
  int i, hprec, mprec, lprec;
  hprec = mprec = 0;
  lprec = prec;

  while (lprec >= 100) {
    hprec++;
    lprec -= 100;
  }
  while (lprec >= 10) {
    mprec++;
    lprec -= 10;
  }

  i = 0;
  fmt[i++] = '%';
  fmt[i++] = '-';
  fmt[i++] = '.';
  if (hprec) {
    fmt[i++] = '0' + hprec;
    fmt[i++] = '0' + mprec;
  } else if (mprec) {
    fmt[i++] = '0' + mprec;
  }
  fmt[i++] = '0' + lprec;
  if (is_quad)
    fmt[i++] = 'L';
  fmt[i++] = c;
  fmt[i++] = '\0';
}

#define EXPONENT_BIAS_D 0x3ff
#define EXPONENT_BIAS_Q 0x3fff
#define INFINITY_STR_LEN 8 /* len of strings "Infinity". */
#define SUBSCRIPT_2 2      /* subscript is 2 */
#define SUBSCRIPT_3 3      /* subscript is 3 */

char *
__fortio_ecvt(__BIGREAL_T lvalue, int width, int ndigit, int *decpt, int *sign,
              int round, int is_quad)
{
  int i, j;

  union ieee ieee_v;
#ifdef TARGET_SUPPORTS_QUADFP
  union ieeeq ieeeq_v;
#endif

  static char tmp[512];
  static char fmt[16];
  int idx, fexp, kdz, engfmt;
  int i0, i1;
  double value = (double)lvalue;
  /* This block of stuff is under consideration */
  engfmt = 0;
  if (round >= 256) {
    round -= 256;
    engfmt = 1;
  }

  if (round == 0)
    round = FIO_COMPATIBLE;
  if (round == FIO_PROCESSOR_DEFINED) {
    idx = __fenv_fegetround();
    if (idx == FE_TONEAREST)
      round = FIO_NEAREST;
    else if (idx == FE_DOWNWARD)
      round = FIO_DOWN;
    else if (idx == FE_UPWARD)
      round = FIO_UP;
    else if (idx == FE_TOWARDZERO)
      round = FIO_ZERO;
    /* Is there anything else? */
  }

#ifdef TARGET_SUPPORTS_QUADFP
  if (is_quad) {
    ieeeq_v.q = lvalue;
    fexp = ieeeq_v.v.e - EXPONENT_BIAS_Q;
    if (fexp == EXPONENT_BIAS_Q + 1) {
      if (ieeeq_v.v.lm[SUBSCRIPT_2] == 0 && ieeeq_v.v.lm[1] == 0 &&
          ieeeq_v.v.lm[0] == 0 && ieeeq_v.v.hm == 0) {
        if (width < INFINITY_STR_LEN + ieeeq_v.v.s)
          strcpy(tmp, "Inf");
        else
          strcpy(tmp, "Infinity");
        *sign = ieeeq_v.v.s;
        *decpt = 0;
        return tmp;
      } else {
        strcpy(tmp, "NaN");
        *sign = 0;
        *decpt = 0;
        return tmp;
      }
    }

    *sign = ieeeq_v.v.s;
    ieeeq_v.v.s = 0;
    lvalue = ieeeq_v.q;
  } else
#endif
  {
    ieee_v.d = value;
    fexp = ieee_v.v.e - EXPONENT_BIAS_D;
    if (fexp == EXPONENT_BIAS_D + 1) {
      if (ieee_v.v.hm == 0 && ieee_v.v.lm == 0) {
        strcpy(tmp, "Inf");
        *sign = ieee_v.v.s;
        *decpt = 0;
        return tmp;
      } else {
        strcpy(tmp, "NaN");
        *sign = 0;
        *decpt = 0;
        return tmp;
      }
    }

    *sign = ieee_v.v.s;
    ieee_v.v.s = 0;
    /* Set lvalue = value for the snprintf calls below when is_quad is true. */
    lvalue = value = ieee_v.d;
  }

  /* For compatible mode, round '5' away from zero */
  /* Compatible rounding, or compatible in number of good bits??? */

  if (round == FIO_COMPATIBLE) {
    writefmt(fmt, ndigit, 'E', is_quad);
    if (!is_quad)
      j = snprintf(tmp, sizeof(tmp), fmt, value);
    else
      j = snprintf(tmp, sizeof(tmp), fmt, lvalue);
    if (ndigit) {
      i0 = 1;
      tmp[i0] = tmp[0];
      } else {
        i0 = 0;
      }
      i = i0 + ndigit + 3;
      kdz = 0;
      while ((tmp[i] >= '0') && (tmp[i] <= '9'))
        kdz = kdz * 10 + tmp[i++] - '0';
      if (tmp[i0 + ndigit + 2] == '-')
        kdz = -kdz;
      *decpt = kdz + 1;
      if (ndigit) {
        if (engfmt) {
          /* if decpt is zero, or a multiple of 3, need to round a little
             closer.  Actual number of bits could be ndigit-2, ndigit-1,
             or ndigit
          */
          short ndigitadj;
          ndigitadj = *decpt;
          ndigitadj = (ndigitadj - 360) % 3;
          ndigit += ndigitadj;
        }
        i1 = i0 + ndigit;

        /* We know sprintf is rounded, so get more bits */
        if (tmp[i1] == '5') {
          writefmt(fmt, ndigit + 20, 'E', is_quad);
          if (!is_quad)
            j = snprintf(tmp, sizeof(tmp), fmt, value);
          else
            j = snprintf(tmp, sizeof(tmp), fmt, lvalue);
          i0 = 1;
          tmp[i0] = tmp[0];
        }
        if (tmp[i1] < '5') {
          tmp[i1] = '\0';
        } else {
          tmp[i1] = '\0';
          i1--;
          while ((tmp[i1] == '9') && (i1 >= i0)) {
            tmp[i1--] = '0';
          }
          if (i1 >= i0) {
            tmp[i1] = tmp[i1] + 1;
          } else {
            i0--;
            tmp[i0] = '1';
            *decpt = kdz + 2;
          }
        }
        return tmp + i0;
      } else {
        tmp[2] = '\0';
        return tmp + i0;
      }
    }

  if ((round == FIO_NEAREST) || (round == FIO_PROCESSOR_DEFINED)) {
      /* Algorithm for round nearest:
         Turns out that sprintf is nearest
      */
      if (ndigit) {
        writefmt(fmt, ndigit - 1, 'E', is_quad);
        if (!is_quad)
          j = snprintf(tmp, sizeof(tmp), fmt, value);
        else
          j = snprintf(tmp, sizeof(tmp), fmt, lvalue);
        if (ndigit > 1) {
          i0 = 1;
          tmp[i0] = tmp[0];
        } else {
          i0 = 0;
        }
        i = i0 + ndigit + 2;
        kdz = 0;
        while ((tmp[i] >= '0') && (tmp[i] <= '9'))
          kdz = kdz * 10 + tmp[i++] - '0';
        if (tmp[i0 + ndigit + 1] == '-')
          kdz = -kdz;
        *decpt = kdz + 1;
        if (engfmt) {
          /* if decpt is zero, or a multiple of 3, need to round a little
             closer.  Actual number of bits could be ndigit-2, ndigit-1,
             or ndigit
          */
          short ndigitadj;
          ndigitadj = *decpt;
          ndigitadj = (ndigitadj - 360) % 3;
          ndigit += ndigitadj;
          i1 = i0 + ndigit;
          if (tmp[i1] == '5') {
            /* Use sprintf to round again */
            writefmt(fmt, ndigit - 1, 'E', is_quad);
            if (!is_quad)
              j = snprintf(tmp, sizeof(tmp), fmt, value);
            else
              j = snprintf(tmp, sizeof(tmp), fmt, lvalue);
            if (ndigit > 1) {
              i0 = 1;
              tmp[i0] = tmp[0];
            } else {
              i0 = 0;
            }
            i = i0 + ndigit + 2;
            kdz = 0;
            while ((tmp[i] >= '0') && (tmp[i] <= '9'))
              kdz = kdz * 10 + tmp[i++] - '0';
            if (tmp[i0 + ndigit + 1] == '-')
              kdz = -kdz;
            *decpt = kdz + 1;
            i1 = i0 + ndigit;
            tmp[i1] = '\0';
            return tmp + i0;
          } else if ((tmp[i1] < '5') || (tmp[i1] == 'E')) {
            /* These are rounded correctly */
            tmp[i1] = '\0';
            return tmp + i0;
          } else {
            /* These need to round up */
            tmp[i1] = '\0';
            i1--;
            while ((tmp[i1] == '9') && (i1 >= 0)) {
              tmp[i1--] = '0';
            }
            if (i1 >= 0) {
              tmp[i1] = tmp[i1] + 1;
              return tmp + i0;
            } else {
              tmp[0] = '1';
              *decpt = *decpt + 1;
              return tmp;
            }
          }
        } else {
          tmp[i0 + ndigit] = '\0';
          return tmp + i0;
        }
      } else {
        tmp[0] = '0';
        tmp[1] = '\0';
        return tmp;
      }
    }

  if (((round == FIO_DOWN) && (*sign == 0)) ||
      ((round == FIO_UP) && (*sign == 1)) || ((round == FIO_ZERO))) {
      /* Algorithm for round down, positive > 1.0:
         Add 1 character to the format.
         call sprintf.
         Find the exponent sprintf gave, and adjust our approx if needed.
         If the extra character(s) are 0, we need to do more work:
           Get a whole bunch more characters.  For now, 20 more
         Round Down:
         Lop everything extra off.
      */
      writefmt(fmt, ndigit, 'E', is_quad);
      if (!is_quad)
        j = snprintf(tmp, sizeof(tmp), fmt, value);
      else
        j = snprintf(tmp, sizeof(tmp), fmt, lvalue);
      i0 = 1;
      tmp[i0] = tmp[0];
      i = ndigit + 4;
      kdz = 0;
      while ((tmp[i] >= '0') && (tmp[i] <= '9'))
        kdz = kdz * 10 + tmp[i++] - '0';
      if (tmp[ndigit + 3] == '-')
        kdz = -kdz;
      *decpt = kdz + 1;
      if (engfmt) {
        /* if decpt is zero, or a multiple of 3, need to round a little
           closer.  Actual number of bits could be ndigit-2, ndigit-1,
           or ndigit
        */
        short ndigitadj;
        ndigitadj = *decpt;
        ndigitadj = (ndigitadj - 360) % 3;
        ndigit += ndigitadj;
      }
      if (ndigit) {
        i = ndigit + 1;
        if (tmp[i] == '0') {
          writefmt(fmt, ndigit + 20, 'E', is_quad);
          if (!is_quad)
            j = snprintf(tmp, sizeof(tmp), fmt, value);
          else
            j = snprintf(tmp, sizeof(tmp), fmt, lvalue);
          i0 = 1;
          tmp[i0] = tmp[0];
        }
        tmp[ndigit + 1] = '\0';
      } else {
        tmp[2] = '\0';
      }
      return tmp + 1;
    }

  if (((round == FIO_UP) && (*sign == 0)) ||
      ((round == FIO_DOWN) && (*sign == 1))) {
      /* Algorithm for round up, positive >= 1.0:
         Add 1 character to the format.
         call sprintf.
         Find the exponent sprintf gave, and adjust our approx if needed.
         If the extra character(s) are 0, we need to do more work:
           Get a whole bunch more characters.  For now, 20 more
           Search through the extra characters to find something > 0.
           If we found it, round up.  Else return
         Round Up:
         If we find 9, set it to zero and keep looking to the left.
         If we find a character other than 9, add 1 and we're done
         If we went all the way, make tmp[0] 1, and return that.
      */
      writefmt(fmt, ndigit, 'E', is_quad);
      if (!is_quad)
        j = snprintf(tmp, sizeof(tmp), fmt, value);
      else
        j = snprintf(tmp, sizeof(tmp), fmt, lvalue);
      i0 = 1;
      tmp[i0] = tmp[0];
      i = ndigit + 4;
      kdz = 0;
      while ((tmp[i] >= '0') && (tmp[i] <= '9'))
        kdz = kdz * 10 + tmp[i++] - '0';
      if (tmp[ndigit + 3] == '-')
        kdz = -kdz;
      *decpt = kdz + 1;
      if (engfmt) {
        /* if decpt is zero, or a multiple of 3, need to round a little
           closer.  Actual number of bits could be ndigit-2, ndigit-1,
           or ndigit
        */
        short ndigitadj;
        ndigitadj = *decpt;
        ndigitadj = (ndigitadj - 360) % 3;
        ndigit += ndigitadj;
      }
      i = ndigit + 1;
      if (ndigit) {
        if (tmp[i] == '0') {
          writefmt(fmt, ndigit + 20, 'E', is_quad);
          if (!is_quad)
            j = snprintf(tmp, sizeof(tmp), fmt, value);
          else
            j = snprintf(tmp, sizeof(tmp), fmt, lvalue);
          i0 = 1;
          tmp[i0] = tmp[0];
          tmp[ndigit + 21] = '\0';
          for (i = ndigit + 1; tmp[i] != '\0'; i++) {
            if (tmp[i] != '0')
              break;
          }
          if (tmp[i] == '\0') {
            tmp[ndigit + 1] = '\0';
            return tmp + 1;
          } else {
            i = ndigit;
            while ((tmp[i] == '9') && (i >= 1)) {
              tmp[i--] = '0';
            }
            tmp[ndigit + 1] = '\0';
            if (i == 0) {
              tmp[0] = '1';
              return tmp;
            } else {
              tmp[i] = tmp[i] + 1;
              return tmp + 1;
            }
          }
        } else { /* if (tmp[i] > '0') round up */
          i--;
          while ((tmp[i] == '9') && (i >= 1)) {
            tmp[i--] = '0';
          }
          tmp[ndigit + 1] = '\0';
          if (i == 0) {
            tmp[0] = '1';
            return tmp;
          } else {
            tmp[i] = tmp[i] + 1;
            return tmp + 1;
          }
        }
      } else {
        tmp[2] = '\0';
        return tmp + 1;
      }
    }
  mtherr("internal convert", FP_UNDEFINED_ERROR);
  return NULL;
}

char *
__fortio_fcvt(__BIGREAL_T lv, int width, int prec, int sf, int *decpt, int *sign,
              int round, int is_quad)
{

  union ieee ieee_v;
#ifdef TARGET_SUPPORTS_QUADFP
  union ieeeq ieeeq_v;
#endif
  static char tmp[512];
  static char fmt[16];
  int idx, fexp, nexp, kdz, ldz;
  int i, j, i0, i1;
  double v = (double)lv;

  /* This block of stuff is under consideration */
  if (round == 0)
    round = FIO_COMPATIBLE;
  if (round == FIO_PROCESSOR_DEFINED) {
    idx = __fenv_fegetround();
    if (idx == FE_TONEAREST)
      round = FIO_NEAREST;
    else if (idx == FE_DOWNWARD)
      round = FIO_DOWN;
    else if (idx == FE_UPWARD)
      round = FIO_UP;
    else if (idx == FE_TOWARDZERO)
      round = FIO_ZERO;
    /* Is there anything else? */
  }

#ifdef TARGET_SUPPORTS_QUADFP
  if (is_quad) {
    ieeeq_v.q = lv;
    fexp = ieeeq_v.v.e - EXPONENT_BIAS_Q;
    if (fexp == EXPONENT_BIAS_Q + 1) {
      if (ieeeq_v.v.lm[SUBSCRIPT_2] == 0 && ieeeq_v.v.lm[1] == 0 &&
          ieeeq_v.v.lm[0] == 0 && ieeeq_v.v.hm == 0) {
        if (width < INFINITY_STR_LEN + ieeeq_v.v.s)
          strcpy(tmp, "Inf");
        else
          strcpy(tmp, "Infinity");
        *sign = ieeeq_v.v.s;
        *decpt = 0;
        return tmp;
      } else {
        strcpy(tmp, "NaN");
        *sign = 0;
        *decpt = 0;
        return tmp;
      }
    }

    /* I've determined sprintf is FIO_NEAREST */
    /* I've determined Intel seems to use PROCESSOR_DEFINED as NEAREST */
    /* Use an sprintf implementation; this is probably the fastest path thru */

    *sign = ieeeq_v.v.s;
    ieeeq_v.v.s = 0;
    lv = ieeeq_v.q;
  } else
#endif
  {
    ieee_v.d = v;
    fexp = ieee_v.v.e - EXPONENT_BIAS_D;
    if (fexp == EXPONENT_BIAS_D + 1) {
      if (ieee_v.v.hm == 0 && ieee_v.v.lm == 0) {
        strcpy(tmp, "Inf");
        *sign = ieee_v.v.s;
        *decpt = 0;
        return tmp;
      } else {
        strcpy(tmp, "NaN");
        *sign = 0;
        *decpt = 0;
        return tmp;
      }
    }

    /* I've determined sprintf is FIO_NEAREST */
    /* I've determined Intel seems to use PROCESSOR_DEFINED as NEAREST */
    /* Use an sprintf implementation; this is probably the fastest path thru */

    *sign = ieee_v.v.s;
    ieee_v.v.s = 0;
    /* Set lv = v for the snprintf calls below when is_quad is true. */
    lv = v = ieee_v.d;
  }

  if (fexp >= 0) {
/* Here for abs(v) >= 1.0 */
/* Compute the rough approximation to the number of whole digits
   required to hold this number.  If the width passed in doesn't
   hold that, do enough to bail out early, which is just an optimization.
   Approx works thusly:
     Pull out the exponent.  The exact value to subtract is 1023*256.
     Fudge some to make sure 10Exx always has extra digits.
     1233 is an approximation of (4096 * log(2)/log(10)).
     Use the lookup table to get log(2) approx of the top 6 bits
     of the mantissa, 1.xxxxxx.
     (2 ** exp) * (1.xxxxxx) means you add these two values together.  */
      static const unsigned char lkup[64] = {
        3,   9,   14,  20,  25,  30,  36,  41,  46,  51,  56,  61,  66,
        71,  75,  80,  85,  89,  94,  98,  103, 107, 111, 116, 120, 124,
        128, 132, 136, 140, 144, 148, 152, 155, 159, 163, 167, 170, 174,
        178, 181, 185, 188, 192, 195, 198, 202, 205, 208, 212, 215, 218,
        221, 224, 228, 231, 234, 237, 240, 243, 246, 249, 252, 255};
#ifdef TARGET_SUPPORTS_QUADFP
      if (is_quad) {
        nexp = ((ieeeq_v.i[SUBSCRIPT_3] & 0x7fff0000) >> 8) -
               4194041; // 4194040 works too
        idx = ((ieeeq_v.i[SUBSCRIPT_3] & 0xfc00) >> 10);
      } else
#endif
      {
        nexp = ((ieee_v.i[1] & 0x7ff00000) >> 12) - 261881; // 261880 works too
        idx = ((ieee_v.i[1] & 0xfc000) >> 14);
      }
      nexp += lkup[idx];
      ldz = (nexp * 1233) >> 20;

    /* For compatible mode, round '5' away from zero */
    /* Compatible rounding, or compatible in number of good bits??? */

    if (round == FIO_COMPATIBLE) {
      prec += sf; /* Only for compatible mode, scale factor contribution
                     goes in before rounding.  According to my reading
                     of the spec */
        /* Algorithm for compatible > 1.0:
           Add 1 character to the format.
           call sprintf.
           Find the exponent sprintf gave, and adjust our approx if needed.
           If the first extra character is 5, we need to do more work:
             Get a whole bunch more characters.  For now, 20 more
           Round according to first extra character.  5+ goes up.  < 5 down.
        */
        if ((prec + ldz + 1) <= 0) {
          writefmt(fmt, 1, 'E', is_quad);
          if (!is_quad)
            j = snprintf(tmp, sizeof(tmp), fmt, v);
          else
            j = snprintf(tmp, sizeof(tmp), fmt, lv);
          i0 = 1;
          tmp[i0] = tmp[0];
          i = 5;
          /* Still need to get the right exponent? */
          kdz = 0;
          while ((tmp[i] >= '0') && (tmp[i] <= '9'))
            kdz = kdz * 10 + tmp[i++] - '0';
          tmp[i0 + 1] = '\0';
          *decpt = kdz + 1;
          return tmp + i0;
        } else {
          writefmt(fmt, prec + ldz + 1, 'E', is_quad);
          if (!is_quad)
            j = snprintf(tmp, sizeof(tmp), fmt, v);
          else
            j = snprintf(tmp, sizeof(tmp), fmt, lv);
          i0 = 1;
          tmp[i0] = tmp[0];
          i = prec + ldz + 4 + 1;
          kdz = 0;
          while ((tmp[i] >= '0') && (tmp[i] <= '9'))
            kdz = kdz * 10 + tmp[i++] - '0';
          i = prec + ldz + 2;
          *decpt = kdz + 1;
          if ((kdz == ldz) && (tmp[i] == '5')) {
            writefmt(fmt, prec + ldz + 1 + 20, 'E', is_quad);
            if (!is_quad)
              j = snprintf(tmp, sizeof(tmp), fmt, v);
            else
              j = snprintf(tmp, sizeof(tmp), fmt, lv);
            i0 = 1;
            tmp[i0] = tmp[0];
          } else if (kdz + 1 == ldz) {
            ldz = kdz;
            if ((tmp[i - 1] == '5') && (tmp[i] == '0')) {
              writefmt(fmt, prec + ldz + 1 + 20, 'E', is_quad);
              if (!is_quad)
                j = snprintf(tmp, sizeof(tmp), fmt, v);
              else
                j = snprintf(tmp, sizeof(tmp), fmt, lv);
              i0 = 1;
              tmp[i0] = tmp[0];
            }
          }
          i1 = i0 + ldz + prec + 1;
        }

        if (tmp[i1] < '5') {
          tmp[i1] = '\0';
          *decpt = ldz + 1;
        } else {
          tmp[i1] = '\0';
          i1--;
          while ((tmp[i1] == '9') && (i1 >= i0)) {
            tmp[i1--] = '0';
          }
          if (i1 >= i0) {
            tmp[i1] = tmp[i1] + 1;
            *decpt = ldz + 1;
          } else {
            i0--;
            tmp[i0] = '1';
            *decpt = ldz + 2;
          }
        }
        return tmp + i0;
      }

    if ((round == FIO_NEAREST) || (round == FIO_PROCESSOR_DEFINED)) {
        /* Algorithm for round nearest, positive or negative > 1.0:
           Turns out that sprintf is nearest
        */
        writefmt(fmt, prec + ldz, 'E', is_quad);
        if (!is_quad)
          j = snprintf(tmp, sizeof(tmp), fmt, v);
        else
          j = snprintf(tmp, sizeof(tmp), fmt, lv);
        i0 = 1;
        tmp[i0] = tmp[0];
        i = prec + ldz + 4;
        kdz = 0;
        while ((tmp[i] >= '0') && (tmp[i] <= '9'))
          kdz = kdz * 10 + tmp[i++] - '0';
        if (kdz + 1 == ldz) {
          ldz = kdz;
          writefmt(fmt, prec + ldz, 'E', is_quad);
          if (!is_quad)
            j = snprintf(tmp, sizeof(tmp), fmt, v);
          else
            j = snprintf(tmp, sizeof(tmp), fmt, lv);
          i0 = 1;
          tmp[i0] = tmp[0];
        }
        *decpt = ldz + 1;
        tmp[prec + ldz + 2] = '\0';
        return tmp + 1;
      }

    if (((round == FIO_DOWN) && (*sign == 0)) ||
        ((round == FIO_UP) && (*sign == 1)) || ((round == FIO_ZERO))) {
        /* Algorithm for round down, positive > 1.0:
           Add 1 character to the format.
           call sprintf.
           Find the exponent sprintf gave, and adjust our approx if needed.
           If the extra character(s) are 0, we need to do more work:
             Get a whole bunch more characters.  For now, 20 more
           Round Down:
           Lop everything extra off.
        */
        writefmt(fmt, prec + ldz + 1, 'E', is_quad);
        if (!is_quad)
          j = snprintf(tmp, sizeof(tmp), fmt, v);
        else
          j = snprintf(tmp, sizeof(tmp), fmt, lv);
        i0 = 1;
        tmp[i0] = tmp[0];
        i = prec + ldz + 4 + 1;
        kdz = 0;
        while ((tmp[i] >= '0') && (tmp[i] <= '9'))
          kdz = kdz * 10 + tmp[i++] - '0';
        if ((kdz == ldz) || (kdz + 1 == ldz)) {
          *decpt = kdz + 1;
          i = prec + ldz + 2;
          j = prec + kdz + 2;
          ldz = kdz;
          if ((tmp[i] == '0') && (tmp[j] == '0')) {
            writefmt(fmt, prec + ldz + 1 + 20, 'E', is_quad);
            if (!is_quad)
              j = snprintf(tmp, sizeof(tmp), fmt, v);
            else
              j = snprintf(tmp, sizeof(tmp), fmt, lv);
            i0 = 1;
            tmp[i0] = tmp[0];
          }
          tmp[prec + ldz + 2] = '\0';
          return tmp + 1;
        }
      }

    if (((round == FIO_UP) && (*sign == 0)) ||
        ((round == FIO_DOWN) && (*sign == 1))) {
        /* Algorithm for round up, positive >= 1.0:
           Add 1 character to the format.
           call sprintf.
           Find the exponent sprintf gave, and adjust our approx if needed.
           If the extra character(s) are 0, we need to do more work:
             Get a whole bunch more characters.  For now, 20 more
             Search through the extra characters to find something > 0.
             If we found it, round up.  Else return
           Round Up:
           If we find 9, set it to zero and keep looking to the left.
           If we find a character other than 9, add 1 and we're done
           If we went all the way, make tmp[0] 1, and return that.
        */
        writefmt(fmt, prec + ldz + 1, 'E', is_quad);
        if (!is_quad)
          j = snprintf(tmp, sizeof(tmp), fmt, v);
        else
          j = snprintf(tmp, sizeof(tmp), fmt, lv);
        i0 = 1;
        tmp[i0] = tmp[0];
        i = prec + ldz + 4 + 1;
        kdz = 0;
        while ((tmp[i] >= '0') && (tmp[i] <= '9'))
          kdz = kdz * 10 + tmp[i++] - '0';
        if ((kdz == ldz) || (kdz + 1 == ldz)) {
          *decpt = kdz + 1;
          i = prec + ldz + 2;
          j = prec + kdz + 2;
          ldz = kdz;
          if ((tmp[i] == '0') && (tmp[j] == '0')) {
            writefmt(fmt, prec + ldz + 1 + 20, 'E', is_quad);
            if (!is_quad)
              j = snprintf(tmp, sizeof(tmp), fmt, v);
            else
              j = snprintf(tmp, sizeof(tmp), fmt, lv);
            i0 = 1;
            tmp[i0] = tmp[0];
            tmp[prec + ldz + 22] = '\0';
            for (i = ldz + prec + 2; tmp[i] != '\0'; i++) {
              if (tmp[i] != '0')
                break;
            }
            if (tmp[i] == '\0') {
              tmp[prec + ldz + 2] = '\0';
              return tmp + 1;
            } else {
              i = prec + ldz + 1;
              while ((tmp[i] == '9') && (i >= 1)) {
                tmp[i--] = '0';
              }
              tmp[prec + ldz + 2] = '\0';
              if (i == 0) {
                tmp[0] = '1';
                (*decpt)++;
                return tmp;
              } else {
                tmp[i] = tmp[i] + 1;
                return tmp + 1;
              }
            }
          } else { /* if (tmp[i] > '0') round up */
            i--;
            while ((tmp[i] == '9') && (i >= 1)) {
              tmp[i--] = '0';
            }
            tmp[prec + ldz + 2] = '\0';
            if (i == 0) {
              tmp[0] = '1';
              (*decpt)++;
              return tmp;
            } else {
              tmp[i] = tmp[i] + 1;
              return tmp + 1;
            }
          }
        }
      }

  } else {
    if (round == FIO_COMPATIBLE)
      prec += sf;

    if (prec > 0) {

        /* Piece together the sprintf format string */
        /* Avoid calling sprintf though.  Another optimization, I think */
        if ((round == FIO_UP) || (round == FIO_DOWN) || (round == FIO_ZERO))
          /* Want one extra bit for rounding */
          writefmt(fmt, prec + 1, 'f', is_quad);
        else if (round == FIO_COMPATIBLE)
          writefmt(fmt, prec + 1, 'f', is_quad);
        else
          writefmt(fmt, prec, 'f', is_quad);

        /* Rely on sprintf to correctly round the result */
        if (!is_quad)
          j = snprintf(tmp, sizeof(tmp), fmt, v);
        else
          j = snprintf(tmp, sizeof(tmp), fmt, lv);

        if (round == FIO_COMPATIBLE) {
          i1 = 2 + prec;
          if (i1 <= 2) {
            tmp[1] = tmp[0];
            tmp[2] = '\0';
            *decpt = 0;
            return tmp + 1;
          }

          if (tmp[0] == '1') { /* Already rounded up to 1.0 */
            tmp[1] = tmp[0];
            *decpt = 1;
            tmp[prec + 2] = '\0';
            return tmp + 1;
          }

          if (tmp[2 + prec] == '5') {
            writefmt(fmt, prec + 21, 'f', is_quad);
            if (!is_quad)
              j = snprintf(tmp, sizeof(tmp), fmt, v);
            else
              j = snprintf(tmp, sizeof(tmp), fmt, lv);
          }
          if (tmp[i1] < '5') {
            tmp[prec + 2] = '\0';
            *decpt = 0;
            return tmp + 2;
          } else {
            i = 1 + prec;
            while ((tmp[i] == '9') && (i != 1)) {
              tmp[i--] = '0';
            }
            if (i != 1) {
              tmp[prec + 2] = '\0';
              tmp[i] = tmp[i] + 1;
              *decpt = 0;
              return tmp + 2;
            } else {
              tmp[prec + 2] = '\0';
              tmp[1] = '1';
              *decpt = 1;
              return tmp + 1;
            }
          }
      }

      if ((round == FIO_NEAREST) || (round == FIO_PROCESSOR_DEFINED)) {
          /* Algorithm for round nearest, positive or negative < 1.0:
             Turns out that sprintf is nearest
          */
          if (tmp[0] == '1') {
            i0 = 1;
            tmp[i0] = tmp[0];
            *decpt = 1;
            return tmp + 1;
          } else {
            tmp[prec + 2] = '\0';
            *decpt = 0;
            return tmp + 2;
          }
        }

      if (((round == FIO_UP) && (*sign == 0)) ||
          ((round == FIO_DOWN) && (*sign == 1))) {
        /* Algorithm for round up, positive < 1.0:
           Add 1 character to the format.
           call sprintf.
           If the extra character was 0, we need to do more work:
           Get a whole bunch more characters.  For now, 20 more
           Search through the extra characters to find something > 0.
           If we found it, round up.  Else return
           Round Up:
           If we find 9, set it to zero and keep looking to the left.
           If we find a character other than 9, add 1 and we're done
           If we went all the way, make tmp[0] 1, and return that.
        */
        if (tmp[2 + prec] == '0') {
          writefmt(fmt, prec + 20, 'f', is_quad);
          if (!is_quad)
            j = snprintf(tmp, sizeof(tmp), fmt, v);
          else
            j = snprintf(tmp, sizeof(tmp), fmt, lv);
          tmp[2 + prec + 20] = '\0';
          for (i = prec + 2; tmp[i] != '\0'; i++) {
            if (tmp[i] != '0')
              break;
            }
            if (i == 2 + prec + 20) {
              tmp[prec + 2] = '\0';
              *decpt = 0;
              return tmp + 2;
            }
          }
        i = prec + 1;
        while ((tmp[i] == '9') && (i != 1)) {
          tmp[i--] = '0';
        }
        if (i != 1) {
          tmp[prec + 2] = '\0';
          tmp[i] = tmp[i] + 1;
          *decpt = 0;
          return tmp + 2;
        } else {
          tmp[prec + 2] = '\0';
          tmp[1] = '1';
          *decpt = 1;
          return tmp + 1;
        }
      }

      if (((round == FIO_DOWN) && (*sign == 0)) ||
          ((round == FIO_UP) && (*sign == 1)) || ((round == FIO_ZERO))) {
/* Algorithm for round down, positive < 1.0:
   Add 1 character to the format.
   call sprintf.
   If the extra character was 0, we need to do more work:
   Get a whole bunch more characters.  For now, 20 more
   lop it off.
*/
        if (tmp[2 + prec] == '0') {
          writefmt(fmt, prec + 20, 'f', is_quad);
          if (!is_quad)
            j = snprintf(tmp, sizeof(tmp), fmt, v);
          else
            j = snprintf(tmp, sizeof(tmp), fmt, lv);
        }
        tmp[2 + prec] = '\0';
        *decpt = 0;
        return tmp + 2;
      }
    } else {
      /* Handle special cases, zero prec, quickly without a call */
      /* Order is important in these cases below... */
      tmp[1] = '\0';
      *decpt = 1;
#ifdef TARGET_SUPPORTS_QUADFP
      if (is_quad) {
        if ((ieeeq_v.i[SUBSCRIPT_3] == 0x0) &&
            (ieeeq_v.i[SUBSCRIPT_2] == 0x0) && (ieeeq_v.i[1] == 0x0) &&
            (ieeeq_v.i[0] == 0x0)) {
          /* Always zero */
          tmp[0] = '0';
        } else if (round == FIO_UP) {
          tmp[0] = (*sign) ? '0' : '1';
        } else if (round == FIO_DOWN) {
          tmp[0] = (*sign) ? '1' : '0';
        } else if (round == FIO_ZERO) {
          tmp[0] = '0';
        } else if (round == FIO_COMPATIBLE) {
          tmp[0] = (ieeeq_v.i[SUBSCRIPT_3] < 0x3ffe0000) ? '0' : '1';
        } else if ((ieeeq_v.i[SUBSCRIPT_3] == 0x3ffe0000) &&
                   (ieeeq_v.i[SUBSCRIPT_2] == 0x0 && (ieeeq_v.i[1] == 0x0) &&
                    (ieeeq_v.i[0] == 0x0))) {
          tmp[0] = '0';
        } else {
          tmp[0] = (ieeeq_v.i[SUBSCRIPT_3] < 0x3ffe0000) ? '0' : '1';
        }
      } else
#endif
      {
        if ((ieee_v.i[1] == 0x0) && (ieee_v.i[0] == 0x0)) {
          /* Always zero */
          tmp[0] = '0';
        } else if (round == FIO_UP) {
          tmp[0] = (*sign) ? '0' : '1';
        } else if (round == FIO_DOWN) {
          tmp[0] = (*sign) ? '1' : '0';
        } else if (round == FIO_ZERO) {
          tmp[0] = '0';
        } else if (round == FIO_COMPATIBLE) {
          tmp[0] = (ieee_v.i[1] < 0x3fe00000) ? '0' : '1';
        } else if ((ieee_v.i[1] == 0x3fe00000) && (ieee_v.i[0] == 0x0)) {
          tmp[0] = '0';
        } else {
          tmp[0] = (ieee_v.i[1] < 0x3fe00000) ? '0' : '1';
        }
      }
      return tmp;
    }
  }
  return NULL;
}

/* Below is code that supports IEEE128 versions of ecvt and fcvt called
 * __fortio_lldecvt and __fortio_lldfcvt
 */

/* copy n bytes from q to p */
#define BYTCOPY(p, q, n) memcpy((char *)(p), (char *)(q), n)

/*
 * floating point error codes
 */
#define FPE_NOERR FP_NOERR                /* Everything OK */
#define FPE_FPOVF FP_OVERFLOW_ERROR       /* floating point overflow */
#define FPE_FPUNF FP_UNDERFLOW_ERROR      /* floating point underflow */
#define FPE_IOVF  FP_OVERFLOW_ERROR       /* integer overflow (fix/dfix only) */
#define FPE_INVOP FP_ILLEGAL_INPUT_OR_NAN /* invalid operand */
#define FPE_DIVZ  FP_OVERFLOW_ERROR       /* reciprocal of zero */

/* Number of 16 bit words in external x type format */
#define NE 10

/* Number of 16 bit words in internal format */
#define NI (NE + 3)

/* Array offset to exponent */
#define E 1

/* Array offset to high guard word */
#define M 2

/* Number of bits of precision */
#define NBITS ((NI - 4) * 16)

/*
 * Maximum number of decimal digits in ASCII conversion = NBITS*log10(2)
 */
#define NDEC (NBITS * 8 / 27)

/* The exponent of 1.0 */
#define EXONE (0x3fff)

typedef unsigned short int USHORT;

typedef USHORT U[NE];

typedef int IEEE128[4]; /* IEEE quad precision float number */

extern void etoasc(USHORT *x, char *string, int ndigs, char let);
extern void e113toe(IEEE128 pe, USHORT *y);

/*  etypdat is defined below */
extern struct etypdat_tag {
  /*
   * Control for rounding precision. This can be set to 80 (if NE=6), 64, 56,
   * 53, or 24 bits. -- lfm, added 48 and 96 for Cray arithmetic.
   */
  int rndprc;
  /* Divide significands */
  USHORT equot[NI];
  /***  A few constants: ***/
  USHORT ezero[NE];
  USHORT ehalf[NE];
  USHORT eone[NE];
  USHORT eewo[NE];
  USHORT e32[NE];
  USHORT elog2[NE];
  USHORT esqrt2[NE];
  USHORT epi[NE];
  USHORT eeul[NE];
  int inf_ok;
} etypdat;

#define NANS
#define SING FPE_DIVZ
#define DOMAIN FPE_INVOP
#define UNDERFLOW FPE_FPUNF
#define OVERFLOW FPE_FPOVF

/* NaN's require infinity support. */

char *
__fortio_lldecvt(int num[4], int ndigit, int *decpt, int *sign)
{
  /* This is the ecvt equivalent for a quad precision double. This
   * function uses cprintf which is defined in scutil.a
   */

  U u;
  int i, j;
  char b1[512];
  char *c;
  int e;
  static char b2[512];

  if (ndigit <= 0) {
    *sign = 0;
    *decpt = -1;
    b2[0] = '\0';
    return b2;
  }

  e113toe(num, u);
  etoasc(u, b1, ndigit, 'E');

  for (c = b1; isspace(*c); ++c)
    ; /* skip leading whitespace */
  if (isalpha(*c)) {
    /* probably 'Inf' or 'NaN' */
    strcpy(b2, c);
    *decpt = *sign = 0;
    return b2;
  }
  if (*c == '-') {
    *sign = 1;
    ++c;
  } else {
    *sign = 0;
  }
  for (*decpt = i = j = 0; c[i] != '\0' && i < 511; ++i) {
    /* Remove '.', 'E', set sign and decpt */
    if (c[i] == 'E') {
      if (c[i + 1] == '+' || c[i + 1] == '-') {
        e = atoi(c + i + 1);
        *decpt = e + 1;
      }
      break;
    }
    if (c[i] == '-') {
      *sign = 1;
      continue;
    }
    if (c[i] == '.') {
      continue;
    }
    b2[j++] = c[i];
  }
  b2[j] = '\0';
  if (j > ndigit)
    b2[ndigit] = '\0';

  return b2;
}

char *
__fortio_lldfcvt(int num[4], int ndigit, int *decpt, int *sign)
{
  /* This is the fcvt equivalent for a quad precision double. */

  if (ndigit <= 0) {
    static char b[1] = {'\0'};
    *sign = 0;
    *decpt = -1;
    return b;
  }

  __fortio_lldecvt(num, ndigit, decpt, sign);
  return __fortio_lldecvt(num, ndigit + *decpt, decpt, sign);
}

/* prototypes for e113toe, etoasc, and its dependencies */
extern int eisnan(USHORT x[]);
extern void emov(USHORT *a, USHORT *b);
extern int ecmp(USHORT *a, USHORT *b);
extern void ediv(USHORT *a, USHORT *b, USHORT *c);
extern void efloor(USHORT *x, USHORT *y);
extern void emul(USHORT *a, USHORT *b, USHORT *c);
extern void emovi(USHORT *a, USHORT *b);
extern int emovo(USHORT *a, USHORT *b);
extern void emovz(USHORT *a, USHORT *b);
extern void eiremain(USHORT *den, USHORT *num);
extern void eaddm(USHORT *x, USHORT *y);
extern void eshdn1(USHORT *x);
extern void eshup1(USHORT *x);
extern void ecleaz(USHORT *xi);
extern int eshift(USHORT *x, int sc);
extern void eclear(USHORT *x);
extern void einfin(USHORT *x);
extern void eneg(USHORT *x);
extern void enan(void *out, int size);
extern int enormlz(USHORT *x);
extern int eisinf(USHORT *x);
extern int eisneg(USHORT *x);
extern int edivm(USHORT *den, USHORT *num);
extern void emdnorm(USHORT *s, int lost, int subflg, INT exp, int rcntrl);
extern void esub(USHORT *a, USHORT *b, USHORT *c);
extern int emulm(USHORT *a, USHORT *b);
extern int eiisnan(USHORT x[]);
extern int ecmpm(USHORT *a, USHORT *b);
extern void esubm(USHORT *x, USHORT *y);
extern void eshup8(USHORT *x);
extern void eshup6(USHORT *x);
extern void eshdn8(USHORT *x);
extern void eshdn6(USHORT *x);
extern void m16m(USHORT a, USHORT *b, USHORT *c);
extern void ecleazs(USHORT *xi);

void
e113toe(IEEE128 pe, USHORT *y)
{
  USHORT r;
  USHORT *p;
  USHORT yy[NI];

  ecleaz(yy);
  r = (UINT)pe[0] >> 16;
  yy[0] = 0;
  if (r & 0x8000)
    yy[0] = 0xffff;
  r &= 0x7fff;
  yy[E] = r;
  p = &yy[M + 1];
  *p++ = pe[0] & 0xFFFF;
  *p++ = (UINT)pe[1] >> 16;
  *p++ = pe[1] & 0xFFFF;
  *p++ = (UINT)pe[2] >> 16;
  *p++ = pe[2] & 0xFFFF;
  *p++ = (UINT)pe[3] >> 16;
  *p++ = pe[3] & 0xFFFF;
  /* If denormal, remove the implied bit; else shift down 1. */
  if (r == 0) {
    yy[M] = 0;
  } else {
    yy[M] = 1;
    (void)eshift(yy, -1);
  }
  (void)emovo(yy, y);
}

#define NTEN 12
#define MAXP 4096

static USHORT etens[NTEN + 1][NE] = {
    {
        0x6576, 0x4a92, 0x804a, 0x153f, 0xc94c, 0x979a, 0x8a20, 0x5202, 0xc460,
        0x7525,
    }, /* 10**4096 */
    {
        0x6a32, 0xce52, 0x329a, 0x28ce, 0xa74d, 0x5de4, 0xc53d, 0x3b5d, 0x9e8b,
        0x5a92,
    }, /* 10**2048 */
    {
        0x526c, 0x50ce, 0xf18b, 0x3d28, 0x650d, 0x0c17, 0x8175, 0x7586, 0xc976,
        0x4d48,
    },
    {
        0x9c66, 0x58f8, 0xbc50, 0x5c54, 0xcc65, 0x91c6, 0xa60e, 0xa0ae, 0xe319,
        0x46a3,
    },
    {
        0x851e, 0xeab7, 0x98fe, 0x901b, 0xddbb, 0xde8d, 0x9df9, 0xebfb, 0xaa7e,
        0x4351,
    },
    {
        0x0235, 0x0137, 0x36b1, 0x336c, 0xc66f, 0x8cdf, 0x80e9, 0x47c9, 0x93ba,
        0x41a8,
    },
    {
        0x50f8, 0x25fb, 0xc76b, 0x6b71, 0x3cbf, 0xa6d5, 0xffcf, 0x1f49, 0xc278,
        0x40d3,
    },
    {
        0x0000, 0x0000, 0x0000, 0x0000, 0xf020, 0xb59d, 0x2b70, 0xada8, 0x9dc5,
        0x4069,
    },
    {
        0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0400, 0xc9bf, 0x8e1b,
        0x4034,
    },
    {
        0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x2000, 0xbebc,
        0x4019,
    },
    {
        0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x9c40,
        0x400c,
    },
    {
        0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0xc800,
        0x4005,
    },
    {
        0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0xa000,
        0x4002,
    }, /* 10**1 */
};

static USHORT emtens[NTEN + 1][NE] = {
    {
        0x2030, 0xcffc, 0xa1c3, 0x8123, 0x2de3, 0x9fde, 0xd2ce, 0x04c8, 0xa6dd,
        0x0ad8,
    }, /* 10**-4096 */
    {
        0x8264, 0xd2cb, 0xf2ea, 0x12d4, 0x4925, 0x2de4, 0x3436, 0x534f, 0xceae,
        0x256b,
    }, /* 10**-2048 */
    {
        0xf53f, 0xf698, 0x6bd3, 0x0158, 0x87a6, 0xc0bd, 0xda57, 0x82a5, 0xa2a6,
        0x32b5,
    },
    {
        0xe731, 0x04d4, 0xe3f2, 0xd332, 0x7132, 0xd21c, 0xdb23, 0xee32, 0x9049,
        0x395a,
    },
    {
        0xa23e, 0x5308, 0xfefb, 0x1155, 0xfa91, 0x1939, 0x637a, 0x4325, 0xc031,
        0x3cac,
    },
    {
        0xe26d, 0xdbde, 0xd05d, 0xb3f6, 0xac7c, 0xe4a0, 0x64bc, 0x467c, 0xddd0,
        0x3e55,
    },
    {
        0x2a20, 0x6224, 0x47b3, 0x98d7, 0x3f23, 0xe9a5, 0xa539, 0xea27, 0xa87f,
        0x3f2a,
    },
    {
        0x0b5b, 0x4af2, 0xa581, 0x18ed, 0x67de, 0x94ba, 0x4539, 0x1ead, 0xcfb1,
        0x3f94,
    },
    {
        0xbf71, 0xa9b3, 0x7989, 0xbe68, 0x4c2e, 0xe15b, 0xc44d, 0x94be, 0xe695,
        0x3fc9,
    },
    {
        0x3d4d, 0x7c3d, 0x36ba, 0x0d2b, 0xfdc2, 0xcefc, 0x8461, 0x7711, 0xabcc,
        0x3fe4,
    },
    {
        0xc155, 0xa4a8, 0x404e, 0x6113, 0xd3c3, 0x652b, 0xe219, 0x1758, 0xd1b7,
        0x3ff1,
    },
    {
        0xd70a, 0x70a3, 0x0a3d, 0xa3d7, 0x3d70, 0xd70a, 0x70a3, 0x0a3d, 0xa3d7,
        0x3ff8,
    },
    {
        0xcccd, 0xcccc, 0xcccc, 0xcccc, 0xcccc, 0xcccc, 0xcccc, 0xcccc, 0xcccc,
        0x3ffb,
    }, /* 10**-1 */
};

void
etoasc(USHORT *x, char *string, int ndigs, char let)
{
  INT digit;
  USHORT y[NI], t[NI], u[NI], w[NI];
  USHORT *p, *r, *ten;
  USHORT sign;
  int i, j, k, expon, rndsav;
  char *s, *ss;
  USHORT m;

  rndsav = etypdat.rndprc;
  etypdat.rndprc = NBITS; /* set to full precision */
  emov(x, y);             /* retain external format */
  if (y[NE - 1] & 0x8000) {
    sign = 0xffff;
    y[NE - 1] &= 0x7fff;
  } else {
    sign = 0;
  }
  expon = 0;
  ten = &etens[NTEN][0];
  emov(etypdat.eone, t);
  /* Test for zero exponent */
  if (y[NE - 1] == 0) {
    for (k = 0; k < NE - 1; k++) {
      if (y[k] != 0)
        goto tnzro; /* denormalized number */
    }
    goto isone; /* legal all zeros */
  }
tnzro:

  /*
   * Test for infinity.
   */
  if (y[NE - 1] == 0x7fff) {
    if (sign)
      sprintf(string, " -Infinity ");
    else
      sprintf(string, " Infinity ");
    goto bxit;
  }

  /*
   * Test for exponent nonzero but significand denormalized. This is an
   * error condition.
   */
  if ((y[NE - 1] != 0) && ((y[NE - 2] & 0x8000) == 0)) {
    mtherr("etoasc", DOMAIN);
    sprintf(string, "NaN");
    goto bxit;
  }

  /* Compare to 1.0 */
  i = ecmp(etypdat.eone, y);
  if (i == 0)
    goto isone;

  if (i < 0) { /* Number is greater than 1 */
               /*
                * Convert significand to an integer and strip trailing decimal
                * zeros.
                */
    emov(y, u);
    u[NE - 1] = EXONE + NBITS - 1;

    p = &etens[NTEN - 4][0];
    m = 16;
    do {
      ediv(p, u, t);
      efloor(t, w);
      for (j = 0; j < NE - 1; j++) {
        if (t[j] != w[j])
          goto noint;
      }
      emov(t, u);
      expon += (int)m;
    noint:
      p += NE;
      m >>= 1;
    } while (m != 0);

    /* Rescale from integer significand */
    u[NE - 1] += y[NE - 1] - (unsigned int)(EXONE + NBITS - 1);
    emov(u, y);
    /* Find power of 10 */
    emov(etypdat.eone, t);
    m = MAXP;
    p = &etens[0][0];
    while (ecmp(ten, u) <= 0) {
      if (ecmp(p, u) <= 0) {
        ediv(p, u, u);
        emul(p, t, t);
        expon += (int)m;
      }
      m >>= 1;
      if (m == 0)
        break;
      p += NE;
    }
  } else { /* Number is less than 1.0 */
    /* Pad significand with trailing decimal zeros. */
    if (y[NE - 1] == 0) {
      while ((y[NE - 2] & 0x8000) == 0) {
        emul(ten, y, y);
        expon -= 1;
      }
    } else {
      emovi(y, w);
      for (i = 0; i < NDEC + 1; i++) {
        if ((w[NI - 1] & 0x7) != 0)
          break;
        /* multiply by 10 */
        emovz(w, u);
        eshdn1(u);
        eshdn1(u);
        eaddm(w, u);
        u[1] += 3;
        while (u[2] != 0) {
          eshdn1(u);
          u[1] += 1;
        }
        if (u[NI - 1] != 0)
          break;
        if (etypdat.eone[NE - 1] <= u[1])
          break;
        emovz(u, w);
        expon -= 1;
      }
      (void)emovo(w, y);
    }
    k = -MAXP;
    p = &emtens[0][0];
    r = &etens[0][0];
    emov(y, w);
    emov(etypdat.eone, t);
    while (ecmp(etypdat.eone, w) > 0) {
      if (ecmp(p, w) >= 0) {
        emul(r, w, w);
        emul(r, t, t);
        expon += k;
      }
      k /= 2;
      if (k == 0)
        break;
      p += NE;
      r += NE;
    }
    ediv(t, etypdat.eone, t);
  }
isone:
  /* Find the first (leading) digit. */
  emovi(t, w);
  emovz(w, t);
  emovi(y, w);
  emovz(w, y);
  eiremain(t, y);
  digit = etypdat.equot[NI - 1];
  while ((digit == 0) && (ecmp(y, etypdat.ezero) != 0)) {
    eshup1(y);
    emovz(y, u);
    eshup1(u);
    eshup1(u);
    eaddm(u, y);
    eiremain(t, y);
    digit = etypdat.equot[NI - 1];
    expon -= 1;
  }
  s = string;
  if (sign)
    *s++ = '-';
  else
    *s++ = ' ';
  /* Examine number of digits requested by caller. */
  if (ndigs < 0)
    ndigs = 0;
  if (ndigs > NDEC)
    ndigs = NDEC;
  if (digit == 10) {
    *s++ = '1';
    *s++ = '.';
    if (ndigs > 0) {
      *s++ = '0';
      ndigs -= 1;
    }
    expon += 1;
  } else {
    *s++ = (char)digit + '0';
    *s++ = '.';
  }
  /* Generate digits after the decimal point. */
  for (k = 0; k <= ndigs; k++) {
    /* multiply current number by 10, without normalizing */
    eshup1(y);
    emovz(y, u);
    eshup1(u);
    eshup1(u);
    eaddm(u, y);
    eiremain(t, y);
    *s++ = (char)etypdat.equot[NI - 1] + '0';
  }
  digit = etypdat.equot[NI - 1];
  --s;
  ss = s;
  /* round off the ASCII string */
  if (digit > 4) {
    /* Test for critical rounding case in ASCII output. */
    if (digit == 5) {
      (void)emovo(y, t);
      if (ecmp(t, etypdat.ezero) != 0)
        goto roun; /* round to nearest */
      if ((*(s - 1) & 1) == 0)
        goto doexp; /* round to even */
    }
  /* Round up and propagate carry-outs */
  roun:
    --s;
    k = *s & 0x7f;
    /* Carry out to most significant digit? */
    if (k == '.') {
      --s;
      k = *s;
      k += 1;
      *s = (char)k;
      /* Most significant digit carries to 10? */
      if (k > '9') {
        expon += 1;
        *s = '1';
      }
      goto doexp;
    }
    /* Round up and carry out from less significant digits */
    k += 1;
    *s = (char)k;
    if (k > '9') {
      *s = '0';
      goto roun;
    }
  }
doexp:
  *ss++ = let;
  if (expon >= 0)
    *ss++ = '+';
  sprintf(ss, "%d", expon);
bxit:
  etypdat.rndprc = rndsav;
}

struct etypdat_tag etypdat = {
    /* rndprc */
    NBITS,
    /* equot */
    {0},
    /* ezero 0.0 */
    {
        0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
        0x0000,
    },

    /* ehalf 5.0E-1 */
    {
        0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x8000,
        0x3ffe,
    },

    /* eone 1.0E0 */
    {
        0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x8000,
        0x3fff,
    },

    /* etwo  2.0E0 */
    {
        0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x8000,
        0x4000,
    },

    /* e32 3.2E1 */
    {
        0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x8000,
        0x4004,
    },

    /* elog2 6.93147180559945309417232121458176568075500134360255E-1 */
    {
        0x40f3, 0xf6af, 0x03f2, 0xb398, 0xc9e3, 0x79ab, 0150717, 0013767,
        0130562, 0x3ffe,
    },

    /* esqrt2 1.41421356237309504880168872420969807856967187537695E0 */
    {
        0x1d6f, 0xbe9f, 0x754a, 0x89b3, 0x597d, 0x6484, 0174736, 0171463,
        0132404, 0x3fff,
    },

    /* epi 3.14159265358979323846264338327950288419716939937511E0 */
    {
        0x2902, 0x1cd1, 0x80dc, 0x628b, 0xc4c6, 0xc234, 0020550, 0155242,
        0144417, 0040000,
    },

    /* eeul 5.7721566490153286060651209008240243104215933593992E-1 */
    {
        0xd1be, 0xc7a4, 0076660, 0063743, 0111704, 0x3ffe,
    },

    /* inf_ok */
    0};

static void
mtherr(const char *s, int c)
{
  fperror(c);
}

static void
fperror(int x)
{
  switch (x) {
  case FP_NOERR:
    break;
  case FP_ILLEGAL_INPUT_OR_NAN:
    printf("illegal input or NaN error\n");
    break;
  case FP_OVERFLOW_ERROR:
    printf("overflow error\n");
    break;
  case FP_UNDERFLOW_ERROR:
    printf("underflow error\n");
    break;
  default:
    printf("unknown error\n");
  }
}

/*
 * Check if e-type number is not a number.
 */
int
eisnan(USHORT *x)
{

  return (0);
}

/*
 * Move external format number from a to b.
 *
 * emov( a, b );
 */

void
emov(USHORT *a, USHORT *b)
{
  int i;

  for (i = 0; i < NE; i++)
    *b++ = *a++;
}

/*
 * Compare two e type numbers.
 *
 *
 * returns:
 *      +1 if a > b
 *       0 if a == b
 *      -1 if a < b
 *      -2 if either a or b is a NaN.
 */
int
ecmp(USHORT *a, USHORT *b)
{
  USHORT ai[NI], bi[NI];
  USHORT *p, *q;
  int i;
  int msign;

  emovi(a, ai);
  p = ai;
  emovi(b, bi);
  q = bi;

  if (*p != *q) { /* the signs are different */
    /* -0 equals + 0 */
    for (i = 1; i < NI - 1; i++) {
      if (ai[i] != 0)
        goto nzro;
      if (bi[i] != 0)
        goto nzro;
    }
    return (0);
  nzro:
    if (*p == 0)
      return (1);
    else
      return (-1);
  }
  /* both are the same sign */
  if (*p == 0)
    msign = 1;
  else
    msign = -1;
  i = NI - 1;
  do {
    if (*p++ != *q++) {
      goto diff;
    }
  } while (--i > 0);

  return (0); /* equality */

diff:

  if (*(--p) > *(--q))
    return (msign); /* p is bigger */
  else
    return (-msign); /* p is littler */
}

/*
 * ediv( a, b, c ); c = b / a
 */
void
ediv(USHORT *a, USHORT *b, USHORT *c)
{
  USHORT ai[NI], bi[NI];
  int i;
  INT lt, lta, ltb;

/* Infinity over anything else is infinity. */
  emovi(a, ai);
  emovi(b, bi);
  lta = ai[E];
  ltb = bi[E];
  if (bi[E] == 0) { /* See if numerator is zero. */
    for (i = 1; i < NI - 1; i++) {
      if (bi[i] != 0) {
        ltb -= enormlz(bi);
        goto dnzro1;
      }
    }
    eclear(c);
    return;
  }
dnzro1:

  if (ai[E] == 0) { /* possible divide by zero */
    for (i = 1; i < NI - 1; i++) {
      if (ai[i] != 0) {
        lta -= enormlz(ai);
        goto dnzro2;
      }
    }
    if (ai[0] == bi[0])
      *(c + (NE - 1)) = 0;
    else
      *(c + (NE - 1)) = 0x8000;
    einfin(c);
    mtherr("ediv", SING);
    return;
  }
dnzro2:

  i = edivm(ai, bi);
  /* calculate exponent */
  lt = ltb - lta + EXONE;
  emdnorm(bi, i, 0, lt, 64);
  /* set the sign */
  if (ai[0] == bi[0])
    bi[0] = 0;
  else
    bi[0] = 0Xffff;
  (void)emovo(bi, c);
}

/*
 * y = largest integer not greater than x (truncated toward minus infinity)
 *
 * USHORT x[NE], y[NE]
 *
 * efloor( x, y );
 */
static USHORT bmask[] = {
    0xffff, 0xfffe, 0xfffc, 0xfff8, 0xfff0, 0xffe0, 0xffc0, 0xff80, 0xff00,
    0xfe00, 0xfc00, 0xf800, 0xf000, 0xe000, 0xc000, 0x8000, 0x0000,
};

void
efloor(USHORT *x, USHORT *y)
{
  USHORT *p;
  int e, expon, i;
  USHORT f[NE];

  emov(x, f); /* leave in external format */
  expon = (int)f[NE - 1];
  e = (expon & 0x7fff) - (EXONE - 1);
  if (e <= 0) {
    eclear(y);
    goto isitneg;
  }
  /* number of bits to clear out */
  e = NBITS - e;
  emov(f, y);
  if (e <= 0)
    return;

  p = &y[0];
  while (e >= 16) {
    *p++ = 0;
    e -= 16;
  }
  /* clear the remaining bits */
  *p &= bmask[e];
/* truncate negatives toward minus infinity */
isitneg:

  if ((USHORT)expon & (USHORT)0x8000) {
    for (i = 0; i < NE - 1; i++) {
      if (f[i] != y[i]) {
        esub(etypdat.eone, y, y);
        break;
      }
    }
  }
}

/*
 * emul( a, b, c ); c = b * a
 */
void
emul(USHORT *a, USHORT *b, USHORT *c)
{
  USHORT ai[NI], bi[NI];
  int i, j;
  INT lt, lta, ltb;

/* Infinity times anything else is infinity. */
  emovi(a, ai);
  emovi(b, bi);
  lta = ai[E];
  ltb = bi[E];
  if (ai[E] == 0) {
    for (i = 1; i < NI - 1; i++) {
      if (ai[i] != 0) {
        lta -= enormlz(ai);
        goto mnzer1;
      }
    }
    eclear(c);
    return;
  }
mnzer1:

  if (bi[E] == 0) {
    for (i = 1; i < NI - 1; i++) {
      if (bi[i] != 0) {
        ltb -= enormlz(bi);
        goto mnzer2;
      }
    }
    eclear(c);
    return;
  }
mnzer2:

  /* Multiply significands */
  j = emulm(ai, bi);
  /* calculate exponent */
  lt = lta + ltb - (EXONE - 1);
  emdnorm(bi, j, 0, lt, 64);
  /* calculate sign of product */
  if (ai[0] == bi[0])
    bi[0] = 0;
  else
    bi[0] = 0xffff;
  (void)emovo(bi, c);
}

/*
 * Move in external format number, converting it to internal format.
 */
void
emovi(USHORT *a, USHORT *b)
{
  USHORT *p, *q;
  int i;

  q = b;
  p = a + (NE - 1); /* point to last word of external number */
  /* get the sign bit */
  if (*p & 0x8000)
    *q++ = 0xffff;
  else
    *q++ = 0;
  /* get the exponent */
  *q = *p--;
  *q++ &= 0x7fff; /* delete the sign bit */
  /* clear high guard word */
  *q++ = 0;
  /* move in the significand */
  for (i = 0; i < NE - 1; i++)
    *q++ = *p--;
  /* clear low guard word */
  *q = 0;
}

/*
 * Move internal format number out, converting it to external format.
 */
int
emovo(USHORT *a, USHORT *b)
{
  USHORT *p, *q;
  int i;

  p = a;
  q = b + (NE - 1); /* point to output exponent */
  /* combine sign and exponent */
  i = *p++;
  if (i)
    *q-- = *p++ | 0x8000;
  else
    *q-- = *p++;
  /* skip over guard word */
  ++p;
  /* move the significand */
  for (i = 0; i < NE - 1; i++)
    *q-- = *p++;
  return 0;
}

/*
 * Move internal format number from a to b.
 */
void
emovz(USHORT *a, USHORT *b)
{
  int i;

  for (i = 0; i < NI - 1; i++)
    *b++ = *a++;
  /* clear low guard word */
  *b = 0;
}

void
eiremain(USHORT *den, USHORT *num)
{
  INT ld, ln;
  USHORT j;

  ld = den[E];
  ld -= enormlz(den);
  ln = num[E];
  ln -= enormlz(num);
  ecleaz(etypdat.equot);
  while (ln >= ld) {
    if (ecmpm(den, num) <= 0) {
      esubm(den, num);
      j = 1;
    } else {
      j = 0;
    }
    eshup1(etypdat.equot);
    etypdat.equot[NI - 1] |= j;
    eshup1(num);
    ln -= 1;
  }
  emdnorm(num, 0, 0, ln, 0);
}

/*
 * Add significands ;   x + y replaces y
 */

void
eaddm(USHORT *x, USHORT *y)
{
  UINT a;
  int i;
  unsigned int carry;

  x += NI - 1;
  y += NI - 1;
  carry = 0;
  for (i = M; i < NI; i++) {
    a = (UINT)(*x) + (UINT)(*y) + carry;
    if (a & 0x10000)
      carry = 1;
    else
      carry = 0;
    *y = a & 0xFFFF;
    --x;
    --y;
  }
}

/*
 * Shift significand down by 1 bit
 */

void
eshdn1(USHORT *x)
{
  USHORT bits;
  int i;

  x += M; /* point to significand area */

  bits = 0;
  for (i = M; i < NI; i++) {
    if (*x & 1)
      bits |= 1;
    *x >>= 1;
    if (bits & 2)
      *x |= 0x8000;
    bits <<= 1;
    ++x;
  }
}

/*
 * Shift significand up by 1 bit
 */

void
eshup1(USHORT *x)
{
  USHORT bits;
  int i;

  x += NI - 1;
  bits = 0;

  for (i = M; i < NI; i++) {
    if (*x & 0x8000)
      bits |= 1;
    *x <<= 1;
    *x &= 0xFFFF;
    if (bits & 2)
      *x |= 1;
    bits <<= 1;
    --x;
  }
}

/*
 * Clear out internal format number.
 */

void
ecleaz(USHORT *xi)
{
  int i;

  for (i = 0; i < NI; i++)
    *xi++ = 0;
}

/*
 * Shift significand: Shifts significand area up or down by the
 * number of bits given by the variable sc.
 */
int
eshift(USHORT *x, int sc)
{
  USHORT lost;
  USHORT *p;

  if (sc == 0)
    return (0);

  lost = 0;
  p = x + NI - 1;

  if (sc < 0) {
    sc = -sc;
    while (sc >= 16) {
      lost |= *p; /* remember lost bits */
      eshdn6(x);
      sc -= 16;
    }

    while (sc >= 8) {
      lost |= *p & 0xff;
      eshdn8(x);
      sc -= 8;
    }

    while (sc > 0) {
      lost |= *p & 1;
      eshdn1(x);
      sc -= 1;
    }
  } else {
    while (sc >= 16) {
      eshup6(x);
      sc -= 16;
    }

    while (sc >= 8) {
      eshup8(x);
      sc -= 8;
    }

    while (sc > 0) {
      eshup1(x);
      sc -= 1;
    }
  }
  if (lost)
    lost = 1;
  return ((int)lost);
}

/*
 * Clear out entire external format number.
 */

void
eclear(USHORT *x)
{
  int i;

  for (i = 0; i < NE; i++)
    *x++ = 0;
}

/*
 * Fill entire number, including exponent and significand, with largest
 * possible number.  These programs implement a saturation value that is an
 * ordinary, legal number.  A special value "infinity" may also be
 * implemented; this would require tests for that value and implementation
 * of special rules for arithmetic operations involving inifinity.
 */

void
einfin(USHORT *x)
{
  int i;

  for (i = 0; i < NE - 1; i++)
    *x++ = 0xffff;
  *x |= 32766;
  if (etypdat.rndprc < NBITS) {
    if (etypdat.rndprc == 113) {
      *(x - 9) = 0;
      *(x - 8) = 0;
    }
    if (etypdat.rndprc == 64) {
      *(x - 5) = 0;
    }
    if (etypdat.rndprc == 53) {
      *(x - 4) = 0xf800;
    } else {
      *(x - 4) = 0;
      *(x - 3) = 0;
      *(x - 2) = 0xff00;
    }
  }
}

/*
 * Negate external format number
 */

void
eneg(USHORT *x)
{

  x[NE - 1] ^= 0x8000; /* Toggle the sign bit */
}

/*
 * NaN bit patterns
 */
static IEEE128 nan113 = {0x7fffffff, (int)0xffffffff, (int)0xffffffff,
                         (int)0xffffffff};
static int nan53[2] = {0x7fffffff, (int)0xffffffff};
static int nan24 = 0x7fffffff;

void
enan(void *out, int size)
{
  int i, n;
  void *p;
  USHORT *nan;

  nan = out;
  switch (size) {
  case 113:
    n = sizeof(nan113);
    p = nan113;
    break;

  case 53:
    n = sizeof(nan53);
    p = nan53;
    break;

  case 24:
    n = sizeof(nan24);
    p = &nan24;
    break;

  case NBITS:
    for (i = 0; i < NE - 2; i++)
      *nan++ = 0;
    *nan++ = 0xc000;
    *nan++ = 0x7fff;
    return;

  case NI * 16:
    *nan++ = 0;
    *nan++ = 0x7fff;
    *nan++ = 0;
    *nan++ = 0xc000;
    for (i = 4; i < NI; i++)
      *nan++ = 0;
    return;
  default:
    mtherr("enan", DOMAIN);
    return;
  }
  BYTCOPY(out, p, n);
}

/*
 * normalize.  Shift normalizes the significand area pointed to by
 * argument. Shift count (up = positive) is returned.
 */
int
enormlz(USHORT *x)
{
  USHORT *p;
  int sc;

  sc = 0;
  p = &x[M];
  if (*p != 0)
    goto normdn;
  ++p;
  if (*p & 0x8000)
    return (0); /* already normalized */
  while (*p == 0) {
    eshup6(x);
    sc += 16;
    /*
     * With guard word, there are NBITS+16 bits available. return true if
     * all are zero.
     */
    if (sc > NBITS)
      return (sc);
  }
  /* see if high byte is zero */
  while ((*p & 0xff00) == 0) {
    eshup8(x);
    sc += 8;
  }
  /* now shift 1 bit at a time */
  while ((*p & 0x8000) == 0) {
    eshup1(x);
    sc += 1;
    if (sc > (NBITS + 16)) {
      mtherr("enormlz", UNDERFLOW);
      return (sc);
    }
  }
  return (sc);

/*
 * Normalize by shifting down out of the high guard word of the
 * significand
 */
normdn:

  if (*p & 0xff00) {
    eshdn8(x);
    sc -= 8;
  }
  while (*p != 0) {
    eshdn1(x);
    sc -= 1;

    if (sc < -NBITS) {
      mtherr("enormlz", OVERFLOW);
      return (sc);
    }
  }
  return (sc);
}

/*
 * Return 1 if external format number has maximum possible exponent, else
 * return zero.
 */
int
eisinf(USHORT *x)
{

  if ((x[NE - 1] & 0x7fff) == 0x7fff) {
    return (1);
  } else
    return (0);
}

/*
 * Return 1 if external format number is negative, else return zero.
 */
int
eisneg(USHORT *x)
{

  if (x[NE - 1] & 0x8000)
    return (1);
  else
    return (0);
}

/*
 * Divide significands. Neither the numerator nor the denominator is
 * permitted to have its high guard word nonzero.
 */

int
edivm(USHORT *den, USHORT *num)
{
  int i;
  USHORT *p;
  UINT tnum;
  USHORT j, tdenm, tquot;
  USHORT tprod[NI + 1];

  p = &etypdat.equot[0];
  *p++ = num[0];
  *p++ = num[1];

  for (i = M; i < NI; i++) {
    *p++ = 0;
  }
  eshdn1(num);
  tdenm = den[M + 1];
  for (i = M; i < NI; i++) {
    /* Find trial quotient digit (the radix is 65536). */
    tnum = (((UINT)num[M]) << 16) + num[M + 1];

    /* Do not execute the divide instruction if it will overflow. */
    if ((unsigned long)(tdenm * 0xffffL) < (unsigned long)tnum)
      tquot = 0xffff;
    else
      tquot = (tnum / tdenm) & 0xFFFF;

    /* Multiply denominator by trial quotient digit. */
    m16m(tquot, den, tprod);
    /* The quotient digit may have been overestimated. */
    if (ecmpm(tprod, num) > 0) {
      tquot -= 1;
      tquot &= 0xFFFF;
      esubm(den, tprod);
      if (ecmpm(tprod, num) > 0) {
        tquot -= 1;
        tquot &= 0xFFFF;
        esubm(den, tprod);
      }
    }
    esubm(tprod, num);
    etypdat.equot[i] = tquot;
    eshup6(num);
  }
  /* test for nonzero remainder after roundoff bit */
  p = &num[M];
  j = 0;
  for (i = M; i < NI; i++) {
    j |= *p++;
  }
  if (j)
    j = 1;

  for (i = 0; i < NI; i++)
    num[i] = etypdat.equot[i];

  return ((int)j);
}

/*
 * Normalize and round off.
 *
 * The internal format number to be rounded is "s". Input "lost" indicates
 * whether the number is exact. This is the so-called sticky bit.
 *
 * Input "subflg" indicates whether the number was obtained by a subtraction
 * operation.  In that case if lost is nonzero then the number is slightly
 * smaller than indicated.
 *
 * Input "exp" is the biased exponent, which may be negative. the exponent field
 * of "s" is ignored but is replaced by "exp" as adjusted by normalization
 * and rounding.
 *
 * Input "rcntrl" is the rounding control.
 */

static int rlast = -1;
static int rw = 0;
static USHORT rmsk = 0;
static USHORT rmbit = 0;
static USHORT rebit = 0;
static int re = 0;
static USHORT rbit[NI] = {0, 0, 0, 0, 0, 0, 0, 0};

void
emdnorm(USHORT *s, int lost, int subflg, INT exp, int rcntrl)
{
  int i, j;
  USHORT r;

  /* Normalize */
  j = enormlz(s);

/* a blank significand could mean either zero or infinity. */
  if (j > NBITS) {
    ecleazs(s);
    return;
  }
  exp -= j;
  if (exp >= 32767L)
    goto overf;
  if (exp < 0L) {
    if (exp > (INT)(-NBITS - 1)) {
      j = (int)exp;
      i = eshift(s, j);
      if (i)
        lost = 1;
    } else {
      ecleazs(s);
      return;
    }
  }
  /* Round off, unless told not to by rcntrl. */
  if (rcntrl == 0)
    goto mdfin;
  /* Set up rounding parameters if the control changed. */
  if (etypdat.rndprc != rlast) {
    ecleaz(rbit);
    switch (etypdat.rndprc) {
    default:
    case NBITS:
      rw = NI - 1; /* low guard word */
      rmsk = 0xffff;
      rmbit = 0x8000;
      rebit = 1;
      re = rw - 1;
      break;
    case 113:
      rw = 10;
      rmsk = 0x7fff;
      rmbit = 0x4000;
      rebit = 0x8000;
      re = rw;
      break;
    case 64:
      rw = 7;
      rmsk = 0xffff;
      rmbit = 0x8000;
      rebit = 1;
      re = rw - 1;
      break;
    /* For DEC arithmetic */
    case 56:
      rw = 6;
      rmsk = 0xff;
      rmbit = 0x80;
      rebit = 0x100;
      re = rw;
      break;
    case 53:
      rw = 6;
      rmsk = 0x7ff;
      rmbit = 0x0400;
      rebit = 0x800;
      re = rw;
      break;
    case 96:
      rw = 9;
      rmsk = 0xffff;
      rmbit = 0x8000;
      rebit = 1;
      re = rw - 1;
      break;
    case 48:
      rw = 6;
      rmsk = 0xffff;
      rmbit = 0x8000;
      rebit = 1;
      re = rw - 1;
      break;
    case 24:
      rw = 4;
      rmsk = 0xff;
      rmbit = 0x80;
      rebit = 0x100;
      re = rw;
      break;
    }
    rbit[re] = rebit;
    rlast = etypdat.rndprc;
  }

  /*
   * Shift down 1 temporarily if the data structure has an implied most
   * significant bit and the number is denormal. For etypdat.rndprc = 64 or
   * NBITS,
   * there is no implied bit.
   */
  if ((exp <= 0) && (etypdat.rndprc != 64) && (etypdat.rndprc != NBITS)) {
    lost |= s[NI - 1] & 1;
    eshdn1(s);
  }
  /*
   * Clear out all bits below the rounding bit, remembering in r if any
   * were nonzero.
   */
  r = s[rw] & rmsk;
  if (etypdat.rndprc < NBITS) {
    i = rw + 1;
    while (i < NI) {
      if (s[i])
        r |= 1;
      s[i] = 0;
      ++i;
    }
  }
  s[rw] &= ~rmsk;
  if ((r & rmbit) != 0) {
    if (r == rmbit) {
      if (lost == 0) { /* round to even */
        if ((s[re] & rebit) == 0)
          goto mddone;
      } else {
        if (subflg != 0)
          goto mddone;
      }
    }
    eaddm(rbit, s);
  }
mddone:
  if ((exp <= 0) && (etypdat.rndprc != 64) && (etypdat.rndprc != NBITS) &&
      (etypdat.rndprc != 48) && (etypdat.rndprc != 96)) {
    eshup1(s);
  }
  if (s[2] != 0) { /* overflow on roundoff */
    eshdn1(s);
    exp += 1;
  }
mdfin:
  s[NI - 1] = 0;
  if (exp >= 32767L) {
  overf:
    s[1] = 32766;
    s[2] = 0;
    for (i = M + 1; i < NI - 1; i++)
      s[i] = 0xffff;
    s[NI - 1] = 0;
    if ((etypdat.rndprc < 64) || (etypdat.rndprc == 113)) {
      s[rw] &= ~rmsk;
      if (etypdat.rndprc == 24) {
        s[5] = 0;
        s[6] = 0;
      }
    }
    return;
  }
  if (exp < 0)
    s[1] = 0;
  else
    s[1] = (USHORT)exp;
}

static void eadd1(USHORT *a, USHORT *b, USHORT *c);

/*
 * Subtract external format numbers.
 * esub( a, b, c );      c = b - a
 */

static int subflg = 0;

void
esub(USHORT *a, USHORT *b, USHORT *c)
{

  subflg = 1;
  eadd1(a, b, c);
}

static void
eadd1(USHORT *a, USHORT *b, USHORT *c)
{
  USHORT ai[NI], bi[NI], ci[NI];
  int i, lost, j, k;
  INT lt, lta, ltb;

  emovi(a, ai);
  emovi(b, bi);
  if (subflg)
    ai[0] = ~ai[0] & 0xFFFF;

  /* compare exponents */
  lta = ai[E];
  ltb = bi[E];
  lt = lta - ltb;
  if (lt > 0L) { /* put the larger number in bi */
    emovz(bi, ci);
    emovz(ai, bi);
    emovz(ci, ai);
    ltb = bi[E];
    lt = -lt;
  }
  lost = 0;
  if (lt != 0L) {
    if (lt < (INT)(-NBITS - 1))
      goto done; /* answer same as larger addend */
    k = (int)lt;
    lost = eshift(ai, k); /* shift the smaller number down */
  } else {
    /* exponents were the same, so must compare significands */
    i = ecmpm(ai, bi);
    if (i == 0) { /* the numbers are identical in magnitude */
      /* if different signs, result is zero */
      if (ai[0] != bi[0]) {
        eclear(c);
        return;
      }
      /* if same sign, result is double */
      /* double denomalized tiny number */
      if ((bi[E] == 0) && ((bi[3] & 0x8000) == 0)) {
        eshup1(bi);
        goto done;
      }
      /* add 1 to exponent unless both are zero! */
      for (j = 1; j < NI - 1; j++) {
        if (bi[j] != 0) {
          /* This could overflow, but let emovo take care of that. */
          ltb += 1;
          break;
        }
      }
      bi[E] = (USHORT)ltb;
      goto done;
    }
    if (i > 0) { /* put the larger number in bi */
      emovz(bi, ci);
      emovz(ai, bi);
      emovz(ci, ai);
    }
  }
  if (ai[0] == bi[0]) {
    eaddm(ai, bi);
    subflg = 0;
  } else {
    esubm(ai, bi);
    subflg = 1;
  }
  emdnorm(bi, lost, subflg, ltb, 64);

done:
  (void)emovo(bi, c);
}

/* Multiply significands */
int
emulm(USHORT *a, USHORT *b)
{
  USHORT *p, *q;
  USHORT pprod[NI];
  USHORT j;
  int i;

  etypdat.equot[0] = b[0];
  etypdat.equot[1] = b[1];
  for (i = M; i < NI; i++)
    etypdat.equot[i] = 0;

  j = 0;
  p = &a[NI - 1];
  q = &etypdat.equot[NI - 1];
  for (i = M + 1; i < NI; i++) {
    if (*p == 0) {
      --p;
    } else {
      m16m(*p--, b, pprod);
      eaddm(pprod, etypdat.equot);
    }
    j |= *q;
    eshdn6(etypdat.equot);
  }

  for (i = 0; i < NI; i++)
    b[i] = etypdat.equot[i];

  /* return flag for lost nonzero bits */
  return ((int)j);
}

/*
 * Return nonzero if internal format number is a NaN.
 */

int
eiisnan(USHORT *x)
{
  int i;

  if ((x[E] & 0x7fff) == 0x7fff) {
    for (i = M + 1; i < NI; i++) {
      if (x[i] != 0)
        return (1);
    }
  }
  return (0);
}

/*
 * Compare significands of numbers in internal format.  Guard words
 * are included in the comparison.
 * for the significands:
 * returns      +1 if a > b
 *               0 if a == b
 *              -1 if a < b
 */
int
ecmpm(USHORT *a, USHORT *b)
{
  int i;

  a += M; /* skip up to significand area */
  b += M;
  for (i = M; i < NI; i++) {
    if (*a++ != *b++)
      goto difrnt;
  }
  return (0);

difrnt:
  if (*(--a) > *(--b))
    return (1);
  else
    return (-1);
}

/*
 * Subtract significands ;      y - x replaces y
 */

void
esubm(USHORT *x, USHORT *y)
{
  UINT a;
  int i;
  unsigned int carry;

  x += NI - 1;
  y += NI - 1;
  carry = 0;
  for (i = M; i < NI; i++) {
    a = (UINT)(*y) - (UINT)(*x) - carry;
    if (a & 0x10000)
      carry = 1;
    else
      carry = 0;
    *y = a & 0xFFFF;
    --x;
    --y;
  }
}

/*
 * ;    Shift significand up by 8 bits
 */

void
eshup8(USHORT *x)
{
  int i;
  USHORT newbyt, oldbyt;

  x += NI - 1;
  oldbyt = 0;

  for (i = M; i < NI; i++) {
    newbyt = *x >> 8;
    *x <<= 8;
    *x &= 0xFFFF;
    *x |= oldbyt;
    oldbyt = newbyt;
    --x;
  }
}

/*
 * ;    Shift significand up by 16 bits
 */

void
eshup6(USHORT *x)
{
  int i;
  USHORT *p;

  p = x + M;
  x += M + 1;

  for (i = M; i < NI - 1; i++)
    *p++ = *x++;

  *p = 0;
}

/*
 * ;    Shift significand down by 8 bits
 */

void
eshdn8(USHORT *x)
{
  USHORT newbyt, oldbyt;
  int i;

  x += M;
  oldbyt = 0;
  for (i = M; i < NI; i++) {
    newbyt = *x << 8;
    newbyt &= 0xFFFF;
    *x >>= 8;
    *x |= oldbyt;
    oldbyt = newbyt;
    ++x;
  }
}

/*
 * ;    Shift significand down by 16 bits
 */

void
eshdn6(USHORT *x)
{
  int i;
  USHORT *p;

  x += NI - 1;
  p = x + 1;

  for (i = M; i < NI - 1; i++)
    *(--p) = *(--x);

  *(--p) = 0;
}

/*
 * Multiply significand of e-type number b by 16-bit quantity a, e-type
 * result to c.
 */

void
m16m(USHORT a, USHORT *b, USHORT *c)
{
  USHORT *pp;
  UINT carry;
  USHORT *ps;
  USHORT p[NI];
  UINT aa, m;
  int i;

  aa = a;
  pp = &p[NI - 2];
  *pp++ = 0;
  *pp = 0;
  ps = &b[NI - 1];

  for (i = M + 1; i < NI; i++) {
    if (*ps == 0) {
      --ps;
      --pp;
      *(pp - 1) = 0;
    } else {
      m = (UINT)aa * *ps--;
      carry = (m & 0xffff) + *pp;
      *pp-- = carry & 0xFFFF;
      carry = (carry >> 16) + (m >> 16) + *pp;
      *pp = carry & 0xFFFF;
      *(pp - 1) = (carry >> 16) & 0xFFFF;
    }
  }
  for (i = M; i < NI; i++)
    c[i] = p[i];
}

/*
 * Clear out internal format number, but don't touch the sign.
 */

void
ecleazs(USHORT *xi)
{
  int i;

  ++xi;
  for (i = 0; i < NI - 1; i++)
    *xi++ = 0;
}
