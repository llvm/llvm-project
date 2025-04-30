/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>

#ifdef PGI_BIG_ENDIAN
union ieee {
  double d;
  struct {
    unsigned int s : 1;
    unsigned int e : 11;
    unsigned int hm : 20;
    unsigned int lm : 32;
  } v;
  int i[2];
};
#else
union ieee {
  double d;
  struct {
    unsigned int lm : 32;
    unsigned int hm : 20;
    unsigned int e : 11;
    unsigned int s : 1;
  } v;
  int i[2];
};
#endif

#if defined(TARGET_OSX)
#include <string.h>
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

static void ui64toa(INT m[2], char *s, int n, int decpl)
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

static void manshftr(INT m[4], int n)
{
  int i;
  int j;
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

static void manshftl(register INT m[4], int n)
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

static void manadd(register INT m1[4], register INT m2[4])
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

static void manrnd(INT m[4], int bits)
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

static void manneg(register INT m[4])
{
  void manadd();
  static INT one[4] = {0L, 0L, 0L, 1L};
  register int i;
  for (i = 0; i < 4; i++)
    m[i] = ~m[i];
  manadd(m, one);
}

static void manmul(register INT m1[4], register INT m2[4])
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

static void ufpnorm(register UFP *u)
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

static int ufpdnorm(UFP *u, int bias)
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

static void ufprnd(UFP *u, int bits)
{
  void ufpnorm();
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

static void ufpxten(UFP *u, int exp)
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

static void ufptosci(UFP *u, char *s, int dp, int *decpt, int *sign)
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

static void dtoufp(IEEE64 d, register UFP *u)
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

static void ufptod(register UFP *u, IEEE64 *r)
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
    errno = ERANGE;
  }
  if (u->fval == INFIN || u->fval == BIG || u->fval == DIVZ) {
    u->fexp = 1024;
    u->fman[0] = 0L;
    u->fman[1] = 0L;
    errno = ERANGE;
  }
  if (u->fval == NORMAL && u->fexp <= -1023) {
    if (ufpdnorm(u, 1022) < 0) {
      u->fval = NIL;
      errno = ERANGE;
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

static int atoxi(register char *s, INT *i, int n, int base)
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

static void atoui64(char *s, INT m[2], int n, INT *exp)
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

static void atoxufp(char *s, UFP *u, char **p)
{
  void atoui64();
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
  if (*s != 'd' && *s != 'D' && *s != 'e' && *s != 'E') {
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
  err = atoxi(start, &exp, s - start, 10);
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

#if defined(PGI_FPCVT)
double atof(char *s)
{
  double strtod();
  int save_errno;
  double d;

  save_errno = errno;
  d = strtod(s, (char **)0);
  errno = save_errno;
  return d;
}

double __strtod(char *s, char **p)
{
  IEEE64 d;
  void atoxufp();
  void ufpxten();
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

double strtod(char *s, char **p)
{
  return __strtod(s, p);
}
#endif

/*
 * Convert to ndigit digits.  *decpt is position of decimal point
 * (0 is before first digit).  *sign is sign.
 */
#define NDIG 25

#if defined(PGI_FPCVT) || defined(INTERIX86)
char *ecvt(double value, int ndigit, int *decpt, int *sign)
{
  char *__ecvt();

  return __ecvt(value, ndigit, decpt, sign);
}
#endif

#ifndef USE_NATIVE_ECVT
static char *
pgio_ecvt(double value, int ndigit, int *decpt, int *sign)
{
  static char ebuf[40];
  static char fmt[16];
  char *p;
  char *s;
  int i;
  int es, exp;

  sprintf(fmt, "%%30.%dE", ndigit - 1);
  sprintf(ebuf, fmt, value);
  *sign = 0;
  for (p = ebuf; *p; p++) {
    switch (*p) {
    case ' ':
      continue;
    case '-':
      *sign = 1;
      continue;
    case '0':
      goto ret0;
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      goto num;
    default:
      /* can't happen? */
      break;
    }
  }

num:
  i = 1;
  ebuf[0] = *p;
  s = &ebuf[1];
  while (1) {
    p++;
    if (*p == '.')
      continue;
    if (*p == 'E') {
      p++;
      if (*p == '+')
        es = 1;
      else
        es = -1;
      p++;
      exp = (*p - '0');
      p++;
      if (!*p)
        break;
      exp = exp * 10 + (*p - '0');
      p++;
      if (!*p)
        break;
      exp = exp * 10 + (*p - '0');
      p++;
      break;
    }
    *s++ = *p;
    i++;
  }
  exp *= es;
  exp++;
  *decpt = exp;
  for (; i < ndigit; i++)
    *s++ = '\0';
  ebuf[ndigit] = '\0';
  return ebuf;
ret0:
  for (i = 0; i < ndigit; i++) {
    ebuf[i] = '0';
  }
  ebuf[ndigit] = '\0';
  *decpt = 0;
  return ebuf;
}
#endif

char *__ecvt(double value, int ndigit, int *decpt, int *sign)
{
  static char buf[64];
  char *s;
  UFP u;
  int i, j, carry, n;
  union ieee ieee_v;

  n = ndigit;
  if (ndigit > 17)
    n = 17;
  s = buf;
  *s = '\0';
#if defined(PGI_FPCVT) || defined(INTERIX86)
  if (ndigit > 17)
    n = 17;
  dtoufp(value, &u);
  ufptosci(&u, s, 17, decpt, sign);
  if (!isdigit(*s)) {
    strcpy(buf, s);
    return buf;
  }
  if (value == 0.0) {
    strcpy(buf, s);
    return buf;
  }
  /* round */
  ++*decpt;
  j = 0;
  if (s[0] == '0')
    j = 1;
  i = j + n;
  if (s[i] >= '5') {
    carry = 1;
    while (--i >= j) {
      s[i] += carry;
      if (s[i] > '9') {
        carry = 1;
        s[i] = '0';
      } else
        carry = 0;
    }
    if (carry)
      s[0] = '1';
  }
  j = 0;
  if (s[0] == '0') {
    j = 1;
  } else {
    ++*decpt;
  }
  for (i = 0; i < n; ++i)
    s[i] = s[j++];
  for (; i < ndigit; ++i)
    s[i] = '0';
  s[i] = 0;
#else
  {
    ieee_v.d = value;
    u.fval = NORMAL;
    u.fexp = ieee_v.v.e - 1023;
    if (IEEE64_SUBNORMAL(ieee_v)) {
      u.fval = SUBNORMAL;
    } else if (u.fexp == 1024) {
      if (ieee_v.v.hm == 0 && ieee_v.v.lm == 0)
        u.fval = INFIN;
      else
        u.fval = NAN;
    } else if (u.fexp == -1023) {
      u.fval = ZERO;
    }
    if (u.fval == NAN) {
      strcpy(buf, "NaN");
      *sign = 0;
      *decpt = 0;
      return buf;
    }
    if (u.fval == INFIN) {
      strcpy(buf, "Inf");
      *sign = ieee_v.v.s;
      *decpt = 0;
      return buf;
    }
#if defined(USE_NATIVE_ECVT)
    extern char *ecvt(double, int, int *, int *);
    s = ecvt(value, ndigit, decpt, sign);
#else
    s = pgio_ecvt(value, ndigit, decpt, sign);
#endif
    strcpy(buf, s);
  }
#endif
  return buf;
}

#if defined(PGI_FPCVT)
char *fcvt(double value, int ndigit, int *decpt, int *sign)
{
  char *__fcvt();

  return __fcvt(value, ndigit, decpt, sign);
}
#endif

char *__fcvt(double v, int prec, int *decpt, int *sign)
{
  char *__ecvt();
  static char tmp[512];
  char *sfx;
  char *digits;
  char *p;
  char *pfx;
  int pfxn;
  int n;
  int lzfd; /* leading zero fractional digits */
  int i, j;

  digits = __ecvt(v, 16, decpt, sign);
  i = *decpt;

  if (!isdigit(*digits)) {
    return digits;
  }

  sfx = tmp + 1; /* +1 for rounding */
  p = sfx;

  /* first put out digits before decimal point */
  if (i < 0)
    j = 0;
  else if (prec < 0) {
    j = i + prec;
    prec = 0;
  } else
    j = i;
  while (j > 0 && *digits != '\0') {
    *p++ = *digits++;
    --j;
  }

  /* now put out zeros after decpt */
  lzfd = (i < 0 ? -i : 0);

  while ((lzfd > 0) && (prec > 0)) {
    *p++ = '0';
    ++*decpt;
    lzfd--;
    prec--;
  }

  while ((*digits) && (prec > 0)) {/* remaining digits */
    *p++ = *digits++;
    prec--;
  }

  while (prec > 0) {
    *p++ = '0';
    prec--;
  }
  *p = 0;

  if (*digits >= '5') {/* try rounding (yuck) */
    while (1) {
      p--;
      if (p == tmp) {
        sfx = tmp;
        *sfx = '1';
        ++*decpt;
        break;
      }
      if (*p < '9') {
        *p += 1;
        break;
      } else {
        *p = '0';
      }
    }
  }

  i = *decpt;
  p = sfx;
  if (i > 0) {
    while (i > 0 && *p) {
      --i;
      ++p;
    }
    if (*p == 0) {
      while (i > 0) {
        *p++ = '0';
        --i;
      }
      *p = 0;
    }
  }
  if (*sfx == 0) {
    sfx[0] = '0';
    sfx[1] = 0;
  }

  return sfx;
}
